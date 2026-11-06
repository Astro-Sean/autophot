"""
Paced access to the Gaia archive and batched GaiaXPy calls.

ESA recommends spacing consecutive heavy queries (e.g. sleep(1)) to reduce
load and failures when the archive is under stress. This module implements:

- Optional pause before/after ADQL jobs
- Batched ``calibrate`` / ``generate`` with pauses between batches
- Retries with exponential backoff on transient failures
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm_bar
except ImportError:  # pragma: no cover
    _tqdm_bar = None


def _chunk_progress(
    chunks: List[List[str]],
    *,
    desc: str,
    unit: str,
    show_progress: bool,
) -> Union[List[List[str]], Any]:
    """Wrap chunk list with tqdm when available and requested."""
    if not show_progress or not chunks or _tqdm_bar is None:
        return chunks
    return _tqdm_bar(chunks, desc=desc, unit=unit, total=len(chunks))


T = TypeVar("T")

# Loggers that often print the long Gaia "archive is unstable" INFO banner.
_QUIET_GAIA_LOGGERS = (
    "astroquery.gaia",
    "astroquery.utils.tap",
    "astroquery.utils",
    "gaiaxpy",
    "gaia_xpy",
    "astropy.io.votable",
)

# Anonymous GaiaXPy load_data requests are capped at 50 source IDs per
# call; a larger chunk returns HTTP 500 "Maximum number of Source Ids
# reached".  Logged-in accounts get a higher cap, but these calls are
# always anonymous.
_GAIAXPY_MAX_SOURCE_IDS_PER_CALL = 50

# astroquery / GaiaXPy status chatter written via plain print().  Lines
# starting with these prefixes are dropped while a Gaia call runs.
_GAIA_PRINT_DROP_PREFIXES = (
    "running query",
    "launched uws job",
    "query finished",
    "retrieving tables",
    "parsing tables",
    "reading",
)

# Self-contained multi-line banners; the whole write is dropped when any
# fragment appears (they arrive as one print call).
_GAIA_PRINT_DROP_SUBSTRINGS = (
    "archive release",
    "cosmos.esa.int/web/gaia",
    "gaia helpdesk",
    "in case of questions or feedback",
    "archive is unstable",
    "workaround solutions",
)

# UWS error dumps arrive as one multi-line print; collapse them to their
# informative header instead of flooding the console with the job XML.
_GAIA_PRINT_DUMP_MARKERS = (
    "cannot process request",
    "uwsjobowner",
    "reqid:",
)


def _gaia_dump_digest(text: str) -> str:
    """One-line digest of a TAP/UWS error dump."""
    keep = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " | ".join(keep[:2])[:300]


class _FilterGaiaPrintSpam:
    """Drop or compact Gaia/astroquery print() noise on a stream.

    astroquery's TapConn reports status and error bodies with plain
    ``print()`` rather than logging, so they bypass logger-level filters
    and hit whichever stream print targets.
    """

    __slots__ = ("_real", "_swallow_eol")

    def __init__(self, real: Any) -> None:
        self._real = real
        self._swallow_eol = False

    def write(self, s: str) -> int:
        if not s:
            return 0
        # print() issues the trailing newline as a second write; after a
        # dropped line it would otherwise leak a blank line.
        if self._swallow_eol:
            if not s.strip():
                return len(s)
            self._swallow_eol = False
        low = s.lower()
        lead = low.lstrip("\r\n\t ")
        if any(lead.startswith(p) for p in _GAIA_PRINT_DROP_PREFIXES):
            self._swallow_eol = True
            return len(s)
        if any(m in low for m in _GAIA_PRINT_DUMP_MARKERS):
            return self._real.write(_gaia_dump_digest(s) + "\n")
        if any(p in low for p in _GAIA_PRINT_DROP_SUBSTRINGS):
            self._swallow_eol = True
            return len(s)
        return self._real.write(s)

    def flush(self) -> Any:
        return self._real.flush()

    def isatty(self) -> bool:
        return getattr(self._real, "isatty", lambda: False)()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _FilterGaiaStderrSpam(_FilterGaiaPrintSpam):
    """Same noise dropped on stderr."""


@contextlib.contextmanager
def quiet_gaia_tap_info_logs(enable: bool = True) -> Iterator[None]:
    """
    Reduce repetitive Gaia archive INFO banners and stderr spam during TAP /
    GaiaXPy calls (logging + common print patterns).

    Set environment ``AUTOPHOT_VERBOSE_GAIA_TAP=1`` to disable this and see
    full Gaia / astroquery messages.
    """
    if not enable:
        yield
        return
    env = os.environ.get("AUTOPHOT_VERBOSE_GAIA_TAP", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        yield
        return

    import sys

    saved_err: Any = sys.stderr
    saved_out: Any = sys.stdout
    sys.stderr = _FilterGaiaStderrSpam(saved_err)
    sys.stdout = _FilterGaiaPrintSpam(saved_out)

    saved: List[tuple[logging.Logger, int]] = []
    for name in _QUIET_GAIA_LOGGERS:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level))
        if lg.getEffectiveLevel() <= logging.INFO:
            lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        sys.stderr = saved_err
        sys.stdout = saved_out
        for lg, lvl in saved:
            lg.setLevel(lvl)


def _sleep_sec(seconds: float, reason: str = "") -> None:
    if seconds is None or seconds <= 0:
        return
    time.sleep(float(seconds))


def _is_permanent_http_error(exc: BaseException) -> bool:
    """True for HTTP 4xx other than 408/429 - retrying a bad request cannot help."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status is None:
        return False
    status = int(status)
    return 400 <= status < 500 and status not in (408, 429)


def _exc_digest(exc: BaseException, limit: int = 200) -> str:
    """First informative line(s) of an exception, for one-line log output."""
    text = str(exc)
    if "\n" in text:
        text = _gaia_dump_digest(text)
    else:
        text = text.strip()
    return text[:limit]


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay_sec: float,
    logger: Optional[logging.Logger],
    op_name: str,
) -> T:
    """Run ``fn``; on failure retry with exponential backoff."""
    last_exc: Optional[BaseException] = None
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or _is_permanent_http_error(exc):
                break
            delay = float(base_delay_sec) * (2**attempt)
            if logger is not None:
                logger.warning(
                    "%s failed (%s); retry %d/%d in %.1fs: %s",
                    op_name,
                    type(exc).__name__,
                    attempt + 1,
                    attempts,
                    delay,
                    _exc_digest(exc),
                )
            _sleep_sec(delay)
    assert last_exc is not None
    raise last_exc


def launch_gaia_adql_to_pandas(
    query: str,
    *,
    pause_before_sec: float = 0.25,
    pause_after_sec: float = 0.25,
    max_retries: int = 3,
    retry_base_delay_sec: float = 2.0,
    logger: Optional[logging.Logger] = None,
    op_name: str = "Gaia ADQL",
    quiet_tap_info_logs: bool = True,
) -> pd.DataFrame:
    """Run a synchronous Gaia ``launch_job`` query and return a DataFrame."""

    from astroquery.gaia import Gaia

    def _once() -> pd.DataFrame:
        _sleep_sec(pause_before_sec)
        with quiet_gaia_tap_info_logs(quiet_tap_info_logs):
            job = Gaia.launch_job(query, dump_to_file=False)
            out = job.get_results().to_pandas()
        _sleep_sec(pause_after_sec)
        return out

    return retry_with_backoff(
        _once,
        max_retries=max_retries,
        base_delay_sec=retry_base_delay_sec,
        logger=logger,
        op_name=op_name,
    )


def _chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _gaiaxpy_chunk_size(
    batch_size: int,
    n_ids: int,
    logger: Optional[logging.Logger],
) -> int:
    """Effective sources-per-GaiaXPy-call, capped at the archive limit.

    ``batch_size <= 0`` means a single call; both cases still clamp to
    the archive's 50-ID request cap whenever the list is larger.
    """
    eff = batch_size if batch_size > 0 else max(n_ids, 1)
    if eff > _GAIAXPY_MAX_SOURCE_IDS_PER_CALL:
        if logger is not None:
            logger.info(
                "GaiaXPy archive requests allow at most %d source IDs "
                "per call; reducing batch_size from %s to %d.",
                _GAIAXPY_MAX_SOURCE_IDS_PER_CALL,
                batch_size if batch_size > 0 else "all",
                _GAIAXPY_MAX_SOURCE_IDS_PER_CALL,
            )
        eff = _GAIAXPY_MAX_SOURCE_IDS_PER_CALL
    return eff


def _table_and_sampling_from_calibrate_output(obj: Any) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Split GaiaXPy ``calibrate()`` return value into (table_like, wavelength_grid_nm).

    GaiaXPy returns ``(spectra_table, sampling)`` where ``sampling`` is a 1-D array
    of wavelength points (nm); the table rows then hold per-source ``flux`` arrays
    aligned to that grid (often without a repeated wavelength column per row).
    """
    sampling: Optional[np.ndarray] = None
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        cand = obj[1]
        try:
            from astropy.units import Quantity  # type: ignore

            if isinstance(cand, Quantity):
                cand = cand.to("nm").value
        except Exception:
            pass
        arr = np.asarray(cand, dtype=float)
        if arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr)):
            sampling = arr
            obj = obj[0]
    return obj, sampling


def _ensure_dataframe(obj: Any) -> pd.DataFrame:
    """
    Normalize GaiaXPy ``calibrate`` / ``generate`` outputs to a pandas DataFrame.

    Recent GaiaXPy returns ``(spectra_table_or_df, sampling_array)``; older code
    paths may return a single DataFrame or Astropy ``Table``.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas") and not isinstance(obj, (tuple, list)):
        return obj.to_pandas()
    # Tuple/list: first element is almost always the per-source table.
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        first = obj[0]
        if isinstance(first, pd.DataFrame):
            return first
        if first is not None and hasattr(first, "to_pandas"):
            return first.to_pandas()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    try:
        return pd.DataFrame(obj)
    except (ValueError, TypeError) as err:
        raise TypeError(
            f"Cannot convert GaiaXPy result to DataFrame (got {type(obj)!r}). "
            "Expected a DataFrame, Table, or (table, sampling) tuple."
        ) from err


def calibrate_source_ids_batched(
    source_ids: List[str],
    *,
    batch_size: int = 200,
    inter_batch_pause_sec: float = 0.5,
    max_retries: int = 3,
    retry_base_delay_sec: float = 2.0,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
    quiet_tap_info_logs: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Call ``gaiaxpy.calibrate`` in batches with pauses between batches.

    Returns
    -------
    spectra_table, sampling_nm
        Concatenated calibrated spectra and the wavelength grid (nm) shared by
        GaiaXPy for that run. ``sampling_nm`` is ``None`` only for empty input or
        if the installed GaiaXPy returns a legacy single-table object with no
        separate sampling array.

    Parameters
    ----------
    batch_size :
        Max sources per GaiaXPy call. Use <= 0 for a single call (not recommended
        for large lists).
    """
    from gaiaxpy import calibrate

    if not source_ids:
        return pd.DataFrame(), None

    chunks = _chunk_list(
        list(source_ids),
        _gaiaxpy_chunk_size(batch_size, len(source_ids), logger),
    )
    frames: List[pd.DataFrame] = []
    sampling_ref: Optional[np.ndarray] = None
    to_iterate = _chunk_progress(
        chunks,
        desc="GaiaXPy calibrate",
        unit="batch",
        show_progress=show_progress,
    )
    for idx, chunk in enumerate(to_iterate):
        if idx > 0:
            _sleep_sec(inter_batch_pause_sec, "between GaiaXPy calibrate batches")

        def _once() -> Any:
            with quiet_gaia_tap_info_logs(quiet_tap_info_logs):
                # save_file=False keeps GaiaXPy from writing output_spectra.*
                # to cwd on every batch call.
                try:
                    return calibrate(chunk, save_file=False)
                except TypeError:
                    return calibrate(chunk)

        part = retry_with_backoff(
            _once,
            max_retries=max_retries,
            base_delay_sec=retry_base_delay_sec,
            logger=logger,
            op_name=f"GaiaXPy calibrate batch {idx + 1}/{len(chunks)}",
        )
        table_like, samp = _table_and_sampling_from_calibrate_output(part)
        frames.append(_ensure_dataframe(table_like))
        if samp is not None:
            if sampling_ref is None:
                sampling_ref = samp
            elif sampling_ref.shape != samp.shape or not np.allclose(
                sampling_ref, samp, rtol=0.0, atol=1e-6
            ):
                if logger is not None:
                    logger.warning(
                        "GaiaXPy calibrate wavelength sampling differed between batches; using the grid from the first batch."
                    )

    if not frames:
        return pd.DataFrame(), sampling_ref
    return pd.concat(frames, ignore_index=True), sampling_ref


def generate_source_ids_batched(
    source_ids: List[str],
    photometric_system: Any,
    *,
    batch_size: int = 200,
    inter_batch_pause_sec: float = 0.5,
    max_retries: int = 3,
    retry_base_delay_sec: float = 2.0,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
    quiet_tap_info_logs: bool = True,
    error_correction: bool = False,
    truncation: bool = False,
) -> pd.DataFrame:
    """Call ``gaiaxpy.generate`` in batches with pauses between batches."""
    from gaiaxpy import generate

    if not source_ids:
        return pd.DataFrame()

    chunks = _chunk_list(
        list(source_ids),
        _gaiaxpy_chunk_size(batch_size, len(source_ids), logger),
    )
    frames: List[pd.DataFrame] = []
    to_iterate = _chunk_progress(
        chunks,
        desc="GaiaXPy generate",
        unit="batch",
        show_progress=show_progress,
    )
    for idx, chunk in enumerate(to_iterate):
        if idx > 0:
            _sleep_sec(inter_batch_pause_sec, "between GaiaXPy generate batches")

        def _once() -> pd.DataFrame:
            with quiet_gaia_tap_info_logs(quiet_tap_info_logs):
                # GaiaXPy supports optional error correction, and (>=2.1.4) truncation.
                # We pass these through when available; older versions may not accept
                # `truncation` and will raise TypeError, in which case we retry without it.
                # save_file=False keeps GaiaXPy from writing
                # output_synthetic_photometry.* to cwd on every batch call.
                try:
                    return generate(
                        chunk,
                        photometric_system=photometric_system,
                        error_correction=bool(error_correction),
                        truncation=bool(truncation),
                        save_file=False,
                    )
                except TypeError:
                    try:
                        return generate(
                            chunk,
                            photometric_system=photometric_system,
                            error_correction=bool(error_correction),
                            save_file=False,
                        )
                    except TypeError:
                        return generate(
                            chunk,
                            photometric_system=photometric_system,
                            error_correction=bool(error_correction),
                        )

        part = retry_with_backoff(
            _once,
            max_retries=max_retries,
            base_delay_sec=retry_base_delay_sec,
            logger=logger,
            op_name=f"GaiaXPy generate batch {idx + 1}/{len(chunks)}",
        )
        frames.append(_ensure_dataframe(part))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Gaia TAP ADQL does not accept DISTANCE(...) in ORDER BY (HTTP 400). For
# nearest-first selection we over-fetch with ORDER BY phot_g_mean_mag, then
# sort by on-sky separation in Python (Astropy).
_GAIA_XP_DISTANCE_MODES = frozenset({"distance", "angular", "nearest", "sky"})


def gaia_xp_uses_python_distance_sort(order_by: str) -> bool:
    return str(order_by or "brightness").strip().lower() in _GAIA_XP_DISTANCE_MODES


def gaia_xp_sql_top_n(
    max_sources: int,
    order_by: str,
    *,
    prefetch_factor: int = 50,
    prefetch_min: int = 200,
    prefetch_max: int = 10000,
) -> tuple[int, bool]:
    """
    Return (sql_top_n, trim_and_sort_by_distance_in_python).

    For ``distance`` / ``nearest`` modes, request more rows from Gaia than
    ``max_sources``, then sort by separation in Python and keep the nearest
    ``max_sources`` rows.
    """
    ms = max(1, int(max_sources))
    if gaia_xp_uses_python_distance_sort(order_by):
        factor = max(1, int(prefetch_factor))
        # Smaller final N -> cap multiplier so we do not over-fetch the archive
        # (e.g. 100 nearest does not need TOP 5000 by default). The caps are
        # non-decreasing in ms so a larger request never prefetches less.
        if ms <= 50:
            factor = min(factor, 5)
        elif ms <= 100:
            factor = min(factor, 8)
        elif ms <= 200:
            factor = min(factor, 10)
        elif ms <= 500:
            factor = min(factor, 15)
        prefetch = min(int(prefetch_max), max(int(prefetch_min), ms * factor))
        return prefetch, True
    return ms, False


def sort_gaia_table_nearest_to_target(
    df: pd.DataFrame,
    ra_deg: float,
    dec_deg: float,
    *,
    max_rows: int,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> pd.DataFrame:
    """Keep up to ``max_rows`` rows with smallest angular distance to the target."""
    if df.empty or max_rows <= 0:
        return df.iloc[:0].copy()
    # angular_separation applies the same Vincenty formula as
    # SkyCoord.separation for ICRS positions, without building frame objects.
    from astropy.coordinates import angular_separation

    sep_deg = np.degrees(
        angular_separation(
            np.radians(float(ra_deg)),
            np.radians(float(dec_deg)),
            np.radians(np.asarray(df[ra_col].values, dtype=float)),
            np.radians(np.asarray(df[dec_col].values, dtype=float)),
        )
    )
    out = df.copy()
    out["_autophot_sep_deg"] = sep_deg
    out = out.sort_values("_autophot_sep_deg", kind="mergesort").head(int(max_rows))
    return out.drop(columns=["_autophot_sep_deg"])


def gaia_xp_source_query(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    sql_top_n: int,
    *,
    include_bp_rp: bool = True,
) -> str:
    """
    ADQL for Gaia DR3 sources with XP continuous spectra in a cone.

    Always ``ORDER BY phot_g_mean_mag`` (Gaia TAP does not support ``DISTANCE``
    in ``ORDER BY``). For nearest-N selection, use a larger ``sql_top_n`` and
    :func:`sort_gaia_table_nearest_to_target` on the result.
    """
    extra = ", phot_bp_mean_mag, phot_rp_mean_mag" if include_bp_rp else ""
    top = max(1, int(sql_top_n))
    return f"""
    SELECT TOP {top}
           source_id, ra, dec, phot_g_mean_mag{extra}
    FROM gaiadr3.gaia_source AS gs
    WHERE has_xp_continuous = 'True'
      AND CONTAINS(
            POINT('ICRS', gs.ra, gs.dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
          ) = 1
      AND phot_g_mean_mag IS NOT NULL
    ORDER BY phot_g_mean_mag
    """


def gaia_cone_count(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    xp_only: bool = False,
    require_g_mag: bool = False,
    max_retries: int = 3,
    retry_base_delay_sec: float = 2.0,
    pause_before_sec: float = 0.25,
    pause_after_sec: float = 0.25,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """COUNT(*) of Gaia DR3 sources in a cone.

    With ``xp_only=True`` counts only sources carrying XP continuous
    spectra (the subset GaiaXPy can synthesize photometry from).
    ``require_g_mag=True`` additionally demands ``phot_g_mean_mag IS NOT
    NULL``, matching the WHERE clause of :func:`gaia_xp_source_query` so the
    count equals the number of rows a data query could return.
    Returns None when the count query fails -- diagnostics must not
    break the catalog build.
    """
    xp_clause = "has_xp_continuous = 'True' AND " if xp_only else ""
    gmag_clause = "phot_g_mean_mag IS NOT NULL AND " if require_g_mag else ""
    query = f"""
    SELECT COUNT(*) AS n
    FROM gaiadr3.gaia_source AS gs
    WHERE {xp_clause}{gmag_clause}CONTAINS(
            POINT('ICRS', gs.ra, gs.dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
          ) = 1
    """
    try:
        df = launch_gaia_adql_to_pandas(
            query,
            pause_before_sec=pause_before_sec,
            pause_after_sec=pause_after_sec,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            logger=logger,
            op_name="Gaia ADQL (cone count)",
        )
        if df is None or len(df) == 0:
            return None
        return int(df.iloc[0]["n"])
    except Exception:
        return None


def query_gaia_xp_cone_growing(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    sql_top_n: int,
    *,
    include_bp_rp: bool = True,
    min_sources: int = 0,
    max_radius_deg: Optional[float] = None,
    grow_factor: float = 1.5,
    pause_before_sec: float = 0.25,
    pause_after_sec: float = 0.25,
    max_retries: int = 3,
    retry_base_delay_sec: float = 2.0,
    logger: Optional[logging.Logger] = None,
    op_name: str = "Gaia ADQL (XP sources)",
) -> Tuple[pd.DataFrame, float]:
    """
    Query the XP-source cone, enlarging the radius when too few rows return.

    XP spectra exist only for the brighter Gaia subset, so sparse fields can
    leave an unusably small catalog at the requested radius. While the count
    stays below ``min_sources`` the cone is grown by ``grow_factor`` up to
    ``max_radius_deg``. Growth is disabled when ``min_sources <= 0``,
    ``grow_factor <= 1``, or ``max_radius_deg <= radius_deg``.

    After the first data fetch, trial radii are probed with cheap COUNT(*)
    queries instead of re-downloading up to ``sql_top_n`` rows per step; the
    full SELECT runs once more at the chosen radius. ``len(rows)`` equals
    ``min(count, sql_top_n)`` for a cone query, so the growth decision is the
    same as re-fetching at each radius.

    Returns ``(rows, used_radius_deg)``.
    """
    used_radius = float(radius_deg)
    cap = used_radius if max_radius_deg is None else float(max_radius_deg)
    cap = max(cap, used_radius)
    needed = max(0, int(min_sources))
    factor = float(grow_factor)
    top = max(1, int(sql_top_n))

    def _fetch() -> pd.DataFrame:
        return launch_gaia_adql_to_pandas(
            gaia_xp_source_query(
                ra_deg,
                dec_deg,
                used_radius,
                top,
                include_bp_rp=include_bp_rp,
            ),
            pause_before_sec=pause_before_sec,
            pause_after_sec=pause_after_sec,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            logger=logger,
            op_name=op_name,
        )

    df = pd.DataFrame()
    fetched = False
    counts_ok = True
    # The loop converges because factor > 1 drives used_radius to cap.
    for it in range(20):
        n_rows: Optional[int] = None
        # Iteration 0 is always a real fetch (dense fields then need only one
        # archive job); later trial radii are probed with COUNT(*) first.
        if needed > 0 and it > 0 and counts_ok:
            n_avail = gaia_cone_count(
                ra_deg,
                dec_deg,
                used_radius,
                xp_only=True,
                require_g_mag=True,
                max_retries=max_retries,
                retry_base_delay_sec=retry_base_delay_sec,
                pause_before_sec=pause_before_sec,
                pause_after_sec=pause_after_sec,
                logger=logger,
            )
            if n_avail is None:
                # A failed count should not grow the query load; fall back to
                # data fetches for the rest of the loop.
                counts_ok = False
            else:
                n_rows = min(int(n_avail), top)
                fetched = False
        if n_rows is None:
            df = _fetch()
            n_rows = len(df)
            fetched = True
        if n_rows >= needed or needed <= 0 or used_radius >= cap or factor <= 1.0:
            break
        new_radius = min(used_radius * factor, cap)
        if new_radius <= used_radius:
            break
        if logger is not None:
            logger.info(
                "Only %d Gaia XP-spectra sources within %.4f deg; expanding the cone to %.4f deg (cap %.4f deg).",
                n_rows,
                used_radius,
                new_radius,
                cap,
            )
        used_radius = new_radius

    if not fetched:
        df = _fetch()

    if used_radius > float(radius_deg) and logger is not None:
        logger.info(
            "Gaia XP cone expanded %.4f -> %.4f deg; %d sources available.",
            float(radius_deg),
            used_radius,
            len(df),
        )
    if needed > 0 and len(df) < needed and logger is not None:
        # Report the usable fraction explicitly: a sparse XP catalog is
        # almost always the has_xp_continuous gate (~5-15% of DR3 carry
        # XP spectra, biased bright), not the cone size -- DS9 shows the
        # full Gaia population.
        _n_all = gaia_cone_count(
            ra_deg,
            dec_deg,
            used_radius,
            xp_only=False,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            logger=logger,
        )
        _frac_msg = (
            f"of {_n_all} total Gaia sources in the cone, only {len(df)} "
            f"carry XP spectra"
            if _n_all is not None
            else f"only {len(df)} sources carry XP spectra"
        )
        logger.warning(
            "Only %d Gaia XP-spectra sources within the maximum %.4f-deg cone "
            "(wanted >= %d): %s. XP-spectra coverage is the limiter, not the "
            "cone; for sparse fields consider a deeper survey catalog (e.g. "
            "'refcat' or 'pan_starrs') for bands it covers natively.",
            len(df),
            used_radius,
            needed,
            _frac_msg,
        )
    return df, used_radius
