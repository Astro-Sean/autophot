"""Shared log formatting helpers for AutoPhOT.

One module owns every repeated layout: 80-column stage banners, the
70-column sub-block rule, aligned tables, hanging-indent wrapping,
once-per-config deduplication, debug-dump files for large data blocks,
and the per-run registry behind the end-of-run warnings recap and
stage timing table.

Standard library only -- log lines must stay plain text and portable.
"""

import logging
import os
import re
import textwrap
import time

LOG_WIDTH = 80
SUB_WIDTH = 70
# Table width is the formatter's effective content width, not the 100-col
# hard limit: hanging indents already consume ~10 columns.
TABLE_MAX_WIDTH = 78
# "STATUS - " / "WARNING - " style prefix; continuation lines hang under
# the first message character.
WARNING_STATUS = 25  # numeric STATUS level (registered in functions.py)

# Credential-shaped keys whose values must never reach a log file.  The
# pattern matches dict-repr dumps ('MASTcasjobs_pwd': 'x'), key=value
# pairs, and "key: value" lines alike.
_SECRET_KEY_RE = re.compile(
    r"(?i)([\w]*(?:pwd|passwd|password|token|secret|apikey|api_key|auth)[\w]*"
    r"['\"]?\s*[:=]\s*)('[^']*'|\"[^\"]*\"|\S+)"
)


def scrub_secrets(text: str) -> str:
    """Mask credential values in rendered log text.

    Config dicts carry MAST/api keys next to ordinary numbers; a single
    ``str(cfg)`` in a log call would write them to disk.  Formatters run
    this on every record so the mask cannot be bypassed by a call site.
    """
    return _SECRET_KEY_RE.sub(lambda m: m.group(1) + "***", str(text))


_STAGE_TAGS = {
    "psf": "PSF",
    "zeropoint": "ZP",
    "catalog": "ZP",
    "filter_transformations": "ZP",
    "wcs": "WCS",
    "limits": "LIMMAG",
    "background": "MASK",
    "cosmic": "MASK",
    "fwhm": "DET",
    "run_sex": "DET",
    "templates": "SUB",
    "run_sfft": "SUB",
    "sfft": "SUB",
    "aperture": "PHOT",
    "telescope": "TEL",
    "driver": "PIPE",
    "main": "PIPE",
    "prepare": "PIPE",
}


def _clean(text) -> str:
    return " ".join(str(text).split())


def banner(title, width: int = LOG_WIDTH) -> str:
    """Major stage banner: 80 '=' rules with the title between them."""
    t = _clean(title)
    width = int(width)
    if len(t) > width:
        t = t[: width - 3] + "..."
    return "\n".join(("=" * width, t, "=" * width))


def subheader(title, width: int = SUB_WIDTH) -> str:
    """Sub-block rule: 70 '-' rules with the title between them."""
    t = _clean(title)
    width = int(width)
    if len(t) > width:
        t = t[: width - 3] + "..."
    return "\n".join(("-" * width, t, "-" * width))


def _auto_align(headers, rows):
    """Right-align columns whose cells are all numeric."""
    aligns = []
    for ci in range(len(headers)):
        is_num = True
        saw = False
        for r in rows:
            try:
                v = _clean(r[ci])
            except IndexError:
                continue
            if not v:
                continue
            saw = True
            try:
                float(v.rstrip("%"))
            except ValueError:
                is_num = False
                break
        aligns.append("r" if (saw and is_num) else "l")
    return aligns


def table(headers, rows, aligns=None, max_width: int = TABLE_MAX_WIDTH) -> str:
    """Render an aligned text table.

    Header row, '-' separator, right-aligned numeric columns, blank
    lines framing the block (added by callers via the log record).
    Rows with more cells than headers are truncated; missing cells are
    blank.  If the natural width exceeds ``max_width`` the widest
    column is shrunk and its cells wrapped onto continuation lines so
    no information is lost.
    """
    headers = [_clean(h) for h in headers]
    body = [[_clean(c) for c in r] for r in rows]
    n_cols = len(headers)
    if n_cols == 0:
        return ""
    body = [(r + [""] * n_cols)[:n_cols] for r in body]
    if aligns is None:
        aligns = _auto_align(headers, body)
    aligns = (list(aligns) + ["l"] * n_cols)[:n_cols]

    widths = [len(h) for h in headers]
    for r in body:
        for ci, c in enumerate(r):
            widths[ci] = max(widths[ci], len(c))

    budget = int(max_width) - 2  # rows are emitted with a 2-col indent
    natural = sum(widths) + 2 * (n_cols - 1)
    wrap_col = None
    if natural > budget:
        widest = max(range(n_cols), key=lambda i: widths[i])
        spare = budget - (natural - widths[widest])
        widths[widest] = max(8, spare)
        wrap_col = widest

    def _fmt(cells):
        return (
            "  "
            + "  ".join(
                (c.rjust(w) if a == "r" else c.ljust(w))
                for c, w, a in zip(cells, widths, aligns)
            ).rstrip()
        )

    lines = [_fmt(headers), "  " + "  ".join("-" * w for w in widths)]
    if not body:
        lines.append("  (none)")
        return "\n".join(lines)
    for r in body:
        cells = list(r)
        extra = []
        if wrap_col is not None and len(cells[wrap_col]) > widths[wrap_col]:
            wrapped = textwrap.wrap(
                cells[wrap_col],
                widths[wrap_col],
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            cells[wrap_col], extra = wrapped[0], wrapped[1:]
        lines.append(_fmt(cells))
        for cont in extra:
            cells = [""] * n_cols
            cells[wrap_col] = cont
            lines.append(_fmt(cells))
    return "\n".join(lines)


def wrap_message(text, level: str = "INFO", width: int = TABLE_MAX_WIDTH) -> str:
    """Wrap one logical line so continuations hang under the first
    character after the ``LEVEL - `` prefix."""
    prefix = f"{level} - "
    body_width = max(40, int(width) - len(prefix))
    out = []
    for seg in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(seg, body_width) or [""]
        out.append(wrapped[0])
        out.extend(" " * len(prefix) + ln for ln in wrapped[1:])
    return "\n".join(out)


# --------------------------------------------------------------------------
# Once-per-config deduplication
# --------------------------------------------------------------------------

_dedup_seen = {}
_dedup_suppressed = {}


def dedup(key, namespace: str = "") -> bool:
    """True the first time ``key`` is seen in ``namespace``.

    Used so repeated setup blocks (SExtractor saturation, convolution
    kernel, per-pass background diagnostics) print once per distinct
    configuration instead of once per call.
    """
    bucket = _dedup_seen.setdefault(namespace, set())
    if key in bucket:
        _dedup_suppressed[namespace] = _dedup_suppressed.get(namespace, 0) + 1
        return False
    bucket.add(key)
    return True


def dedup_suppressed(namespace: str = "") -> int:
    """Repeat count suppressed in ``namespace`` since the last reset."""
    return _dedup_suppressed.get(namespace, 0)


def dedup_reset(namespace=None) -> None:
    if namespace is None:
        _dedup_seen.clear()
        _dedup_suppressed.clear()
    else:
        _dedup_seen.pop(namespace, None)
        _dedup_suppressed.pop(namespace, None)


# --------------------------------------------------------------------------
# Debug dump files
# --------------------------------------------------------------------------


def _main_log_path() -> str:
    for h in logging.getLogger("").handlers:
        if isinstance(h, logging.FileHandler):
            return getattr(h, "baseFilename", "")
    return ""


def write_debug_dump(title, body) -> str:
    """Append ``body`` to ``LOG_<base>_debug_dump.log`` beside the main log.

    Returns the dump file's basename for the one-line pointer, or an
    empty string when no file handler exists (e.g. unit tests).
    """
    main_log = _main_log_path()
    if not main_log:
        return ""
    root, ext = os.path.splitext(main_log)
    dump_path = root + "_debug_dump" + (ext or ".log")
    try:
        with open(dump_path, "a", encoding="utf-8") as fh:
            fh.write(subheader(str(title)))
            fh.write("\n")
            fh.write(scrub_secrets(str(body).rstrip("\n")))
            fh.write("\n\n")
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Could not write debug dump %s: %s", dump_path, exc
        )
        return ""
    return os.path.basename(dump_path)


# --------------------------------------------------------------------------
# Per-run registry: stages, warnings, timings
# --------------------------------------------------------------------------


class _RunState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.time()
        self.stage_name = None
        self.last_stage = None
        self.stage_t0 = self.t0
        self.stage_warn = 0
        # (name, seconds) for finished stages; last stage closes at
        # summary time via open_stage.
        self.stages = []
        # (tag, stage, first_line, count) - identical messages merge.
        self.warnings = []

    def record_warning(self, tag, message):
        stage = self.stage_name or self.last_stage or "setup"
        self.stage_warn += 1
        key = (tag, stage, message)
        for i, (t, s, m, c) in enumerate(self.warnings):
            if (t, s, m) == key:
                self.warnings[i] = (t, s, m, c + 1)
                return
        self.warnings.append((tag, stage, message, 1))

    def close_stage(self):
        if self.stage_name is not None:
            self.stages.append((self.stage_name, time.time() - self.stage_t0))
            self.last_stage = self.stage_name
        self.stage_name = None
        self.stage_warn = 0


_RUN = _RunState()


def reset_run_state() -> None:
    """Call once per image run, right after logging is configured."""
    _RUN.reset()
    dedup_reset()


def begin_stage(name) -> None:
    """Emit the stage banner at STATUS and open the stage in the registry."""
    _RUN.close_stage()
    _RUN.stage_name = str(name)
    _RUN.stage_t0 = time.time()
    _RUN.stage_warn = 0
    logging.getLogger(__name__).log(WARNING_STATUS, banner(name))


def end_stage(detail: str = "", *args) -> None:
    """Close the stage with one ``[OK|WARN]`` status line + elapsed time.

    ``[WARN]`` is emitted when any WARNING was recorded while the stage
    was open.  ``detail`` keeps the existing field order: counts, key
    measured value; the elapsed seconds are appended last.
    """
    status = "WARN" if _RUN.stage_warn else "OK"
    elapsed = time.time() - _RUN.stage_t0
    name = _RUN.stage_name
    _RUN.close_stage()
    if detail:
        msg = str(detail) % args if args else str(detail)
        line = f"[{status}] {msg}; {elapsed:.1f} s"
    else:
        line = f"[{status}] {name or 'stage'} complete; {elapsed:.1f} s"
    logging.getLogger(__name__).log(WARNING_STATUS, line, extra={"result": True})


def _kv_lines(pairs, width: int = LOG_WIDTH):
    """label ..... value rows matching the ascii_kv house style."""
    rows = [(str(k).rstrip(), str(v)) for k, v in pairs if str(v)]
    if not rows:
        return []
    label_w = max(len(k) for k, _ in rows)
    val_col = min(label_w + 6, width - 8)
    out = []
    for k, v in rows:
        dots = "." * max(3, val_col - 2 - len(k))
        lead = f"  {k} {dots} "
        segs = textwrap.wrap(v, max(10, width - len(lead))) or [""]
        out.append(lead + segs[0])
        out.extend(" " * len(lead) + s for s in segs[1:])
    return out


def run_header(version, input_name, started, config_pairs=None) -> str:
    """Run-opening block: version, input, start time, key config."""
    pairs = [
        ("Version", f"v{version}"),
        ("Input", str(input_name)),
        ("Started", str(started)),
    ]
    for k, v in config_pairs or []:
        pairs.append((str(k), scrub_secrets(str(v))))
    lines = ["=" * LOG_WIDTH, "RUN", "=" * LOG_WIDTH]
    lines += _kv_lines(pairs)
    lines.append("=" * LOG_WIDTH)
    return "\n".join(lines)


def warnings_summary_lines():
    """'WARNINGS (N):' plus one numbered line per distinct warning."""
    if not _RUN.warnings:
        return []
    total = sum(c for _, _, _, c in _RUN.warnings)
    lines = ["", f"  WARNINGS ({total}):"]
    for i, (tag, stage, msg, count) in enumerate(_RUN.warnings, 1):
        suffix = f" (x{count})" if count > 1 else ""
        lines.append(f"  {i:>2}. [{tag}] {msg}{suffix}  -- {stage}")
    return lines


def timings_summary_lines(total_s: float):
    """'STAGE TIMINGS:' table: stage, seconds, percent of total."""
    _RUN.close_stage()
    entries = list(_RUN.stages)
    # Time before the first banner and after the last stage closes is
    # folded into adjacent stages; report the wall-clock total only.
    rows = [
        (name, f"{secs:10.1f}", f"{100.0 * secs / max(total_s, 1e-9):5.1f}%")
        for name, secs in entries
    ]
    rows.append(("TOTAL", f"{total_s:10.1f}", "100.0%"))
    lines = [
        "",
        "  STAGE TIMINGS:",
        table(("stage", "seconds", "%"), rows, max_width=LOG_WIDTH - 4),
    ]
    return lines


def final_summary(finished, time_taken_s, usage, warnings_stage=None) -> str:
    """End-of-run box: result tokens, warnings recap, stage timings."""
    lines = ["=" * LOG_WIDTH, "RUN SUMMARY", "=" * LOG_WIDTH]
    lines += _kv_lines(
        [
            ("Photometry finished", str(finished)),
            ("Time Taken", f"{time_taken_s:.1f} s"),
            ("Usage", str(usage)),
        ]
    )
    lines += warnings_summary_lines()
    lines += timings_summary_lines(float(time_taken_s))
    lines.append("=" * LOG_WIDTH)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Warning tagging filter
# --------------------------------------------------------------------------

_BRACKETED = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9_ ]{0,11})\]")


def _tag_for(record: logging.LogRecord) -> str:
    """Stable grep tag derived from the emitting module."""
    name = getattr(record, "name", "") or ""
    mod = name.rsplit(".", 1)[-1]
    return _STAGE_TAGS.get(mod, "PIPE")


class WarningTagger(logging.Filter):
    """Attach to handlers: prefixes ``[TAG]`` on WARNING records and
    registers them for the end-of-run recap.

    Records that already start with a bracket tag keep it.  Tagging
    mutates the record once; every handler sees the same one-line
    message.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        if getattr(record, "_warn_tagged", False):
            return True
        try:
            msg = record.getMessage()
        except (TypeError, ValueError):
            return True
        m = _BRACKETED.match(msg)
        if m:
            tag = m.group(1).strip().upper()
            body = msg[m.end() :].lstrip()
        else:
            tag = _tag_for(record)
            body = msg
        first_line = " ".join(str(body).splitlines())
        if len(first_line) > 160:
            first_line = first_line[:160].rsplit(" ", 1)[0].rstrip(",;") + " ..."
        _RUN.record_warning(tag, first_line.strip())
        record.msg = f"[{tag}] {body}"
        record.args = None
        record._warn_tagged = True
        return True


def install_warning_tagger() -> int:
    """Attach the tagger to every root handler; returns count attached."""
    tagger = WarningTagger()
    n = 0
    for h in logging.getLogger("").handlers:
        h.addFilter(tagger)
        n += 1
    return n
