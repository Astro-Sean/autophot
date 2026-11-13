#!/usr/bin/env python3
"""Generate the LaTeX parameter-reference appendix from default_input.yml.

Parses ``autophot/data/default_input.yml`` textually (comments carry the
``# type --- description`` annotations that PyYAML discards) and emits one
``longtable`` per top-level config section, ready for ``\\input`` in the
methods document.

Usage:
    python scripts/generate_parameter_appendix.py \
        > docs/autophot_parameters.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
YML = REPO / "autophot" / "data" / "default_input.yml"

# Ordered presentation of the YAML sections (matches the file's own order).
SECTION_TITLES = {
    "default_input": "General and run-control parameters",
    "preprocessing": "Pre-processing",
    "photometry": "Photometry",
    "wcs": "World coordinate system",
    "catalog": "Reference catalogues",
    "zeropoint": "Photometric zero-point",
    "cosmic_rays": "Cosmic rays and defects",
    "source_detection": "Source detection and FWHM",
    "limiting_magnitude": "Limiting magnitudes",
    "host_model": "Host-galaxy modelling",
    "alignment": "Image reprojection",
    "template_subtraction": "Template subtraction",
    "background": "Background estimation",
    "postprocessing": "Post-processing",
}

# Thematic sub-grouping applied inside the (large) ``photometry`` section.
# First matching prefix wins; unmatched keys fall into "Other".
PHOTOMETRY_GROUPS = [
    ("detection", "Detection and forced-photometry decision"),
    ("upper_limit_snr", "Detection and forced-photometry decision"),
    ("force_photometry", "Detection and forced-photometry decision"),
    ("fitting_xy_bounds", "Centroid window of the PSF fit"),
    ("psf_group_separation", "Centroid window of the PSF fit"),
    ("centroid_max_wander", "Centroid window of the PSF fit"),
    ("aperture_", "Aperture photometry"),
    ("apply_aperture", "Aperture photometry"),
    ("plot_aperture", "Aperture photometry"),
    ("find_optimum", "Aperture photometry"),
    ("optimum_radius", "Aperture photometry"),
    ("annulus_", "Aperture photometry"),
    ("min_aperture", "Aperture photometry"),
    ("crowded_optimum", "Aperture photometry"),
    ("crowded_field", "Aperture photometry"),
    ("empirical_noise_factor", "Aperture photometry"),
    ("enforce_nonnegative", "Aperture photometry"),
    ("systematic_error_floor", "Aperture photometry"),
    ("apply_colorterms", "Instrumental colour terms"),
    ("color_term", "Instrumental colour terms"),
    ("lpi_", "Local background infill (LPI)"),
    ("psf_contam_", "ePSF star vetoes: cutout contamination"),
    ("psf_shape_", "ePSF star vetoes: ensemble shape outliers"),
    ("psf_fft_", "ePSF star vetoes: Fourier-domain outliers"),
    ("psf_norm_radius", "ePSF star vetoes: normalisation mask"),
    ("psf_grid", "Spatially varying (gridded) ePSF"),
    ("psf_spatial", "Spatially varying (gridded) ePSF"),
    ("psf_analytic_", "Analytic composite-Moffat PSF"),
    ("psf_use_analytic", "Analytic composite-Moffat PSF"),
    ("psf_init_moffat", "Analytic composite-Moffat PSF"),
    ("psf_fit_", "PSF fit and fitters"),
    ("psf_diff_negative", "PSF fit and fitters"),
    ("psf_target_fit", "PSF fit and fitters"),
    ("psf_localbkg", "PSF fit and fitters"),
    ("psf_fwhm_guard", "PSF fit and fitters"),
    ("psf_fit_positive", "PSF fit and fitters"),
    ("perform_emcee", "PSF fit and fitters"),
    ("check_inverted", "PSF fit and fitters"),
    ("use_poisson", "PSF fit and fitters"),
    ("poisson_", "PSF fit and fitters"),
    ("emcee_", "PSF fit and fitters"),
    ("psf_", "ePSF construction"),
    ("fwhm_", "Field FWHM estimation"),
    ("sextractor_reject", "ePSF construction"),
    ("oversample_psf", "ePSF construction"),
    ("linearity_", "ePSF construction"),
    ("undersampled_fwhm", "Sampling regime"),
    ("critical_fwhm", "Sampling regime"),
    ("oversampled_fwhm", "Sampling regime"),
    ("wcs_offset_min_snr", "Diagnostics"),
    ("target_defect", "Diagnostics"),
    ("aperture_n_jobs", "Parallelism"),
    ("photutils_n_threads", "Parallelism"),
    ("redo_sources", "Other"),
]

GROUP_ORDER = [
    "Detection and forced-photometry decision",
    "Centroid window of the PSF fit",
    "Aperture photometry",
    "Instrumental colour terms",
    "Local background infill (LPI)",
    "Sampling regime",
    "Field FWHM estimation",
    "ePSF construction",
    "ePSF star vetoes: cutout contamination",
    "ePSF star vetoes: ensemble shape outliers",
    "ePSF star vetoes: Fourier-domain outliers",
    "ePSF star vetoes: normalisation mask",
    "Analytic composite-Moffat PSF",
    "Spatially varying (gridded) ePSF",
    "PSF fit and fitters",
    "Diagnostics",
    "Parallelism",
    "Other",
]

KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)\s*:(?P<rest>.*)$"
)

# Keys with no consumer in the autophot package, verified with
# scripts/audit_config_usage.py plus manual checks against dynamic key
# construction and external-tool pass-through.  Dotted section.key paths.
# Currently empty: the 2026-11 audit removed the seven dead keys found.
UNREFERENCED_KEYS: set[str] = set()


def esc(text: str) -> str:
    """Escape a plain-text fragment for LaTeX."""
    if text is None:
        return ""
    t = str(text).strip()
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in (
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("<", "\\textless{}"),
        (">", "\\textgreater{}"),
        ("|", "\\textbar{}"),
    ):
        t = t.replace(a, b)
    # YAML comments mark inline code with backticks; render it in tt.
    t = re.sub(r"`([^`]+)`", r"\\texttt{\1}", t)
    return t


def breakable(tt_text: str) -> str:
    """Allow LaTeX to break a \\texttt{...} span after underscores."""
    return r"\texttt{" + tt_text.replace(r"\_", r"\_{\allowbreak}") + "}"


def break_tt_spans(text: str) -> str:
    """Insert breakpoints inside every \\texttt{...} in text.

    Long literal values (env vars, key lists, dicts, paths) have to wrap
    inside the narrow table columns; breaks are allowed after underscores,
    commas, slashes, dots, and colons.
    """
    def _fix(m):
        inner = m.group(1).replace(r"\_", r"\_{\allowbreak}")
        for ch in (",", "/", ".", ":"):
            inner = inner.replace(ch, ch + r"\allowbreak{}")
        return r"\texttt{" + inner + "}"

    return re.sub(r"\\texttt\{([^}]*)\}", _fix, text)


def break_plain(text: str) -> str:
    """Insert breakpoints in plain (non-tt) text segments.

    Descriptions quote long identifiers in roman text (env vars, paths,
    host names); breaks are allowed after escaped underscores, slashes,
    and letter-followed dots.
    """
    parts = re.split(r"(\\texttt\{[^}]*\})", text)
    for i, part in enumerate(parts):
        if part.startswith(r"\texttt{"):
            continue
        part = part.replace(r"\_", r"\_{\allowbreak}")
        part = part.replace("/", r"/\allowbreak{}")
        part = re.sub(r"\.(?=[A-Za-z])", r".\\allowbreak ", part)
        parts[i] = part
    return "".join(parts)


def parse_yml(path: Path):
    """Return {section: [(key, default, type, desc, runtime)]} preserving order."""
    sections: dict[str, list] = {}
    stack: list[tuple[int, str]] = []  # (indent, key) of open mappings

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(raw)
        if not m:
            continue
        indent = len(m.group("indent").replace("\t", "    "))
        key = m.group("key")
        rest = m.group("rest") or ""

        # Pop closed mappings.
        while stack and indent <= stack[-1][0]:
            stack.pop()

        has_value = bool(rest.strip()) and not rest.strip().startswith("#")
        is_mapping = not rest.strip()  # "key:" alone opens a section

        if is_mapping:
            stack.append((indent, key))
            if key not in SECTION_TITLES:
                # nested mapping, e.g. wcs.crowded -> keep under parent title
                parent = stack[-2][1] if len(stack) > 1 else "default_input"
                sections.setdefault(parent, [])
                sections[parent].append(("__HEADER__", key, "", "", False))
            else:
                sections.setdefault(key, [])
            continue

        # Scalar/list entry.
        value_part, _, comment = rest.partition("#")
        value = value_part.strip() or "null"
        dtype, desc = "", comment.strip()
        if "---" in desc:
            dtype, _, desc = desc.partition("---")
            dtype, desc = dtype.strip(), desc.strip()
        runtime = desc.lower().startswith("runtime")
        # Attach to the nearest ancestor that is a documented section;
        # nested mappings (wcs.crowded etc.) stay inside their parent table.
        parent = "default_input"
        for _, skey in reversed(stack):
            if skey in SECTION_TITLES:
                parent = skey
                break
        sections.setdefault(parent, []).append(
            (key, value, dtype, desc, runtime)
        )

    return sections


def phot_group(key: str) -> str:
    for prefix, group in PHOTOMETRY_GROUPS:
        if key.startswith(prefix):
            return group
    return "Other"


def fmt_default(v: str) -> str:
    if v in ("null", "None", ""):
        return r"\emph{null}"
    return break_tt_spans(r"\texttt{" + esc(v) + "}")


def emit_table(title: str, rows, path_prefix: str, section: str, fid):
    """Emit one longtable; rows may contain __HEADER__ pseudo-rows."""
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", path_prefix).strip("-")
    fid.write(
        "{\\footnotesize\n"
        "\\begin{longtable}{@{}p{5.6cm}p{2.2cm}p{7.4cm}@{}}\n"
        "\\caption{%s --- configuration reference.} \\label{tab:%s}\\\\\n"
        "\\toprule\n"
        "\\textbf{Key} & \\textbf{Default (type)} & \\textbf{Description} \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{3}{l}{\\footnotesize\\emph{continued}}\\\\\n"
        "\\toprule\n"
        "\\textbf{Key} & \\textbf{Default (type)} & \\textbf{Description} \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n" % (esc(title), label)
    )
    for key, value, dtype, desc, runtime in rows:
        if key == "__HEADER__":
            fid.write(
                "\\multicolumn{3}{l}{\\textbf{sub-section \\texttt{%s}}}\\\\\n"
                "\\midrule\n" % esc(value)
            )
            continue
        dv = fmt_default(value)
        if dtype:
            # Long union/list type annotations must wrap in the narrow
            # Default column.
            t = esc(dtype).replace(r"\textbar{}", r"\textbar{}\allowbreak{}")
            t = t.replace("[", r"[\allowbreak{}")
            dv += " {\\scriptsize(%s)}" % t
        d = break_plain(break_tt_spans(esc(desc)))
        marks = []
        if runtime:
            marks.append("runtime")
        # "Deprecated" at the head of a description is the deprecation
        # marker; the same word later in the text just refers to old keys.
        if "deprecat" in desc[:15].lower():
            marks.append("deprecated")
        path = key if section == "default_input" else "%s.%s" % (section, key)
        if path in UNREFERENCED_KEYS:
            marks.append("unreferenced")
        if marks:
            d = "\\emph{(%s)} %s" % (", ".join(marks), d)
        fid.write(
            "%s & %s & %s \\\\\n"
            % (breakable(esc(key)), dv, d)
        )
    fid.write("\\end{longtable}}\n\n")


def main() -> int:
    sections = parse_yml(YML)

    out = sys.stdout
    out.write(
        "% Auto-generated by scripts/generate_parameter_appendix.py\n"
        "% Source: autophot/data/default_input.yml\n"
        "% Do not edit by hand; regenerate after changing default_input.yml.\n\n"
    )

    order = [s for s in SECTION_TITLES if s in sections]
    order += [s for s in sections if s not in SECTION_TITLES]
    for sec in order:
        rows = sections[sec]
        title = SECTION_TITLES.get(sec, sec)
        if sec == "photometry":
            # Split the large photometry block into thematic sub-tables.
            grouped: dict[str, list] = {}
            for r in rows:
                grouped.setdefault(phot_group(r[0]), []).append(r)
            for gname in GROUP_ORDER:
                grows = grouped.get(gname)
                if not grows:
                    continue
                emit_table(
                    "%s: %s" % (title, gname),
                    grows,
                    f"photometry-{gname}",
                    sec,
                    out,
                )
            leftovers = [k for k in grouped if k not in GROUP_ORDER]
            for k in leftovers:
                emit_table(
                    f"{title}: {k}", grouped[k], f"photometry-{k}", sec, out
                )
        else:
            emit_table(title, rows, sec, sec, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
