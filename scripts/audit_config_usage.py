#!/usr/bin/env python3
"""Audit which keys in default_input.yml are still referenced by the code.

Parses ``autophot/data/default_input.yml`` for every leaf parameter and
searches the package source for references. Keys are classified as:

- ``used``    : literal key string found in code
- ``dynamic`` : not found literally, but a code-side prefix/constructed
                access could plausibly consume it
- ``runtime`` : description marks the key as pipeline-populated
- ``missing`` : no code reference at all - removal candidate

Dynamic access (``cfg[f"x_{i}"]``, ``startswith`` iteration, kwargs
splats) means ``missing`` is a *candidate*, not a verdict; the report
exists to be reviewed by hand.

Usage:
    python scripts/audit_config_usage.py [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
YML = REPO / "autophot" / "data" / "default_input.yml"
PKG = REPO / "autophot"

KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)\s*:(?P<rest>.*)$")

# Sections that are documented top-level groups in the YAML.
TOP_LEVEL = {
    "preprocessing",
    "photometry",
    "wcs",
    "catalog",
    "zeropoint",
    "cosmic_rays",
    "source_detection",
    "limiting_magnitude",
    "host_model",
    "alignment",
    "template_subtraction",
    "background",
    "postprocessing",
}


def parse_leaf_keys(path: Path) -> list[dict]:
    """Return [(section_path, key, dtype, desc, runtime)] in file order."""
    out = []
    stack: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(raw)
        if not m:
            continue
        indent = len(m.group("indent").replace("\t", "    "))
        key, rest = m.group("key"), m.group("rest") or ""
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not rest.strip():
            stack.append((indent, key))
            continue
        value_part, _, comment = rest.partition("#")
        if not value_part.strip() and not comment:
            continue
        dtype, desc = "", comment.strip()
        if "---" in desc:
            dtype, _, desc = desc.partition("---")
            dtype, desc = dtype.strip(), desc.strip()
        section = stack[0][1] if stack else "default_input"
        if section not in TOP_LEVEL:
            section = "default_input"
        out.append(
            {
                "section": section,
                "path": ".".join([k for _, k in stack] + [key]),
                "key": key,
                "dtype": dtype,
                "desc": desc,
                "runtime": desc.lower().startswith("runtime"),
            }
        )
    return out


def code_corpus(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (pkg_corpus, other_corpus) of {relpath: text}.

    ``pkg`` is the autophot package itself - the only corpus that decides
    whether a parameter is consumed.  ``other`` (scripts, tests, examples,
    notebooks) is reported for context but does not count as usage.
    """
    pkg, other = {}, {}
    for sub in (root / "autophot",):
        for f in sub.rglob("*.py"):
            pkg[str(f.relative_to(root))] = f.read_text(
                encoding="utf-8", errors="replace"
            )
    for sub in (root / "scripts", root / "tests", root / "examples"):
        if not sub.is_dir():
            continue
        for f in list(sub.rglob("*.py")) + list(sub.rglob("*.ipynb")):
            other[str(f.relative_to(root))] = f.read_text(
                encoding="utf-8", errors="replace"
            )
    return pkg, other


def find_refs(key: str, corpus: dict[str, str]) -> list[str]:
    """Files containing the literal key as a standalone token."""
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])")
    return sorted(p for p, txt in corpus.items() if pat.search(txt))


def dynamic_prefixes(corpus_text: str) -> list[str]:
    """Prefixes used in startswith()/f-string key construction."""
    prefixes = set()
    for m in re.finditer(r"startswith\(\s*['\"]([^'\"]+)['\"]", corpus_text):
        prefixes.add(m.group(1))
    # f-string key construction such as f"snr_{thr}" -> prefix "snr_".
    for m in re.finditer(r"[fF]['\"]([A-Za-z0-9_]{3,}_)\{", corpus_text):
        prefixes.add(m.group(1))
    return sorted(prefixes, key=len, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    keys = parse_leaf_keys(YML)
    pkg, other = code_corpus(REPO)
    all_text = "\n".join(list(pkg.values()) + list(other.values()))
    prefixes = dynamic_prefixes(all_text)

    report = []
    for k in keys:
        refs = find_refs(k["key"], pkg)
        other_refs = find_refs(k["key"], other)
        status = "used" if refs else "missing"
        dyn = ""
        if status == "missing":
            for p in prefixes:
                if k["key"].startswith(p):
                    status = "dynamic"
                    dyn = p
                    break
        if k["runtime"] and status == "missing":
            status = "runtime"
        report.append(
            {**k, "status": status, "refs": refs, "other": other_refs, "dyn": dyn}
        )

    if args.json:
        json.dump(report, sys.stdout, indent=1)
        return 0

    by_status = {}
    for r in report:
        by_status.setdefault(r["status"], []).append(r)
    total = len(report)
    print(f"{total} leaf parameters in {YML.name}")
    for s in ("used", "dynamic", "runtime", "missing"):
        n = len(by_status.get(s, []))
        print(f"  {s:8s}: {n}")
    print()
    for s in ("dynamic", "missing"):
        for r in by_status.get(s, []):
            note = f"  (prefix {r['dyn']!r})" if r["dyn"] else ""
            print(f"[{s:7s}] {r['path']}{note}")
            if r["other"]:
                print(f"          non-package refs: {', '.join(r['other'])}")
            if r["desc"]:
                print(f"          {r['desc'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
