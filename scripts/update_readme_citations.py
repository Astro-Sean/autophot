#!/usr/bin/env python3
"""Refresh the AutoPhOT citation count in README.md.

Runs during the conda build so every released version advertises a current
count.  Uses the ADS API when ADS_API_TOKEN (or SCIX_API_TOKEN) is exported;
otherwise falls back to OpenAlex, whose astronomy coverage tracks ADS more
closely than Semantic Scholar.  Always exits 0 -- a stale count must not
block a build.
"""

import json
import os
import re
import sys
import urllib.request
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, "README.md")

BIBCODE = "2022A&A...667A..62B"
DOI = "10.1051/0004-6361/202243067"

# The README line carries HTML markers so the count can be rewritten
# without depending on the surrounding wording.
_MARK_RE = re.compile(
    r"(<!--\s*citation-count\s*-->)[^<]*(<!--\s*/citation-count\s*-->)"
)
_DATE_RE = re.compile(
    r"(<!--\s*citation-date\s*-->)[^<]*(<!--\s*/citation-date\s*-->)"
)


def _get_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _count_from_ads(token: str, api_base: str) -> int:
    q = f"citations(bibcode:{BIBCODE})"
    url = (
        f"{api_base}/v1/search/query"
        f"?q={urllib.request.quote(q)}&rows=0&fl=bibcode"
    )
    data = _get_json(url, {"Authorization": f"Bearer {token}"})
    return int(data["response"]["numFound"])


def _count_from_openalex() -> int:
    url = f"https://api.openalex.org/works/doi:{DOI}?select=cited_by_count"
    data = _get_json(url)
    return int(data["cited_by_count"])


def _count_from_semantic_scholar() -> int:
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}"
        "?fields=citationCount"
    )
    data = _get_json(url)
    return int(data["citationCount"])


def main() -> int:
    try:
        if os.environ.get("SCIX_API_TOKEN"):
            count = _count_from_ads(
                os.environ["SCIX_API_TOKEN"], "https://api.scixplorer.org"
            )
            source = "SciX"
        elif os.environ.get("ADS_API_TOKEN"):
            count = _count_from_ads(
                os.environ["ADS_API_TOKEN"], "https://api.adsabs.harvard.edu"
            )
            source = "ADS"
        else:
            try:
                count = _count_from_openalex()
                source = "OpenAlex"
            except Exception:
                count = _count_from_semantic_scholar()
                source = "Semantic Scholar"
    except Exception as exc:
        print(f"[citations] WARNING: count lookup failed ({exc}); README unchanged.")
        return 0

    if count <= 0:
        print(f"[citations] WARNING: {source} returned {count}; README unchanged.")
        return 0

    with open(README, encoding="utf-8") as fh:
        text = fh.read()
    if not _MARK_RE.search(text):
        print("[citations] WARNING: citation-count marker not found in README.md.")
        return 0
    updated, n_sub = _MARK_RE.subn(rf"\g<1>{count}\g<2>", text)
    if not n_sub:
        print("[citations] WARNING: citation-count substitution failed.")
        return 0

    # Stamp the lookup date whenever a query succeeds so the README shows
    # how fresh the advertised count is.
    today = date.today().strftime("%d-%m-%Y")
    updated, n_date = _DATE_RE.subn(rf"\g<1>{today}\g<2>", updated)
    if not n_date:
        print("[citations] WARNING: citation-date marker not found in README.md.")

    if updated != text:
        with open(README, "w", encoding="utf-8") as fh:
            fh.write(updated)
        print(
            f"[citations] README updated: {count} citing papers "
            f"as of {today} ({source})."
        )
    else:
        print(f"[citations] Count unchanged at {count} ({source}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
