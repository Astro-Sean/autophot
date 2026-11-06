#!/usr/bin/env python3
"""Plot papers citing AutoPhOT per year and the running total.

Uses the ADS/SciX API when ADS_API_TOKEN (or SCIX_API_TOKEN) is exported,
otherwise falls back to OpenAlex, matching update_readme_citations.py.
Writes figures/citations_over_time.{png,pdf} and a JSON record of the
yearly counts so the figure can be audited without re-querying.
"""

import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import date

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    # The conda build env does not carry matplotlib; skip rather than
    # block a release over a README figure.
    print("[citations] WARNING: matplotlib/numpy missing; figure skipped.")
    sys.exit(0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "figures")

BIBCODE = "2022A&A...667A..62B"
OPENALEX_ID = "W4308595501"


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _years_from_ads(token, api_base):
    years = []
    start = 0
    while True:
        q = f"citations(bibcode:{BIBCODE})"
        url = (
            f"{api_base}/v1/search/query"
            f"?q={urllib.request.quote(q)}&rows=200&start={start}&fl=year"
        )
        data = _get_json(url, {"Authorization": f"Bearer {token}"})
        docs = data["response"]["docs"]
        if not docs:
            break
        years.extend(int(d["year"]) for d in docs if d.get("year"))
        start += len(docs)
        if start >= int(data["response"]["numFound"]):
            break
    return years


def _years_from_openalex():
    years = []
    cursor = "*"
    while cursor:
        url = (
            "https://api.openalex.org/works"
            f"?filter=cites:{OPENALEX_ID}"
            "&select=publication_year&per-page=200"
            f"&cursor={cursor}"
        )
        data = _get_json(url)
        years.extend(
            int(w["publication_year"])
            for w in data["results"]
            if w.get("publication_year")
        )
        cursor = data["meta"].get("next_cursor")
    return years


def fetch_citing_years():
    if os.environ.get("SCIX_API_TOKEN"):
        return _years_from_ads(
            os.environ["SCIX_API_TOKEN"], "https://api.scixplorer.org"
        ), "SciX"
    if os.environ.get("ADS_API_TOKEN"):
        return _years_from_ads(
            os.environ["ADS_API_TOKEN"], "https://api.adsabs.harvard.edu"
        ), "ADS"
    return _years_from_openalex(), "OpenAlex"


def make_figure(years, out_base):
    counts = Counter(years)
    year_axis = list(range(min(counts), max(counts) + 1))
    per_year = [counts.get(y, 0) for y in year_axis]
    cumulative = np.cumsum(per_year)

    plt.rcParams.update(
        {
            "font.size": 7,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(
        year_axis,
        cumulative,
        color="#1f78b4",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
        zorder=3,
    )
    ax.annotate(
        str(int(cumulative[-1])),
        (year_axis[-1], cumulative[-1]),
        textcoords="offset points",
        xytext=(-4, 8),
        ha="right",
        fontsize=7,
        color="#1f78b4",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers using AutoPhOT")
    ax.set_xticks(year_axis)
    ax.set_ylim(0, cumulative[-1] * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(
        0.99,
        0.01,
        f"Updated: {date.today().strftime('%d-%m-%Y')}",
        ha="right",
        va="bottom",
        fontsize=5,
        color="0.4",
    )
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    return year_axis, per_year


def main():
    try:
        years, source = fetch_citing_years()
    except Exception as exc:
        print(f"[citations] WARNING: lookup failed ({exc}); figure unchanged.")
        return 0
    if not years:
        print(f"[citations] WARNING: {source} returned no citing papers.")
        return 0

    os.makedirs(FIG_DIR, exist_ok=True)
    out_base = os.path.join(FIG_DIR, "citations_over_time")
    year_axis, per_year = make_figure(years, out_base)

    record = {
        "source": source,
        "fetched": date.today().isoformat(),
        "total": len(years),
        "per_year": dict(zip(year_axis, per_year)),
    }
    with open(out_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    print(
        f"[citations] {len(years)} citing papers ({source}), "
        f"figure written to {out_base}.png/.pdf"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
