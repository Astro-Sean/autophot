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
    from matplotlib.ticker import AutoMinorLocator
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


def _months_from_ads(token, api_base):
    months = []
    start = 0
    while True:
        q = f"citations(bibcode:{BIBCODE})"
        url = (
            f"{api_base}/v1/search/query"
            f"?q={urllib.request.quote(q)}&rows=200&start={start}&fl=pubdate"
        )
        data = _get_json(url, {"Authorization": f"Bearer {token}"})
        docs = data["response"]["docs"]
        if not docs:
            break
        # ADS pubdate is YYYY-MM-DD with the day zeroed; month can be 00
        # for records that only carry a year, so fall back to January.
        for d in docs:
            pub = d.get("pubdate", "")
            parts = pub.split("-")
            if len(parts) >= 2 and parts[1] not in ("", "0", "00"):
                months.append((int(parts[0]), int(parts[1])))
            elif parts and parts[0]:
                months.append((int(parts[0]), 1))
        start += len(docs)
        if start >= int(data["response"]["numFound"]):
            break
    return months


def _months_from_openalex():
    months = []
    cursor = "*"
    while cursor:
        url = (
            "https://api.openalex.org/works"
            f"?filter=cites:{OPENALEX_ID}"
            "&select=publication_date&per-page=200"
            f"&cursor={cursor}"
        )
        data = _get_json(url)
        for w in data["results"]:
            pub = w.get("publication_date", "")
            parts = pub.split("-")
            if len(parts) >= 2:
                months.append((int(parts[0]), int(parts[1])))
        cursor = data["meta"].get("next_cursor")
    return months


def fetch_citing_months():
    if os.environ.get("SCIX_API_TOKEN"):
        return _months_from_ads(
            os.environ["SCIX_API_TOKEN"], "https://api.scixplorer.org"
        ), "SciX"
    if os.environ.get("ADS_API_TOKEN"):
        return _months_from_ads(
            os.environ["ADS_API_TOKEN"], "https://api.adsabs.harvard.edu"
        ), "ADS"
    return _months_from_openalex(), "OpenAlex"


def make_figure(months, out_base):
    # Monthly pubdates are fetched so the same data can drive finer
    # binning later; the figure reports yearly totals.
    counts = Counter(y for y, _ in months)
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
    ax.xaxis.set_minor_locator(AutoMinorLocator(12))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="minor", length=2)
    ax.text(
        0.98,
        0.03,
        f"Updated: {date.today().strftime('%d-%m-%Y')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5,
        color="0.4",
    )
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    return year_axis, per_year


def main():
    try:
        months, source = fetch_citing_months()
    except Exception as exc:
        print(f"[citations] WARNING: lookup failed ({exc}); figure unchanged.")
        return 0
    if not months:
        print(f"[citations] WARNING: {source} returned no citing papers.")
        return 0

    os.makedirs(FIG_DIR, exist_ok=True)
    out_base = os.path.join(FIG_DIR, "citations_over_time")
    year_axis, per_year = make_figure(months, out_base)

    record = {
        "source": source,
        "fetched": date.today().isoformat(),
        "total": len(months),
        "per_year": dict(zip(year_axis, per_year)),
    }
    with open(out_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    print(
        f"[citations] {len(months)} citing papers ({source}), "
        f"figure written to {out_base}.png/.pdf"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
