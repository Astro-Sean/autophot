#!/usr/bin/env python3
"""
test_vizier_mirrors.py
======================
Probe every known VizieR mirror and report which ones answer a real
catalog query. The CDS host (vizier.cds.unistra.fr) has intermittent
outages; when /viz-bin/votable starts returning 404, set astroquery to a
working mirror:

    from astroquery.vizier import Vizier
    Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"

Usage
-----
    python test_vizier_mirrors.py [--ra DEG] [--dec DEG] [--radius ARCMIN]
                                  [--catalog CATALOG_ID] [--timeout SEC]
                                  [--astroquery-check]

Options
-------
    --ra, --dec     Cone centre (default: 47.9087, -2.0267).
    --radius        Cone radius in arcmin (default: 1.0).
    --catalog       VizieR catalog ID (default: II/349, PS1 DR1).
    --timeout       Per-mirror HTTP timeout in seconds (default: 15).
    --astroquery-check   Also run an astroquery query_region against the
                         fastest working mirror (needs astroquery).
"""

import argparse
import sys
import time

import requests

# Mirror list from https://vizier.cds.unistra.fr/doc/asu-summary.htx
# (host label, hostname, location)
MIRRORS = [
    ("CDS", "vizier.cds.unistra.fr", "Strasbourg, FR"),
    ("CFA", "vizier.cfa.harvard.edu", "Harvard, US"),
    ("CADC", "vizier.hia.nrc.ca", "Victoria, CA"),
    ("ADAC", "vizier.nao.ac.jp", "Tokyo, JP"),
    ("CAM", "vizier.ast.cam.ac.uk", "Cambridge, UK"),
    ("JAC", "vizier.ukirt.jach.hawaii.edu", "Hawaii, US"),
    ("IUCAA", "vizier.iucaa.in", "Pune, IN"),
    ("IUCAA2", "viziers.iucaa.in", "Pune, IN (alt)"),
    ("INASAN", "vizier.inasan.ru", "Moscow, RU"),
    ("BAO", "data.bao.ac.cn", "Beijing, CN"),
    ("BAO2", "vizier.bao.ac.cn", "Beijing, CN (alt)"),
]

ASU_PATH = "/viz-bin/votable"


def probe_mirror(host, payload, timeout):
    """POST one ASU cone query to `host` and time the response.

    Returns (status_label, detail, seconds, rows) where status_label is
    'OK', 'HTTP', 'EMPTY', or 'ERR'.
    """
    url = f"https://{host}{ASU_PATH}"
    t0 = time.monotonic()
    try:
        r = requests.post(url, data=payload, timeout=timeout)
    except Exception as exc:
        return "ERR", f"{type(exc).__name__}", None, 0
    dt = time.monotonic() - t0

    if r.status_code != 200:
        return "HTTP", str(r.status_code), dt, 0
    body = r.text
    if "<VOTABLE" not in body:
        return "EMPTY", "non-VOTable body", dt, 0
    rows = body.count("<TR>")
    return "OK", "votable", dt, rows


def astroquery_check(host, ra, dec, radius_arcmin, catalog, timeout):
    """Run the real astroquery path against `host`; returns row count or raises."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    Vizier.VIZIER_SERVER = host
    Vizier.TIMEOUT = timeout
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    res = Vizier.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog)
    return 0 if res is None else len(res[0])


def main():
    ap = argparse.ArgumentParser(description="Probe VizieR mirrors.")
    ap.add_argument("--ra", type=float, default=47.9087)
    ap.add_argument("--dec", type=float, default=-2.0267)
    ap.add_argument("--radius", type=float, default=1.0, help="arcmin")
    ap.add_argument("--catalog", default="II/349")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--astroquery-check", action="store_true")
    args = ap.parse_args()

    payload = {
        "-source": args.catalog,
        "-c": f"{args.ra} {args.dec}",
        "-c.rs": str(args.radius),
        "-out.max": "5",
    }

    print(f"VizieR mirror probe: {args.catalog} @ ({args.ra}, {args.dec}), r={args.radius}'")
    print(f"{'mirror':<7} {'host':<32} {'status':<6} {'rows':>5} {'time':>7}  detail")
    print("-" * 78)

    best = None
    for label, host, loc in MIRRORS:
        status, detail, dt, rows = probe_mirror(host, payload, args.timeout)
        tstr = f"{dt:6.1f}s" if dt is not None else "     -"
        print(f"{label:<7} {host:<32} {status:<6} {rows:>5} {tstr}  {detail} ({loc})")
        if status == "OK" and (best is None or dt < best[2]):
            best = (label, host, dt)

    print("-" * 78)
    if best is None:
        print("No working mirror found.")
        return 1

    print(f"Fastest working mirror: {best[0]} ({best[1]}) in {best[2]:.1f}s")
    print(f"Use: Vizier.VIZIER_SERVER = '{best[1]}'")

    if args.astroquery_check:
        try:
            n = astroquery_check(
                best[1], args.ra, args.dec, args.radius, args.catalog, args.timeout
            )
            print(f"astroquery check: query_region returned {n} row(s) via {best[1]}")
        except Exception as exc:
            print(f"astroquery check failed via {best[1]}: {exc!r}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
