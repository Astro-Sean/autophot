#!/usr/bin/env python3
"""
Download astrometry.net index files into the data directory that
solve-field actually reads.

Locates the correct install location by (in order):
  1. --data-dir, if given
  2. $ASTROMETRY_DATA / $ASTROMETRY_INDEX_PATH
  3. `add_path` entries in the astrometry config files next to the
     solve-field binary found on PATH (etc/astrometry.cfg, backend.cfg)
  4. the `data/` directory beside the solve-field install prefix

The "correct location" is whatever the solve-field config already
lists, so files land where the solver looks rather than where we guess.

File selection:
  --series 5200 --scales 0-6   (default: all of the chosen series)
  --from-log path/to/Astrometry*.log   (only the index names a failed
                                        solve actually asked for)

Series base URLs follow https://data.astrometry.net/ (the 5200 LITE
series is hosted on the NERSC portal).
"""

import argparse
import os
import re
import shutil
import sys
import urllib.request
import urllib.error

DATA_ROOT = "https://data.astrometry.net"
SERIES_URLS = {
    "4100": f"{DATA_ROOT}/4100/",
    "4200": f"{DATA_ROOT}/4200/",
    "5000": f"{DATA_ROOT}/5000/",
    "5200": (
        "https://portal.nersc.gov/project/cosmo/temp/dstn/"
        "index-5200/LITE/"
    ),
    "5200-heavy": (
        "https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/"
    ),
    "6000": f"{DATA_ROOT}/6000/",
    "6100": f"{DATA_ROOT}/6100/",
}
INDEX_RE = re.compile(r"index-[0-9A-Za-z._-]+\.fits")
ADD_PATH_RE = re.compile(r"^\s*add_path\s+(\S+)", re.M)


def eprint(*a):
    print(*a, file=sys.stderr)


def find_solve_field(explicit=None):
    """Return (path_as_found, realpath) for the solve-field binary."""
    candidates = ([explicit] if explicit else []) + [
        shutil.which("solve-field"),
        "/usr/local/astrometry/bin/solve-field",
        "/usr/bin/solve-field",
        "/home/linuxbrew/.linuxbrew/bin/solve-field",
        "/opt/homebrew/bin/solve-field",
        "/usr/local/bin/solve-field",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand, os.path.realpath(cand)
    return None, None


def config_add_paths(prefix_dirs):
    """add_path entries from astrometry config files near the install.

    Looks under every prefix the binary can map to: solve-field locates
    astrometry.cfg relative to the *invoked* path (bin/../etc), while a
    symlinked binary (e.g. a brew Cellar link) can put it under the
    resolved prefix instead.  Both are tried.
    """
    candidates = []
    for p in prefix_dirs:
        candidates += [
            os.path.join(p, "etc", "astrometry.cfg"),
            os.path.join(p, "etc", "astrometry", "backend.cfg"),
            os.path.join(p, "etc", "backend.cfg"),
        ]
    candidates += [
        os.path.expanduser("~/.astrometry/backend.cfg"),
        "/usr/local/astrometry/etc/astrometry.cfg",
        "/etc/astrometry/backend.cfg",
    ]
    found = []
    seen = set()
    for cfg in candidates:
        if cfg in seen:
            continue
        seen.add(cfg)
        try:
            with open(cfg) as fh:
                base = os.path.dirname(os.path.dirname(cfg))
                for m in ADD_PATH_RE.finditer(fh.read()):
                    p = m.group(1)
                    if not os.path.isabs(p):
                        p = os.path.normpath(os.path.join(base, p))
                    found.append((p, cfg))
        except OSError:
            continue
    return found


def find_data_dir(args):
    if args.data_dir:
        return args.data_dir, "--data-dir"
    for env in ("ASTROMETRY_DATA", "ASTROMETRY_INDEX_PATH"):
        if os.environ.get(env):
            return os.environ[env], f"${env}"
    exe_found, exe_real = find_solve_field(args.solve_field)
    if exe_real is None:
        eprint("Could not find solve-field on PATH; use --data-dir.")
        return None, None
    # Prefix candidates: the PATH-visible install and the realpath'd one
    # can differ (e.g. a brew symlink into Cellar); astrometry searches
    # etc/ relative to the invoked binary, so both feed the cfg lookup.
    prefix_dirs = []
    for e in (exe_found, exe_real):
        d = os.path.dirname(os.path.dirname(e))
        if d not in prefix_dirs:
            prefix_dirs.append(d)
    cfg_paths = config_add_paths(prefix_dirs)
    if cfg_paths:
        under = [
            p for p, _ in cfg_paths
            if any(p.startswith(pd) for pd in prefix_dirs)
        ]
        pick = under[0] if under else cfg_paths[0][0]
        cfg = dict((p, c) for p, c in cfg_paths)[pick]
        return pick, f"add_path in {cfg}"
    for guess in [
        os.path.join(d, "data") for d in reversed(prefix_dirs)
    ] + ["/usr/local/astrometry/data", "/usr/share/astrometry"]:
        if os.path.isdir(guess):
            return guess, f"existing dir under {exe_real}"
    guessed = os.path.join(prefix_dirs[-1], "data")
    return guessed, f"inferred from {exe_real} (no add_path found)"


def fetch_listing(url):
    req = urllib.request.Request(url, headers={"User-Agent": "autophot-index-helper"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(INDEX_RE.findall(html)))


def scale_of(name):
    """Scale digits are positions 6:8 ('index-5200-12' -> '00',
    'index-4119' -> '19'). '52mN' merged files return 'mN'."""
    m = re.match(r"index-\d{2}([0-9m]\d?)[-.]", name)
    if not m:
        return None
    tok = m.group(1)
    if tok.startswith("m"):
        return tok
    try:
        return f"{int(tok):02d}"
    except ValueError:
        return None


def in_scale_range(name, lo, hi):
    s = scale_of(name)
    if s is None:
        return False
    if s.startswith("m"):
        return True  # merged files always useful for the series
    return lo <= int(s) <= hi


def parse_scales(text):
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", text)
    if not m:
        raise argparse.ArgumentTypeError("--scales expects e.g. 0-6")
    return int(m.group(1)), int(m.group(2))


def remote_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD", timeout=30)
        with urllib.request.urlopen(req) as r:
            return int(r.headers.get("Content-Length") or -1)
    except Exception:
        return -1


def download(url, dest, dry_run=False):
    size = remote_size(url)
    if os.path.exists(dest) and size > 0 and os.path.getsize(dest) == size:
        return "skip"
    if dry_run:
        return f"would download {size / 1e6:.1f} MB" if size > 0 else "would download"
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "autophot-index-helper"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1024 * 1024)
    os.replace(tmp, dest)
    if size > 0 and os.path.getsize(dest) != size:
        return "size-mismatch"
    return "ok"


def main():
    ap = argparse.ArgumentParser(
        description="Download astrometry.net index files into the "
        "data dir solve-field reads."
    )
    ap.add_argument("--series", default="5200", choices=sorted(SERIES_URLS),
                    help="index series (default 5200 LITE, fields < 1 deg; "
                    "use 4100 for > 1 deg, 4200 for 2MASS 0-19)")
    ap.add_argument("--scales", type=parse_scales, default=None,
                    metavar="LO-HI", help="scale range, e.g. 0-6 "
                    "(default: every scale in the series)")
    ap.add_argument("--from-log", metavar="LOG",
                    help="only download the index names a solve-field "
                    "log actually asked for")
    ap.add_argument("--solve-field", help="path to the solve-field binary "
                    "(default: PATH plus common brew/astrometry locations)")
    ap.add_argument("--data-dir", help="destination index dir "
                    "(default: detected from solve-field config)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be downloaded")
    args = ap.parse_args()

    data_dir, source = find_data_dir(args)
    if data_dir is None:
        return 2
    os.makedirs(data_dir, exist_ok=True)
    if not os.access(data_dir, os.W_OK):
        eprint(f"Not writable: {data_dir} (check permissions)")
        return 2
    print(f"Index dir: {data_dir}  (from {source})")

    listing = fetch_listing(SERIES_URLS[args.series])
    if not listing:
        eprint("No index files found in remote listing; series URL wrong?")
        return 2

    if args.from_log:
        wanted_names = set()
        with open(args.from_log, errors="replace") as fh:
            for m in INDEX_RE.finditer(fh.read()):
                wanted_names.add(m.group(0))
        wanted = [n for n in listing if n in wanted_names]
        if not wanted:
            eprint("No index-*.fits names from the log exist in this "
                   "series listing; check --series.")
            return 2
    elif args.scales:
        lo, hi = args.scales
        wanted = [n for n in listing if in_scale_range(n, lo, hi)]
    else:
        wanted = listing

    print(f"{len(wanted)} file(s) to fetch from {SERIES_URLS[args.series]}")
    base = SERIES_URLS[args.series]
    n_ok = n_skip = n_fail = 0
    for name in sorted(wanted):
        url = base + name
        dest = os.path.join(data_dir, name)
        try:
            res = download(url, dest, dry_run=args.dry_run)
        except urllib.error.URLError as exc:
            res = f"FAILED: {exc}"
        print(f"  {name}: {res}")
        if res == "ok":
            n_ok += 1
        elif res == "skip":
            n_skip += 1
        elif res.startswith("FAILED") or res == "size-mismatch":
            n_fail += 1

    print(f"Done: {n_ok} downloaded, {n_skip} already present, "
          f"{n_fail} failed -> {data_dir}")
    if source and source.startswith("inferred"):
        eprint(f"Note: {data_dir} was inferred, not read from the config. "
               "If solves still fail, add it to astrometry.cfg via "
               f"'add_path {data_dir}'.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
