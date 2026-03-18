#!/usr/bin/env python3
"""
Batch launcher for the autophot pipeline.

This script lets you run the existing single-image pipeline (`main.py`)
on many FITS files in parallel, using multiple CPU cores on a single node.

Design choices:
- We DO NOT touch `main.py` or `run_photometry()`. Each image is processed
  by spawning a separate Python process that calls `main.py -f <file> -c <yaml>`.
- Parallelism is at the *image* level here; within each image, `main.py`
  still uses its own multiprocessing (aperture, CoG) and NTHREADS
  for SExtractor/SCAMP/SWarp.

Typical usage:

  # Process a list of FITS files in parallel
  python batch_main.py -f image1.fits image2.fits image3.fits -c config.yml --jobs 3

  # Or, give a text file with one FITS path per line
  python batch_main.py --file-list images.txt -c config.yml --jobs 4

On an HPC system, you would usually run this inside a single job that
requests enough cores (e.g. --cpus-per-task) to cover the requested
number of parallel images times the per-image threads.
"""

from __future__ import annotations

import argparse
import os

# Force BLAS/OpenMP to 1 thread so subprocesses and any early imports don't exhaust
# process/thread limits on HPC (OpenBLAS often defaults to 128 threads per process).
for _env in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env] = "1"

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple


def _read_file_list(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File list not found: {p}")
    lines: List[str] = []
    with p.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _resolve_main_py() -> str:
    """
    Return the absolute path to main.py in the same directory as this file.
    """
    here = Path(__file__).resolve().parent
    main_py = here / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Could not find main.py next to {__file__}")
    return str(main_py)


def _run_single_image(
    fits_path: str,
    yaml_path: str,
    prepare_template: bool = False,
) -> Tuple[str, int]:
    """
    Run the existing single-image pipeline (main.py) on one FITS file.

    Returns (fits_path, return_code).
    """
    main_py = _resolve_main_py()
    cmd = [
        sys.executable,
        main_py,
        "-f",
        fits_path,
        "-c",
        yaml_path,
    ]
    if prepare_template:
        cmd.append("-temp")

    # Inherit stdout/stderr so logs appear live for each subprocess.
    # This keeps behaviour close to calling main.py directly.
    try:
        result = subprocess.run(cmd, check=False)
        return fits_path, result.returncode
    except Exception:
        # Treat unexpected failures as non-zero for reporting.
        return fits_path, 1


def _build_file_list(args: argparse.Namespace) -> List[str]:
    files: List[str] = []

    if args.files:
        files.extend(args.files)

    if args.file_list:
        files.extend(_read_file_list(args.file_list))

    # Basic normalisation and existence check
    resolved: List[str] = []
    for f in files:
        p = Path(f).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"FITS file not found: {p}")
        resolved.append(str(p.resolve()))

    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for f in resolved:
        if f in seen:
            continue
        seen.add(f)
        unique.append(f)

    if not unique:
        raise ValueError("No FITS files specified. Use -f/--files or --file-list.")

    return unique


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the autophot pipeline (main.py) on many FITS files "
            "in parallel on a single node."
        )
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="One or more FITS files to process.",
    )
    parser.add_argument(
        "--file-list",
        help="Text file with one FITS path per line (comments starting with # are ignored).",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="YAML configuration file (same as -c for main.py).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help=(
            "Number of images to process in parallel. "
            "Default: min(number of files, max(1, os.cpu_count() // 4))."
        ),
    )
    parser.add_argument(
        "-t",
        "--prepare-template",
        action="store_true",
        help="Pass the -temp flag through to main.py for all images.",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    fits_files = _build_file_list(args)
    n_files = len(fits_files)

    if args.jobs is None:
        # Conservative default: leave headroom for per-image threads (SExtractor, SCAMP, SWarp, etc.).
        cpu_count = os.cpu_count() or 1
        default_jobs = max(1, cpu_count // 4)
        jobs = min(n_files, default_jobs)
    else:
        jobs = max(1, min(args.jobs, n_files))

    print(f"Found {n_files} FITS file(s). Running up to {jobs} job(s) in parallel.")

    failures: List[str] = []

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_path = {
            executor.submit(
                _run_single_image,
                f,
                args.config,
                args.prepare_template,
            ): f
            for f in fits_files
        }

        for future in as_completed(future_to_path):
            fpath = future_to_path[future]
            try:
                _, rc = future.result()
            except Exception as exc:
                print(f"Error: {fpath}: {exc}", flush=True)
                failures.append(fpath)
                continue

            if rc == 0:
                print(f"OK: {fpath}", flush=True)
            else:
                print(f"[FAIL]  {fpath} (exit code {rc})", flush=True)
                failures.append(fpath)

    if failures:
        print("The following files failed:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("All files processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

