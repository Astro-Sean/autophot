#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIPER/HPC entry point for AutoPhOT.

Reads per-transient settings from:
  1. A local file autophot_job_params.yml in the current directory (for bookkeeping:
     run from the transient folder with ready-to-go params), or
  2. Environment variables (set by the Slurm driver when submitting each job).

Required env vars when not using a local params file:
  AUTOPHOT_FITS_DIR, AUTOPHOT_TARGET_RA, AUTOPHOT_TARGET_DEC, AUTOPHOT_NAME

Optional: --ncpu N (default 1).
"""
import argparse
import os
import shutil
import sys

# Ensure we can import autophot from this repo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from autophot import AutomatedPhotometry  # noqa: E402
from check import FitsInfo  # type: ignore

# Params file written into each transient folder for bookkeeping
PARAMS_FILENAME = "autophot_job_params.yml"


def _load_params_from_file(cwd: str):
    """Load fits_dir, target_ra, target_dec, target_name, nCPU from local YAML if present."""
    path = os.path.join(cwd, PARAMS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data or "fits_dir" not in data:
            return None
        return {
            "fits_dir": os.path.abspath(os.path.join(cwd, data["fits_dir"]))
            if not os.path.isabs(data["fits_dir"])
            else data["fits_dir"],
            "target_ra": float(data["target_ra"]),
            "target_dec": float(data["target_dec"]),
            "target_name": str(data["target_name"]).strip(),
            "nCPU": int(data.get("nCPU", 1)),
        }
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoPhOT for one transient (VIPER).")
    parser.add_argument("--ncpu", type=int, default=None, help="Override nCPU (default from params or env).")
    args = parser.parse_args()

    # Prefer local params file (bookkeeping: run from transient folder)
    cwd = os.getcwd()
    params = _load_params_from_file(cwd)
    if params is None:
        # Fall back to environment variables
        fits_dir = os.environ.get("AUTOPHOT_FITS_DIR")
        ra_str = os.environ.get("AUTOPHOT_TARGET_RA")
        dec_str = os.environ.get("AUTOPHOT_TARGET_DEC")
        name = os.environ.get("AUTOPHOT_NAME")
        if not fits_dir or ra_str is None or dec_str is None or not name:
            print(
                "Error: Set AUTOPHOT_FITS_DIR, AUTOPHOT_TARGET_RA, AUTOPHOT_TARGET_DEC, "
                "and AUTOPHOT_NAME, or run from a transient folder containing "
                f"{PARAMS_FILENAME}.",
                file=sys.stderr,
            )
            sys.exit(1)
        fits_dir = os.path.abspath(os.path.expanduser(fits_dir))
        try:
            ra = float(ra_str)
            dec = float(dec_str)
        except ValueError:
            print(
                f"Error: Invalid AUTOPHOT_TARGET_RA or AUTOPHOT_TARGET_DEC: {ra_str!r}, {dec_str!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        params = {
            "fits_dir": fits_dir,
            "target_ra": ra,
            "target_dec": dec,
            "target_name": name.strip(),
            "nCPU": args.ncpu if args.ncpu is not None else 1,
        }
    else:
        if args.ncpu is not None:
            params["nCPU"] = args.ncpu

    fits_dir = params["fits_dir"]
    if not os.path.isdir(fits_dir):
        print(f"Error: fits_dir is not a directory: {fits_dir}", file=sys.stderr)
        sys.exit(1)

    # Capture all print/output to a per-transient log file: {name}_out.log
    safe_name = params["target_name"].replace("/", "_").replace("\\", "_").strip() or "transient"
    log_dir = os.path.dirname(fits_dir)
    log_path = os.path.join(log_dir, f"{safe_name}_out.log")
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _log_file
    sys.stderr = _log_file
    print(f"Log file: {log_path}")

    try:
        ap = AutomatedPhotometry()
        config = ap.load()

        # Core per-transient settings
        config["fits_dir"] = fits_dir
        config["wdir"] = fits_dir  # where telescope.yml is looked for (fits_dir/telescope.yml)
        config["target_ra"] = params["target_ra"]
        config["target_dec"] = params["target_dec"]
        config["target_name"] = params["target_name"]
        config["nCPU"] = params["nCPU"]

        # Inject REFCAT (MAST CasJobs) credentials from autophot_tokens if available,
        # so we never have to hard-code secrets in default_input.yml.
        try:
            import autophot_tokens  # type: ignore

            config.setdefault("catalog", {})
            config["catalog"]["MASTcasjobs_wsid"] = getattr(
                autophot_tokens,
                "MASTcasjobs_wsid",
                config["catalog"].get("MASTcasjobs_wsid"),
            )
            config["catalog"]["MASTcasjobs_pwd"] = getattr(
                autophot_tokens,
                "MASTcasjobs_pwd",
                config["catalog"].get("MASTcasjobs_pwd"),
            )
        except Exception:
            # If tokens aren't available, leave as-is; refcat branch will complain clearly.
            pass

        # Make batch runs fully non-interactive by disabling header/telescope checks.
        config["ignore_no_telescop"] = True
        config["skip_file_check"] = True

        # Use prebuilt catalog (from run_sample_VIPER prebuild); default to refcat if unset.
        if "catalog" not in config:
            config["catalog"] = {}
        config["catalog"]["use_catalog"] = config["catalog"].get("use_catalog") or "refcat"
        config["catalog"]["build_catalog"] = False

        # Enable template subtraction using user-supplied templates under
        # <fits_dir>/templates/<filter>_template/, matching the layout created by
        # run_sample_VIPER.py.
        config.setdefault("templates", {})
        config["templates"]["use_user_template"] = True
        config["templates"]["get_PS1_template"] = False

        config.setdefault("template_subtraction", {})
        config["template_subtraction"]["do_subtraction"] = True
        # Required for main.py: align() and subtract() need these keys (not in default_input.yml).
        config["template_subtraction"].setdefault("alignment_method", "reproject")
        config["template_subtraction"].setdefault("method", "hotpants")
        # Keep prepare_templates False so the main clean() pass looks at the
        # science images (fits_dir), not the templates tree; template discovery
        # is still handled via find_templates().
        config["template_subtraction"]["prepare_templates"] = False
        # Ensure we never try to download templates on the Slurm node.
        config["template_subtraction"]["download_templates"] = False

        # Make all FitsInfo-based checks non-interactive by overriding ask_question
        # so any prompts (e.g. telescope location) automatically take the default.
        def _noninteractive_ask(self, question, default_answer="n", expect_answer_type=str,
                                options=None, ignore_word="skip"):
            return default_answer

        try:
            FitsInfo.ask_question = _noninteractive_ask  # type: ignore[assignment]
        except Exception:
            # If monkeypatching fails for some reason, continue; worst case, the
            # code behaves as before and may prompt.
            pass

        # Prefer SCAMP-based WCS solving for crowded, blurry VIPER fields when available.
        config.setdefault("wcs", {})
        config["wcs"].setdefault("solver", "scamp")

        output = ap.run_photometry(config, do_photometry=True)
        print(f"Output light curve: {output}")

        # Copy lightcurve output to central output folder for bookkeeping
        output_root = os.environ.get(
            "AUTOPHOT_OUTPUT_DIR",
            "/u/sbrennan/Precursor_Sample/output",
        )
        if output_root:
            output_root = os.path.abspath(os.path.expanduser(output_root))
            reduced_dir = os.path.dirname(output)
            dest_dir = os.path.join(output_root, params["target_name"])
            try:
                os.makedirs(output_root, exist_ok=True)
                if os.path.isdir(reduced_dir):
                    shutil.copytree(reduced_dir, dest_dir, dirs_exist_ok=True)
                    print(f"Copied output to {dest_dir}")
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(output, os.path.join(dest_dir, os.path.basename(output)))
                    print(f"Copied {output} to {dest_dir}")
            except Exception as e:
                print(f"Warning: Could not copy to {dest_dir}: {e}", file=sys.stderr)
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()


if __name__ == "__main__":
    main()
