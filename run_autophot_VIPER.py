#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic VIPER AutoPhOT runner.
Reads configuration from autophot_job_params.yml in the current directory
and runs AutomatedPhotometry.
"""

import os
import sys
import argparse
import yaml

# Ensure we can import autophot from the repo
AUTOPHOT_DIR = "/u/sbrennan/autophot"
if AUTOPHOT_DIR not in sys.path:
    sys.path.insert(0, AUTOPHOT_DIR)

import autophot_tokens
from autophot import AutomatedPhotometry


def load_params_from_yaml(filepath="autophot_job_params.yml"):
    """Load AutoPhOT parameters from YAML file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Params file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoPhOT VIPER pipeline from autophot_job_params.yml"
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of images to process in parallel",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="autophot_job_params.yml",
        help="Path to YAML params file",
    )
    args = parser.parse_args()

    # Load parameters from YAML
    params = load_params_from_yaml(args.params)

    # Load AutoPhOT defaults and merge with params
    autophot_input = AutomatedPhotometry.load()

    # Override with params from YAML
    for key, value in params.items():
        if key == "catalog" and isinstance(value, dict):
            # Merge catalog settings
            for cat_key, cat_value in value.items():
                autophot_input["catalog"][cat_key] = cat_value
        else:
            autophot_input[key] = value

    # Set nCPU from CLI if provided
    autophot_input["nCPU"] = args.ncpu

    autophot_input['wcs']['redo_wcs'] = False

    # Add credentials from tokens
    autophot_input["catalog"]["MASTcasjobs_wsid"] = autophot_tokens.MASTcasjobs_wsid
    autophot_input["catalog"]["MASTcasjobs_pwd"] = autophot_tokens.MASTcasjobs_pwd
    autophot_input['limiting_magnitude']['recovery_method'] = 'PSF'

    # -----------------------------------------------------------------------------
    # Template subtraction
    # -----------------------------------------------------------------------------
    autophot_input['template_subtraction']['do_subtraction'] = 1
    autophot_input['template_subtraction']['alignment_method'] = 'swarp'

    autophot_input['template_subtraction']['method'] = 'sfft'
    autophot_input['template_subtraction']['kernel_order'] = 0

    # -----------------------------------------------------------------------------
    # Parallelism control (image-level)
    # -----------------------------------------------------------------------------
    autophot_input['nCPU'] = int(args.ncpu)

    autophot_input["photometry"]["check_inverted_image"] = True
    autophot_input["photometry"]["apply_colorterms"] =False

    print(f"[INFO] Running AutoPhOT with fits_dir: {autophot_input.get('fits_dir')}")
    print(f"[INFO] Working directory (run_sample_VIPER.py): {autophot_input.get('wdir')}")
    print(f"[INFO] Output directory suffix: {autophot_input.get('outdir_name')}")
    print(f"[INFO] Target: {autophot_input.get('target_name')}")

    # Run photometry
    loc = AutomatedPhotometry.run_photometry(default_input=autophot_input)

    from lightcurve import (
        plot_lightcurve,
        check_detection_plots,
        generate_photometry_table,
    )

    detections_loc = plot_lightcurve(
        loc,
        snr_limit=3,
        method='PSF',
        format='png',
        offset=1,
        show=True,
        plot_color=False,
        color_match_days=0.5,
        # use_SNR_limit=True,
        # show_limits=False,
    )
    check_detection_plots(detections_loc, method='PSF')
    generate_photometry_table(
        loc,
        snr_limit=3,
        method='PSF',
        reference_epoch=0,
    )


    print(f"[SUCCESS] AutoPhOT completed. Output at: {loc}")


if __name__ == "__main__":
    main()
