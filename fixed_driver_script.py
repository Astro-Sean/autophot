#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run automated photometry with autophot using dynamic filter system.
Fixed version that properly uses transmission curves with arbitrary filter names.
"""

import sys
import argparse
sys.path.append('/home/sbrennan/Documents/autophot_object')

import autophot_tokens
from autophot import AutomatedPhotometry


# -----------------------------------------------------------------------------
# CLI and load defaults
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run AutoPhOT with optional image-level parallelism via nCPU."
)
parser.add_argument(
    "--ncpu",
    type=int,
    default=1,
    help="Number of images to process in parallel (nCPU>1 enables multiprocessing inside AutoPhOT).",
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Load defaults and set paths
# -----------------------------------------------------------------------------
autophot_input = AutomatedPhotometry.load()

autophot_input['outdir_name'] = 'REDUCED'
autophot_input['wdir'] = '/home/sbrennan/Desktop/autophot_db'

# =============================================================================
# BASIC SETTINGS
# =============================================================================
autophot_input['restart'] = False
autophot_input['target_name'] = 'SN2026gzf'
autophot_input['fits_dir'] = '/home/sbrennan/Desktop/SN2026gzf/panstarrs'

# =============================================================================
# DYNAMIC FILTER SYSTEM CONFIGURATION
# =============================================================================
# CORRECT: Use 'gaia_custom' for transmission curves (NOT 'custom')
autophot_input['catalog']['use_catalog'] = 'gaia_custom'

# This now works with ANY filter names - no restrictions!
autophot_input['catalog']['transmission_curve_map'] = {
    'g': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.g.dat',
    'r': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.r.dat',
    'i': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.i.dat',
    'z': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.z.dat',
    'w': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.w.dat',
    'y': '/home/sbrennan/Desktop/SN2026gzf/panstarrs_transmission_curves/PAN-STARRS_PS1.y.dat',
    # You can add ANY filter names here - no restrictions!
    # 'my_custom_filter': '/path/to/my_custom.dat',
    # 'NIR_band': '/path/to/nir.dat',
    # 'special_survey': '/path/to/survey.dat',
}

# Gaia-specific settings for transmission curves
autophot_input['catalog']['gaia_curve_map_max_sources'] = 100
autophot_input['catalog']['gaia_curve_map_order_by'] = 'distance'
autophot_input['catalog']['gaia_xp_radius_deg'] = 0.1667
autophot_input['catalog']['gaia_xp_max_sources'] = 5000

# Alternative: Use custom catalog instead (no transmission curves)
# Uncomment this and comment out the above if you have a custom CSV catalog:
# autophot_input['catalog']['use_catalog'] = 'custom'
# autophot_input['catalog']['catalog_custom_fpath'] = '/path/to/my_custom_catalog.csv'

# =============================================================================
# AUTHENTICATION
# =============================================================================
autophot_input['catalog']['MASTcasjobs_wsid'] = autophot_tokens.MASTcasjobs_wsid
autophot_input['catalog']['MASTcasjobs_pwd'] = autophot_tokens.MASTcasjobs_pwd

# =============================================================================
# PREPROCESSING AND COSMIC RAYS
# =============================================================================
autophot_input['cosmic_rays']['remove_cmrays'] = 0
autophot_input['preprocessing']['trim_image'] = 3   # arcmin box

# =============================================================================
# PHOTOMETRY
# =============================================================================
autophot_input['photometry']['perform_emcee_fitting_s2n'] = 10
autophot_input['photometry']['check_inverted_image'] = True

# =============================================================================
# WCS
# =============================================================================
autophot_input['wcs']['redo_wcs'] = 1

# =============================================================================
# LIMITING MAGNITUDE
# =============================================================================
autophot_input['limiting_magnitude']['recovery_method'] = 'PSF'

# =============================================================================
# TEMPLATE SUBTRACTION
# =============================================================================
autophot_input['template_subtraction']['do_subtraction'] = 1
autophot_input['template_subtraction']['alignment_method'] = 'swarp'
autophot_input['template_subtraction']['method'] = 'sfft'
autophot_input['template_subtraction']['kernel_order'] = 0
autophot_input['template_subtraction']['download_templates'] = 'panstarrs'

# =============================================================================
# PARALLELISM CONTROL
# =============================================================================
autophot_input['nCPU'] = int(args.ncpu)

# =============================================================================
# RUN PHOTOMETRY
# =============================================================================
print("Starting AutoPHOT with dynamic filter system...")
print(f"Using catalog: {autophot_input['catalog']['use_catalog']}")
print(f"Transmission curves: {len(autophot_input['catalog']['transmission_curve_map'])} filters")

loc = AutomatedPhotometry.run_photometry(
    default_input=autophot_input,
)

# =============================================================================
# POST-RUN: LIGHTCURVE AND TABLES
# =============================================================================
if loc:
    from lightcurve import (
        plot_lightcurve,
        check_detection_plots,
        generate_photometry_table,
    )

    print("Generating lightcurve and tables...")
    detections_loc = plot_lightcurve(
        loc,
        method='PSF',
        format='png',
        offset=1,
        show=True,
        plot_color=False,
        color_match_days=0.5,
    )
    check_detection_plots(detections_loc, method='PSF')
    generate_photometry_table(
        loc,
        snr_limit=3,
        method='PSF',
        reference_epoch=0,
    )
    print("Done!")
else:
    print("Error: photometry run failed")
