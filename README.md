AutoPHOT Object Photometry Pipeline
===================================

AutoPHOT is a Python pipeline for performing calibrated aperture and PSF photometry on CCD/NIR imaging data. It is designed for time-domain astronomy workflows (e.g. supernovae, transients) and supports end-to-end processing from raw science frames to calibrated light curves and diagnostic plots.

## Features

- **Image preparation**: trimming, background modeling and subtraction, optional north-up reprojection, cosmic-ray removal.
- **WCS handling**: validation, (re)solution, and refinement of WCS headers.
- **Source handling**: reference catalog querying, FWHM estimation, optimal aperture selection.
- **Photometry**:
  - Aperture photometry with aperture corrections.
  - ePSF / PSF photometry with diagnostic plots for PSF quality.
- **Calibration**:
  - Zeropoint fitting with robust clipping and optional colour terms.
  - Header updates with calibrated zeropoints and metadata.
- **Template subtraction**: science–template alignment and subtraction, with subtraction quality plots.
- **Diagnostics and plots**: consistently styled plots for PSF, background, zeropoints, subtraction checks, and more.

## Repository layout (key files)

- `main.py` – command-line entry point; orchestrates the full photometry pipeline.
- `prepare.py` – prepares and validates input FITS files and directory layout.
- `check.py` – inspects FITS headers and helps build telescope/instrument databases.
- `background.py` – background estimation and diagnostic plots.
- `aperture.py` – aperture photometry and aperture-correction handling.
- `psf.py` – PSF/ePSF construction, fitting, and PSF-photometry diagnostics.
- `zeropoint.py` – zeropoint and colour-term fitting from reference catalogs.
- `plot.py` – subtraction and source-diagnostics plotting helpers.
- `lightcurve.py` – light-curve assembly utilities.
- `autophot.mplstyle` – shared Matplotlib style used by all plots.
- `requirements.txt` – Python dependencies.

Most modules have module-level docstrings and per-function / per-class docstrings that explain intent and non-obvious logic. Inline comments are used sparingly to clarify tricky parts of the algorithms without repeating the code.

## Installation

### Option A: pip install (recommended)

Install into an existing environment (venv/conda):

```bash
pip install -e .
```

If installing directly from a GitHub clone without editable mode:

```bash
pip install .
```

### Option B: conda (local build)

This repository includes a minimal conda recipe in `conda/recipe/`.
To build and install locally:

```bash
conda install -c conda-forge conda-build
conda build conda/recipe
```

Then install the resulting package from your local conda-bld directory (printed by conda-build), or upload it to your own channel.

### Option C: create a fresh conda environment

```bash
conda env create -f environment.yml
conda activate autophot-object
```

### Notes

- If you use the system Python on Debian/Ubuntu, prefer conda or ensure `python3-venv` is installed before using `python -m venv`.

## Legacy installation (venv)

1. **Create and activate a virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

AutoPHOT targets a recent Python 3 version (3.9+ is recommended).

## Basic usage

The main entry point is `run_photometry()` in `main.py`, which is wired to a simple CLI:

```bash
python main.py -f /path/to/image.fits -c /path/to/config.yml
```

- `-f` – path to the science FITS image.
- `-c` – path to the YAML configuration file describing the observation, instrument, and pipeline options.
- `-temp` – optional flag to prepare a template rather than a science image.

The configuration YAML controls details such as:

- Working directory and output directory naming.
- WCS solving and refinement options.
- Background modelling choices.
- Aperture vs PSF photometry settings.
- Zeropoint and colour-term fitting options.
- Template subtraction settings.

See the existing configuration examples under `databases/` (e.g. `default_input.yml`) for typical values.

## Recommended usage (Python driver script)

For most real datasets you’ll want a small “driver” script that:

- Loads `databases/default_input.yml` defaults
- Overrides the few fields that are specific to your target/dataset (paths, target coords, catalog, template subtraction, etc.)
- Runs the pipeline (optionally with image-level parallelism via `nCPU`)
- Optionally generates lightcurves/tables after the run

Below is a cleaned, copy‑pasteable template based on the script you shared (replace paths/coordinates as needed).

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run AutoPHOT with a small driver script.
Override defaults and run the pipeline; optionally plot lightcurve and tables.
"""

import argparse

from autophot import AutomatedPhotometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AutoPHOT with optional image-level parallelism via nCPU."
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of images to process in parallel (nCPU>1 enables multiprocessing inside AutoPHOT).",
    )
    args = parser.parse_args()

    # Load defaults from databases/default_input.yml
    autophot_input = AutomatedPhotometry.load()

    # ---------------------------------------------------------------------
    # Paths / output
    # ---------------------------------------------------------------------
    autophot_input["outdir_name"] = "REDUCED"
    autophot_input["wdir"] = "/path/to/autophot_db"     # working directory
    autophot_input["fits_dir"] = "/path/to/images/"     # folder containing science FITS files

    # Optional: set to True/False to restart operations from scratch
    # autophot_input["restart"] = False

    # ---------------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------------
    autophot_input["target_name"] = "SN2024xuo"
    autophot_input["target_ra"] = 152.207497
    autophot_input["target_dec"] = -67.047493

    # ---------------------------------------------------------------------
    # Catalog
    # ---------------------------------------------------------------------
    autophot_input["catalog"]["use_catalog"] = "refcat"
    # Other options: 'sdss', 'gaia', 'tic', 'pan_starrs', 'apass', 'skymapper', 'custom'
    # autophot_input["catalog"]["catalog_custom_fpath"] = "/path/to/custom.csv"

    # If you need Refcat via MAST CasJobs, prefer env vars (recommended):
    #   export MASTCASJOBS_WSID="..."
    #   export MASTCASJOBS_PWD="..."
    #
    # Or provide values in the driver script:
    # autophot_input["catalog"]["MASTcasjobs_wsid"] = ...
    # autophot_input["catalog"]["MASTcasjobs_pwd"] = ...

    # ---------------------------------------------------------------------
    # Preprocessing / cosmic rays
    # ---------------------------------------------------------------------
    autophot_input["cosmic_rays"]["remove_cmrays"] = True
    autophot_input["preprocessing"]["trim_image"] = 5  # arcmin box (if enabled in your config)

    # ---------------------------------------------------------------------
    # Photometry
    # ---------------------------------------------------------------------
    autophot_input["photometry"]["psf_oversample"] = 1
    autophot_input["photometry"]["perform_emcee_fitting_s2n"] = 50
    autophot_input["photometry"]["redo_sources"] = False
    autophot_input["photometry"]["fitting_xy_bounds"] = 1

    # ---------------------------------------------------------------------
    # WCS
    # ---------------------------------------------------------------------
    autophot_input["wcs"]["redo_wcs"] = True
    autophot_input["wcs"]["solve_field_exe_loc"] = "solve-field"

    # TNS credentials (recommended: env vars)
    #   export TNS_BOT_ID="..."
    #   export TNS_BOT_NAME="..."
    #   export TNS_BOT_API="..."

    # ---------------------------------------------------------------------
    # Template subtraction
    # ---------------------------------------------------------------------
    autophot_input["template_subtraction"]["do_subtraction"] = True
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"
    autophot_input["template_subtraction"]["method"] = "sfft"  # or "hotpants"
    autophot_input["template_subtraction"]["hotpants_exe_loc"] = "hotpants"
    autophot_input["template_subtraction"]["inpaint_template_cores"] = True

    # ---------------------------------------------------------------------
    # Parallelism control (image-level)
    # ---------------------------------------------------------------------
    autophot_input["nCPU"] = max(1, int(args.ncpu))

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    loc = AutomatedPhotometry.run_photometry(default_input=autophot_input)

    # ---------------------------------------------------------------------
    # Post-run: lightcurve and tables
    # ---------------------------------------------------------------------
    from lightcurve import plot_lightcurve, check_detection_plots, generate_photometry_table

    detections_loc = plot_lightcurve(
        loc,
        snr_limit=3,
        method="PSF",
        format="png",
        offset=1,
        show=True,
        plot_color=True,
        color_match_days=0.5,
    )
    check_detection_plots(detections_loc, method="PSF")
    generate_photometry_table(
        loc,
        snr_limit=3,
        method="PSF",
        reference_epoch=0,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Notes on driver scripts

- If you installed the package (`pip install -e .`), **do not** use `sys.path.append(...)` in your driver script.
- `nCPU>1` parallelises over images (processes). If you run on HPC, keep BLAS/OpenMP threads low (the code sets common env vars to `1` at startup).
- If you use template subtraction, make sure your template exists under `fits_dir/templates/<filter>p_template/` (see the “Minimal example” section below).

### Minimal example: reference-subtracted (template) photometry

To run **reference-subtracted photometry** (science image minus a reference/template, then photometry on the difference), use the following minimal setup.

**1. Directory layout**

Place your science FITS files in a directory and put the reference image in a filter-matched subfolder under `templates/`:

```
my_field/                          # fits_dir in config
├── science_2024_01_15_r.fits      # science images (any name)
├── science_2024_01_20_r.fits
└── templates/
    └── rp_template/               # for r-band (use gp_template, ip_template, etc. for g, i, …)
        └── my_reference_template.fits   # name must contain "_template", not "PSF_model" or ".weight"
```

For filters `g`, `r`, `i`, `z`, `u` the code expects a subfolder named `{filter}p_template` (e.g. `rp_template` for r). Other filters use `{filter}_template` (e.g. `J_template` for 2MASS J).

**2. Minimal config YAML**

Start from `databases/default_input.yml` and set at least:

```yaml
default_input:
  fits_dir: /path/to/my_field
  outdir_name: REDUCED
  target_ra: 123.456        # degrees (target position)
  target_dec: -45.678       # degrees

  templates:
    use_user_template: True

  template_subtraction:
    do_subtraction: True
    alignment_method: reproject   # or swarp, astroalign
    method: sfft                  # or zogy, hotpants
```

**3. Optional: preprocess templates**

If your reference images need the same preprocessing as science (WCS, FWHM, cosmic-ray removal), run once with `prepare_templates: True` and point the pipeline at the template directory so it builds WCS and writes preprocessed templates; then set `prepare_templates: False` for normal science runs.

**4. Run**

```bash
python main.py -f /path/to/my_field/science_2024_01_15_r.fits -c /path/to/config.yml
```

The pipeline will find the template under `my_field/templates/rp_template/`, align it to the science image, perform the subtraction (SFFT by default), and then run photometry on the difference image. Subtraction quicklook PDFs are saved when `template_subtraction.save_subtraction_quicklook` is `True`. If you use ZOGY, set `photometry.save_PSF_models_fits: True` in the config so PSF models are available for the subtraction.

## Comments and documentation style

- Each major module (`main.py`, `prepare.py`, `check.py`, `psf.py`, etc.) starts with a high-level docstring describing its role in the pipeline.
- Public classes and key functions have docstrings that explain:
  - The purpose of the function or class.
  - The most important arguments and return values.
  - Any non-obvious assumptions or algorithmic choices.
- Inline comments are reserved for **non-obvious logic**, e.g.:
  - Why a particular clipping threshold or scaling choice is used.
  - How restart behaviour or directory mirroring is implemented.
  - Corner cases in PSF building or background estimation.

This style keeps the code readable and maintainable without cluttering it with comments that simply restate what each line of code does.

## Preparing for publication on GitHub

Before pushing this repository to GitHub, you should:

- **Review configuration and token files**:
  - Do not commit secrets such as API keys or authentication tokens (e.g. `autophot_tokens.py` or similar).
  - If such files are needed at runtime, document them in the README and ignore them via `.gitignore`.

### Credentials (recommended: environment variables)

AutoPHOT can read credentials from environment variables (preferred) or from a local override file.

- **Environment variables**:
  - `TNS_BOT_ID`, `TNS_BOT_NAME`, `TNS_BOT_API` (or `TNS_BOT_API_KEY`)
  - `MASTCASJOBS_WSID`, `MASTCASJOBS_PWD`
- **Local override file**:
  - Copy `autophot_tokens_local.py.example` → `autophot_tokens_local.py` and fill it in.
  - `autophot_tokens_local.py` is ignored by git via `.gitignore`.
- **Check example data**:
  - Large or proprietary FITS files should normally be excluded from version control.
  - If you want to ship small example data, keep it minimal and well documented.
- **Add a LICENSE**:
  - Choose an appropriate open-source license (e.g. MIT, BSD-3-Clause, GPL) and add a `LICENSE` file so others know how they may use the code.

## Contributing

Contributions and issue reports are welcome. When opening a pull request:

- Keep functions small and focused.
- Add / update docstrings when you touch non-trivial logic.
- Maintain the consistent plotting style (via `autophot.mplstyle` and the shared `set_size` helper).

