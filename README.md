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

## External dependencies (optional but common)

AutoPHOT can run without some external tools, but the following are commonly used:

### Astrometry.net (`solve-field`) for WCS solving

- **Install (conda-forge, recommended on Linux/HPC)**:

```bash
conda install -c conda-forge astrometry-net
```

- **Verify**:

```bash
solve-field --help
```

- **Index files (required)**:
  - `solve-field` needs astrometry index files (can be tens of GB depending on which sets you download).
  - Index files are listed at [`https://data.astrometry.net/`](https://data.astrometry.net/).
  - Set the index location via environment variables (common) or by installing them into the default search path used by your build.

One common approach is to download a specific index series into a directory, then point astrometry.net at it:

```bash
mkdir -p /path/to/astrometry_index
# download appropriate index files for your image scale/FOV into that directory
# (example: 4200-series; downloads all .fits in that directory)
cd /path/to/astrometry_index
wget -c -nd -r -np -A "*.fits" "https://data.astrometry.net/4200/"
export ASTROMETRY_NET_DATA_DIR="/path/to/astrometry_index"
```

If `solve-field` is missing, AutoPHOT will warn and skip WCS solving (unless you force it).

### HOTPANTS for template subtraction

AutoPHOT can call HOTPANTS for image subtraction when `template_subtraction.method: hotpants`.

- **Install dependencies** (CFITSIO is required):

```bash
conda install -c conda-forge cfitsio make gcc
```

- **Build HOTPANTS from source**:

```bash
git clone https://github.com/acbecker/hotpants
cd hotpants
make
```

AutoPHOT defaults to running the `hotpants` command from your `PATH`, and prints a warning if it cannot be found.

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
    # outdir_name: the output folder suffix. If fits_dir is `/data/field`, outputs
    # go to `/data/field_REDUCED` when outdir_name is `REDUCED`.
    autophot_input["outdir_name"] = "REDUCED"

    # wdir: working directory used for pipeline scratch/products (catalog caches,
    # intermediate files). Set this to a persistent location you can write to.
    autophot_input["wdir"] = "/path/to/autophot_db"

    # fits_dir: directory containing the science FITS images to process.
    # The pipeline writes results into a sibling directory: `{fits_dir}_{outdir_name}`.
    autophot_input["fits_dir"] = "/path/to/images/"

    # Optional: set to True/False to restart operations from scratch
    # autophot_input["restart"] = False

    # ---------------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------------
    # target_name: optional label used in filenames/plots (and for TNS lookups if configured).
    autophot_input["target_name"] = "SN2024xuo"

    # target_ra / target_dec: target sky position in *degrees* (J2000).
    # If these are wrong, forced photometry/trim boxes will be wrong.
    autophot_input["target_ra"] = 152.207497
    autophot_input["target_dec"] = -67.047493

    # ---------------------------------------------------------------------
    # Catalog
    # ---------------------------------------------------------------------
    # catalog.use_catalog: which reference catalog to calibrate against.
    # `refcat` uses MAST CasJobs credentials; others are survey-query based.
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
    # cosmic_rays.remove_cmrays: enable/disable cosmic-ray cleaning.
    autophot_input["cosmic_rays"]["remove_cmrays"] = True

    # preprocessing.trim_image (arcmin): if >0, trim the image to a square box
    # of size (2*trim_image) arcmin on a side, centred on the target.
    # Use 0 to disable trimming.
    autophot_input["preprocessing"]["trim_image"] = 5

    # ---------------------------------------------------------------------
    # WCS
    # ---------------------------------------------------------------------
    # wcs.redo_wcs: if True, attempt to (re)solve WCS (recommended when WCS is unreliable).
    autophot_input["wcs"]["redo_wcs"] = True

    # wcs.solve_field_exe_loc: path/name of astrometry.net `solve-field`.
    # Leave as `solve-field` if it is on PATH.
    autophot_input["wcs"]["solve_field_exe_loc"] = "solve-field"

    # TNS credentials (recommended: env vars)
    #   export TNS_BOT_ID="..."
    #   export TNS_BOT_NAME="..."
    #   export TNS_BOT_API="..."

    # ---------------------------------------------------------------------
    # Template subtraction
    # ---------------------------------------------------------------------
    # template_subtraction.do_subtraction: enable/disable template subtraction.
    autophot_input["template_subtraction"]["do_subtraction"] = True

    # template_subtraction.alignment_method: how to align template -> science grid.
    # `reproject` uses WCS; other options include `swarp` and `astroalign`.
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"

    # template_subtraction.method: subtraction backend.
    # `sfft` is pure-Python; `hotpants` requires the external HOTPANTS executable.
    autophot_input["template_subtraction"]["method"] = "sfft"

    # ---------------------------------------------------------------------
    # Parallelism control (image-level)
    # ---------------------------------------------------------------------
    # nCPU: image-level parallelism (number of images processed in parallel).
    # Use 1 for debugging; increase for large datasets if memory allows.
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

### Template setup (detailed)

AutoPHOT expects templates to live under a `templates/` directory inside your `fits_dir`. At runtime it chooses the template based on the **science image filter** and the folder naming conventions below.

#### 1) Directory structure and naming rules

Templates must be organised into per-filter subfolders. The folder name depends on the filter:

- For common optical filters **`g r i z u`**:
  - Folder name: **`{filter}p_template/`**
  - Examples: `gp_template/`, `rp_template/`, `ip_template/`, `zp_template/`, `up_template/`
- For **Johnson–Cousins UBVRI** (use uppercase band letters **`U B V R I`**):
  - Folder name: **`{filter}_template/`**
  - Examples: `U_template/`, `B_template/`, `V_template/`, `R_template/`, `I_template/`
- For other filters (e.g. NIR **`J H K`**, custom filters, instrument-specific names):
  - Folder name: **`{filter}_template/`**
  - Examples: `J_template/`, `H_template/`, `K_template/`

Example layout:

```
my_field/                         # fits_dir
├── science_2024_01_15_r.fits
├── science_2024_01_20_r.fits
└── templates/
    ├── rp_template/
    │   └── r_template.fits
    └── K_template/
        └── K_template.fits
```

#### 2) Template filename requirements

- The template can be **any `*.fits` file** in the appropriate `{filter}_template/` folder (the filename does **not** need to contain `_template`).
- Files containing `PSF_model` or ending in `.weight` are ignored as templates.
- Keep exactly **one** usable template per filter folder to avoid ambiguity.

#### 3) Using a user-provided template vs downloading one

- **User-provided template**:
  - Place the template FITS into the correct folder under `templates/` (as above)
  - No special config is required beyond enabling subtraction (AutoPHOT will discover templates in `fits_dir/templates/...` automatically):

```python
autophot_input["template_subtraction"]["do_subtraction"] = True
```

- **Downloaded template**:
  - AutoPHOT can download templates into `fits_dir/templates/` before running subtraction.
  - Select the download source via `template_subtraction.download_templates`:

```python
autophot_input["template_subtraction"]["do_subtraction"] = True
autophot_input["template_subtraction"]["download_templates"] = "panstarrs"  # or "legacy", "sdss"
autophot_input["template_subtraction"]["templates_size"] = 10               # arcmin cutout size
```

#### 4) Template preparation

When template folders are present/used, AutoPHOT will automatically apply the required preparation steps for template subtraction (e.g. WCS checks, preprocessing steps required by the selected alignment/subtraction backend).

#### 5) Alignment method (science–template registration)

Alignment is controlled by:

```yaml
default_input:
  template_subtraction:
    alignment_method: reproject  # or swarp, astroalign
```

- **`reproject`**: WCS-based registration (recommended when WCS is good; robust and deterministic)
- **`swarp`**: external SWarp-based registration (useful in some survey-like workflows)
- **`astroalign`**: feature-matching registration (can help when WCS is poor)

#### 6) Subtraction backend

Choose the subtraction backend with:

```yaml
default_input:
  template_subtraction:
    method: hotpants  # or sfft, zogy (if configured)
```

- **`hotpants`**: requires the external HOTPANTS executable (see install section above)
- **`sfft`**: uses the Python SFFT backend

If you use the SFFT backend, please cite the SFFT method and see upstream documentation/source:

- Repo: `https://github.com/thomasvrussell/sfft`
- ADS (Hu et al. 2022, ApJ 936, 157): `https://ui.adsabs.harvard.edu/abs/2022ApJ...936..157H/abstract`

**1. Directory layout**

Place your science FITS files in a directory and put the reference image in a filter-matched subfolder under `templates/`:

```
my_field/                          # fits_dir in config
├── science_2024_01_15_r.fits      # science images (any name)
├── science_2024_01_20_r.fits
└── templates/
    └── rp_template/               # for r-band (use gp_template, ip_template, etc. for g, i, …)
        └── my_reference_template.fits   # any *.fits is OK; files containing "PSF_model" are ignored
```

For filters `g`, `r`, `i`, `z`, `u` the code expects a subfolder named `{filter}p_template` (e.g. `rp_template` for r). Other filters (including **UBVRI** and NIR like `JHK`) use `{filter}_template` (e.g. `V_template`, `J_template`).

**2. Minimal config YAML**

Start from `databases/default_input.yml` and set at least:

```yaml
default_input:
  fits_dir: /path/to/my_field
  outdir_name: REDUCED
  target_ra: 123.456        # degrees (target position)
  target_dec: -45.678       # degrees

  template_subtraction:
    do_subtraction: True
    alignment_method: reproject   # or swarp, astroalign
    method: sfft                  # or zogy, hotpants
```

**3. Run**

```bash
python main.py -f /path/to/my_field/science_2024_01_15_r.fits -c /path/to/config.yml
```

The pipeline will find the template under `my_field/templates/rp_template/`, align it to the science image, perform the subtraction (SFFT by default), and then run photometry on the difference image. Subtraction quicklook PDFs are saved when `template_subtraction.save_subtraction_quicklook` is `True`. If you use ZOGY, set `photometry.save_PSF_models_fits: True` in the config so PSF models are available for the subtraction.

## Citing AutoPHOT

If you use AutoPHOT in a publication, please cite:

- ADS: `https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B/abstract`

BibTeX:

```bibtex
@ARTICLE{2022A&A...667A..62B,
       author = {{Brennan}, S.~J. and {Fraser}, M.},
        title = "{The Automated Photometry of Transients pipeline (AUTOPHOT)}",
      journal = {\\aap},
     keywords = {techniques: photometric, techniques: image processing, methods: data analysis, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2022,
        month = nov,
       volume = {667},
          eid = {A62},
        pages = {A62},
          doi = {10.1051/0004-6361/202243067},
archivePrefix = {arXiv},
       eprint = {2201.02635},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022A&A...667A..62B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

