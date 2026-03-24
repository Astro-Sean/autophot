AutoPHoT
========

The Automated photometry of transients

AutoPHoT is a Python pipeline for publication-quality photometry of transients and variable sources. It reduces CCD/NIR FITS images through WCS solving, cosmic-ray removal, and background subtraction; builds or uses reference catalogs (e.g. Gaia, Pan-STARRS); measures **aperture** and **PSF** (ePSF) photometry; and calibrates zeropoints with robust fitting and optional colour terms. Optional **template subtraction** (SFFT by default, or HOTPANTS/ZOGY) produces difference images for transient detection. The pipeline reports target magnitudes, errors, limiting magnitudes (via injection/recovery), and can output light curves and detection-limit plots.

Project links:
- GitHub: https://github.com/Astro-Sean/autophot
- Paper: https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B

## Installation (Conda)

Install from the `astro-sean` channel:

```bash
conda install astro-sean::autophot
```

Recommended in a fresh environment:

```bash
conda create -n autophot python=3.11 -y
conda activate autophot
conda install astro-sean::autophot
```

To update:

```bash
conda update astro-sean::autophot
```

**Maintainers:** publishing the same version again requires bumping the conda **build number** in `conda/recipe/meta.yaml` (or using `anaconda upload --force`). See `conda/README.md` if uploads fail with HTTP 409.

You can sanity‑check the install with:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPHoT import OK')"
autophot-main -h
```

## External tools (optional but common)

### Optional: PyZOGY (for ZOGY subtraction)

With the `autophot` environment active:

```bash
git clone https://github.com/dguevel/PyZOGY
cd PyZOGY
python setup.py install
cd ..
```

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

If `solve-field` is missing, AutoPHoT will warn and skip WCS solving (unless you force it).

### Astromatic tools (SExtractor/SCAMP/SWarp)

These are common external tools for source catalogs, astrometric calibration, and resampling/mosaicking.

- **Install (conda-forge)**:

```bash
conda install -c conda-forge astromatic-source-extractor
conda install -c conda-forge astromatic-scamp
conda install -c conda-forge astromatic-swarp
```

### HOTPANTS for template subtraction

AutoPHoT can call HOTPANTS for image subtraction when `template_subtraction.method` is set to `hotpants` (default is `sfft`).

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

When using the HOTPANTS backend, AutoPHoT runs the `hotpants` command from your `PATH` and prints a warning if it cannot be found.

## Recommended usage (Python driver script)

For most real datasets you’ll want a small “driver” script that:

- Loads `databases/default_input.yml` defaults
- Overrides the few fields that are specific to your target/dataset (paths, target coords, catalog, template subtraction, etc.)
- Runs the pipeline (optionally with image-level parallelism via `nCPU`)
- Optionally generates lightcurves/tables after the run

Below is a copy-pasteable template you can adapt for your target, paths, and catalog settings.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run AutoPHoT with a small driver script.
Override defaults and run the pipeline; optionally plot lightcurve and tables.
"""

import argparse

from autophot import AutomatedPhotometry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AutoPHoT with optional image-level parallelism via nCPU."
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of images to process in parallel (nCPU>1 enables multiprocessing inside AutoPHoT).",
    )
    args = parser.parse_args()

    # Load defaults from databases/default_input.yml.
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

    # Catalog (required): choose a reference catalog for calibration.
    autophot_input["catalog"]["use_catalog"] = "gaia"  # or "pan_starrs", "sdss", "legacy", "apass", "2mass", "refcat", "custom"

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

    # WCS: re-solve if needed.
    autophot_input["wcs"]["redo_wcs"] = True

    # TNS credentials (recommended: env vars)
    #   export TNS_BOT_ID="..."
    #   export TNS_BOT_NAME="..."
    #   export TNS_BOT_API="..."

    # Template subtraction (optional).
    autophot_input["template_subtraction"]["do_subtraction"] = True

    # template_subtraction.alignment_method: how to align template -> science grid.
    # `reproject` uses WCS; other options include `swarp` and `astroalign`.
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"

    # Subtraction backend.
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

### Full-featured driver script (sanitized)

If you prefer a single, explicit driver script with many runtime toggles (target,
catalog mapping, subtraction backend, lightcurve output), use this pattern.

Notes:
- All filesystem paths below are placeholders (`/path/to/...`).
- `autophot_tokens` is optional. Use it only to load credentials from a local module.
- Do not hard-code secrets in the script; prefer environment variables where possible.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run automated photometry with AutoPHoT.
Override defaults and run the pipeline; optionally plot lightcurve and tables.
"""

import argparse
import sys
from pathlib import Path

# Optional: if running this script outside the repository root.
PROJECT_ROOT = Path("/path/to/autophot_object")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from autophot import AutomatedPhotometry

# Optional local token module (credentials are not required unless your setup uses them).
try:
    import autophot_tokens  # local, user-managed module
except ImportError:
    autophot_tokens = None


def main() -> int:
    # ---------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Run AutoPHoT with optional image-level parallelism via nCPU."
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of images to process in parallel (nCPU>1 enables multiprocessing inside AutoPHoT).",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load defaults and set core paths
    # ---------------------------------------------------------------------
    autophot_input = AutomatedPhotometry.load()
    autophot_input["outdir_name"] = "REDUCED"
    autophot_input["wdir"] = "/path/to/autophot_db"
    autophot_input["fits_dir"] = "/path/to/images/"

    # Optional: set False to skip already-processed files.
    # autophot_input["restart"] = False

    # ---------------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------------
    autophot_input["target_name"] = "SNXXXXabc"  # optional
    autophot_input["target_ra"] = 123.456789
    autophot_input["target_dec"] = -12.345678

    # ---------------------------------------------------------------------
    # Catalog
    # ---------------------------------------------------------------------
    # You can use one catalog globally:
    # autophot_input["catalog"]["use_catalog"] = "gaia"
    #
    # Or a per-filter mapping:
    autophot_input["catalog"]["use_catalog"] = {
        "griz": "refcat",
        "u": "gaia",
        "UBVRI": "apass",
        # "default": "gaia",
    }

    # Custom catalog examples:
    # autophot_input["catalog"]["use_catalog"] = "custom"
    # autophot_input["catalog"]["catalog_custom_fpath"] = "/path/to/my_catalog.csv"

    # Optional credentials for Refcat/CasJobs if needed by your setup.
    if autophot_tokens is not None:
        autophot_input["catalog"]["MASTcasjobs_wsid"] = getattr(
            autophot_tokens, "MASTcasjobs_wsid", None
        )
        autophot_input["catalog"]["MASTcasjobs_pwd"] = getattr(
            autophot_tokens, "MASTcasjobs_pwd", None
        )

    # ---------------------------------------------------------------------
    # Preprocessing / photometry / WCS
    # ---------------------------------------------------------------------
    autophot_input["cosmic_rays"]["remove_cmrays"] = False
    autophot_input["preprocessing"]["trim_image"] = 5  # arcmin

    autophot_input["photometry"]["psf_oversample"] = 2
    autophot_input["photometry"]["perform_emcee_fitting_s2n"] = 10
    autophot_input["photometry"]["redo_sources"] = False

    autophot_input["wcs"]["redo_wcs"] = True
    autophot_input["wcs"]["solve_field_exe_loc"] = "solve-field"

    # ---------------------------------------------------------------------
    # Template subtraction
    # ---------------------------------------------------------------------
    autophot_input["template_subtraction"]["do_subtraction"] = True
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"
    autophot_input["template_subtraction"]["method"] = "sfft"

    # ---------------------------------------------------------------------
    # Parallelism (image-level)
    # ---------------------------------------------------------------------
    autophot_input["nCPU"] = max(1, int(args.ncpu))

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    loc = AutomatedPhotometry.run_photometry(default_input=autophot_input)

    # ---------------------------------------------------------------------
    # Post-run: lightcurve and tables (optional)
    # ---------------------------------------------------------------------
    from lightcurve import (
        plot_lightcurve,
        check_detection_plots,
        generate_photometry_table,
    )

    detections_loc = plot_lightcurve(
        loc,
        snr_limit=3,
        method="PSF",
        format="png",
        offset=1,
        show=True,
        plot_color=False,
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

## Template subtraction setup

AutoPHoT expects templates to live under a `templates/` directory inside your `fits_dir`. At runtime it chooses the template based on the **science image filter** and the folder naming conventions below.

### Directory structure and naming rules

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

### Template file rules

- The template can be **any `*.fits` file** in the appropriate `{filter}_template/` folder (the filename does **not** need to contain `_template`).
- Files containing `PSF_model` or ending in `.weight` are ignored as templates.
- Keep exactly **one** usable template per filter folder to avoid ambiguity.

### User-provided vs downloaded templates

- **User-provided template:**
  - Place the template FITS into the correct folder under `templates/` (as above)
  - No special config is required beyond enabling subtraction (AutoPHoT will discover templates in `fits_dir/templates/...` automatically):

```python
autophot_input["template_subtraction"]["do_subtraction"] = True
```

- **Downloaded template:**
  - AutoPHoT can download templates into `fits_dir/templates/` before running subtraction.
  - Select the download source via `template_subtraction.download_templates`:

```python
autophot_input["template_subtraction"]["do_subtraction"] = True
autophot_input["template_subtraction"]["download_templates"] = "panstarrs"  # or "legacy", "sdss"
autophot_input["template_subtraction"]["templates_size"] = 10               # arcmin cutout size
```

### Alignment method (science-template registration)

Set alignment with:

```python
autophot_input["template_subtraction"]["alignment_method"] = "reproject"  # or "swarp", "astroalign"
```

- **`reproject`:** WCS-based registration (recommended when WCS is good; robust and deterministic)
- **`swarp`:** external SWarp-based registration (useful in some survey-like workflows)
- **`astroalign`:** feature-matching registration (can help when WCS is poor)

### Subtraction backend

Set subtraction backend with:

```python
autophot_input["template_subtraction"]["method"] = "sfft"  # default; or "hotpants", "zogy"
```

- **`hotpants`:** requires the external HOTPANTS executable (see install section above)
- **`sfft`:** uses the Python SFFT backend
- **`zogy`:** requires PyZOGY to be installed (optional dependency).

If you use the SFFT backend, please cite the SFFT method and see upstream documentation/source:

- Repo: `https://github.com/thomasvrussell/sfft`
- ADS (Hu et al. 2022, ApJ 936, 157): `https://ui.adsabs.harvard.edu/abs/2022ApJ...936..157H/abstract`

When template folders are present/used, AutoPHoT will automatically apply the required preparation steps for subtraction (e.g. WCS checks and preprocessing required by the chosen backend).

## Citing AutoPHoT

If you use AutoPHoT in a publication, please cite:

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

