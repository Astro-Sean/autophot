![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Relative Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT: Automated Photometry Of Transients

**AutoPhOT** is a comprehensive photometric pipeline built on [Photutils](https://photutils.readthedocs.io/) and [Astropy](https://www.astropy.org/). It provides automated aperture and PSF photometry for transients and variable sources, including catalogue calibration, WCS solving, and optional template subtraction.

## Quick Links
- **Conda Package**: [https://anaconda.org/astro-sean/autophot](https://anaconda.org/astro-sean/autophot)
- **Paper**: [A&A 667, A62 (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)
- **Issues**: [GitHub Issues](https://github.com/Astro-Sean/autophot/issues)

> [!NOTE]
> I am the sole developer and maintainer of AutoPhOT and also a full-time researcher at MPE. Please open issues on GitHub and I will do my best to resolve them.

---



## Installation

### Conda (Recommended)

**Important**: AutoPHOT requires the `conda-forge` channel for dependency resolution.

```bash
# Method 1: Install with conda-forge (recommended)
conda install -c conda-forge -c astro-juanlu autophot

# Method 2: Add conda-forge permanently
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c astro-juanlu autophot

# Method 3: Create dedicated environment
conda create -n autophot -c conda-forge python=3.11
conda activate autophot
conda install -c astro-juanlu autophot
```

Verify installation:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

---

## Quick Start

```python
from autophot import AutomatedPhotometry

# Load default configuration
config = AutomatedPhotometry.load()

# Set basic parameters
config["fits_dir"] = "/path/to/your/images"
config["target_name"] = "SN2024A"
config["target_ra"] = 123.456789
config["target_dec"] = -12.345678

# Run photometry
output_file = AutomatedPhotometry.run_photometry(default_input=config)
print(f"Results saved to: {output_file}")
```

---

## Optional Dependencies

### Astrometry.net (`solve-field`)

For WCS solving when FITS headers lack astrometry:

```bash
conda install -c conda-forge astrometry-net
```

### Astromatic Suite (SExtractor, SCAMP, SWarp)

```bash
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp
```

### HOTPANTS

For template subtraction with the `hotpants` method:

```bash
conda install -c conda-forge cfitsio make gcc
git clone https://github.com/acbecker/hotpants
cd hotpants && make
```

---

## Detection Logic

AutoPhOT classifies sources as detections or upper limits based on **Signal-to-Noise Ratio (SNR)**.

### Detection Criteria

A source is classified as a **detection** when:
- `SNR >= snr_limit` (default: 3)
- Magnitude and error are finite

A source is an **upper limit** (non-detection) when:
- `SNR < snr_limit`
- Magnitude is fainter than limiting magnitude (`mag > lmag`)

### Key Points

- **SNR is the primary detection metric** - beta values from injection experiments are not used for target detection
- Low S/N sources visible in subtraction images will be correctly classified as detections if `SNR >= 3`
- The `snr_limit` parameter controls detection sensitivity (default: 3.0)

### Setting Detection Threshold

```python
# Stricter detection (higher confidence)
config["photometry"]["detection_limit"] = 5  # SNR >= 5

# More permissive (for faint transients)
config["photometry"]["detection_limit"] = 3  # SNR >= 3
```

---

## Output Products

### Per-Image Outputs

For each processed image, AutoPhOT creates:
- `OUTPUT_<image>.csv` - Photometry results (magnitudes, errors, SNR)
- `CALIB_<image>.csv` - Calibration diagnostics
- `PSFSources_*.csv` - PSF model stars used
- `targetPSF_*.png` or `PSF_Target_*.png` - Diagnostic plots

### Columns in Output CSV

| Column | Description |
|--------|-------------|
| `*_PSF` / `*_AP` | Calibrated magnitude (PSF or aperture) |
| `*_PSF_err` / `*_AP_err` | Magnitude uncertainty |
| `SNR` / `snr_psf` / `snr_ap` | Signal-to-noise ratio |
| `zp_*` | Zero point magnitude |
| `lmag` | Limiting magnitude |
| `fwhm` | FWHM in arcsec |
| `background` | Background level (ADU) |
| `background_rms` | Background noise (ADU) |

### Post-Processing Products

```python
from lightcurve import plot_lightcurve, generate_photometry_table

# Lightcurve plot with detections and limits
plot_lightcurve(output_file, snr_limit=3, method="PSF")

# ASCII photometry table
# Output: lightcurve_PSF.dat with columns:
# MJD, Date, Mag, Error, Filter, Limit
generate_photometry_table(output_file, snr_limit=3, method="PSF")
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Images ignored | Check FITS headers have `TELESCOP`, `INSTRUME`, and `FILTER` |
| No catalogs found | Set `catalog.use_catalog` for your filters |
| Template subtraction fails | Ensure templates are in `templates/<filter>_template/` |
| Poor photometry | Check `fwhm` and `background_rms` values in outputs |

### Environment Variables

For TNS lookups and catalog access, set these (do not hard-code):

```bash
export MASTCASJOBS_WSID="..."
export MASTCASJOBS_PWD="..."
export TNS_BOT_ID="..."
export TNS_BOT_NAME="..."
export TNS_BOT_API="..."
```

---

## Photometry Methods

### PSF Photometry (Recommended)

- Best for point sources (supernovae, variable stars)
- Uses empirical PSF from field stars
- Configurable via `perform_emcee_fitting_s2n`

### Aperture Photometry

- Best for extended sources
- Configurable aperture sizes based on FWHM

---

## Example Usage

### Complete Driver Script

```python
#!/usr/bin/env python3
"""
Example AutoPhOT driver script.
"""
import os
from autophot import AutomatedPhotometry, prepare_template_directory

# Load default configuration
config = AutomatedPhotometry.load()
config["nCPU"] = 4  # Parallel processing

# Paths
config["outdir_name"] = "REDUCED"
config["wdir"] = "/path/to/working/directory"
config["fits_dir"] = "/path/to/images"

# Target coordinates
config["target_name"] = "SN2024A"
config["target_ra"] = 123.456789
config["target_dec"] = -12.345678

# Photometric catalogs by filter
config["catalog"]["use_catalog"] = {
    "griz": "refcat",
    "u": "gaia", 
    "UBVRI": "apass",
}

# Processing options
config["cosmic_rays"]["remove_cmrays"] = False
config["wcs"]["redo_wcs"] = True

# Enable template subtraction
config["template_subtraction"]["do_subtraction"] = True
config["template_subtraction"]["method"] = "sfft"

# Create template directories
prepare_template_directory(
    fits_dir=config["fits_dir"],
    include_legacy_p_folders=False,
    confirm_before_continue=True,
)

# Run photometry
output = AutomatedPhotometry.run_photometry(default_input=config, do_photometry=True)

# Generate plots and tables
from lightcurve import plot_lightcurve, generate_photometry_table
plot_lightcurve(output, snr_limit=3, method="PSF")
generate_photometry_table(output, snr_limit=3, method="PSF")
```

### Listing All Parameters

```python
from autophot import list_parameters
list_parameters()
```

> [!IMPORTANT]
> FITS images **must** have `TELESCOP` and `INSTRUME` header keywords, plus a bandpass keyword (e.g., `FILTER`). Images without these will be ignored.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run automated photometry with AutoPhOT.
Override default_input and run pipeline; optionally plot lightcurve and tables.
"""

import os
from autophot import AutomatedPhotometry, prepare_template_directory


def main() -> int:
    autophot_input = AutomatedPhotometry.load()
    autophot_input["nCPU"] = 1

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    autophot_input["outdir_name"] = "REDUCED"
    autophot_input["wdir"] = "/path/to/autophot_db"
    autophot_input["fits_dir"] = "/path/to/images"

    # Optional: do not re-run files already processed.
    # autophot_input["restart"] = False

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    autophot_input["target_name"] = "SNXXXXabc"
    autophot_input["target_ra"] = 123.456789
    autophot_input["target_dec"] = -12.345678

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    autophot_input["catalog"]["use_catalog"] = {
        "griz": "refcat",
        "u": "gaia",
        "UBVRI": "apass",
        # "default": "gaia",
    }

    # Optional: Gaia + custom throughput curves ("gaia_custom")
    # autophot_input["catalog"]["use_catalog"] = {"gri": "gaia_custom", ...}
    # autophot_input["catalog"]["transmission_curve_map"] = {"g": "path/to/g.dat", ...}

    # Optional credentials from environment (do not hard-code secrets):
    if os.getenv("MASTCASJOBS_WSID"):
        autophot_input["catalog"]["MASTcasjobs_wsid"] = os.getenv("MASTCASJOBS_WSID")
    if os.getenv("MASTCASJOBS_PWD"):
        autophot_input["catalog"]["MASTcasjobs_pwd"] = os.getenv("MASTCASJOBS_PWD")
    if os.getenv("TNS_BOT_ID"):
        autophot_input["wcs"]["TNS_BOT_ID"] = os.getenv("TNS_BOT_ID")
    if os.getenv("TNS_BOT_NAME"):
        autophot_input["wcs"]["TNS_BOT_NAME"] = os.getenv("TNS_BOT_NAME")
    if os.getenv("TNS_BOT_API"):
        autophot_input["wcs"]["TNS_BOT_API"] = os.getenv("TNS_BOT_API")

    # ------------------------------------------------------------------
    # Preprocessing / photometry / WCS
    # ------------------------------------------------------------------
    autophot_input["cosmic_rays"]["remove_cmrays"] = False
    autophot_input["preprocessing"]["trim_image"] = 5
    autophot_input["photometry"]["perform_emcee_fitting_s2n"] = 10
    autophot_input["wcs"]["redo_wcs"] = True

    # ------------------------------------------------------------------
    # Template subtraction
    # ------------------------------------------------------------------
    autophot_input["template_subtraction"]["do_subtraction"] = True
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"
    autophot_input["template_subtraction"]["method"] = "sfft"
    autophot_input["template_subtraction"]["kernel_order"] = 1

    # Create template folder structure and ask before continuing.
    prepare_template_directory(
        fits_dir=autophot_input["fits_dir"],
        include_legacy_p_folders=False,
        confirm_before_continue=True,
    )

    loc = AutomatedPhotometry.run_photometry(
        default_input=autophot_input,
        do_photometry=True,
    )

    # Optional post-run products.
    from lightcurve import plot_lightcurve, check_detection_plots, generate_photometry_table

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
    generate_photometry_table(loc, snr_limit=3, method="PSF", reference_epoch=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Preparing Template-Subtracted Photometry

1. Set subtraction options:
   - `autophot_input["template_subtraction"]["do_subtraction"] = True`
   - `autophot_input["template_subtraction"]["alignment_method"] = "reproject"`
   - `autophot_input["template_subtraction"]["method"] = "sfft"` (or `hotpants`, `zogy`)
2. Create template directories:
   - Call `prepare_template_directory(...)`.
  - AutoPhOT prints where folders were created and asks if you want to continue.
3. Place template FITS files:
   - Put one usable template per filter in `fits_dir/templates/<filter>_template/`.
4. Run photometry.

Notes:
- Modern folder names are preferred: `r_template`, `g_template`, etc.
- Legacy `rp_template` naming is still supported by the pipeline if needed.
- Keep paths and credentials sanitized in public scripts.

---

## Citation

If you use AutoPhOT in your research, please cite:

> Brennan, S. J., & Fraser, M. 2022, A&A, 667, A62

```bibtex
@ARTICLE{2022A&A...667A..62B,
       author = {{Brennan}, S.~J. and {Fraser}, M.},
       title = "{The AUTOmated Photometry Of Transients pipeline (AutoPhOT)}",
      journal = {\aap},
         year = 2022,
        month = nov,
       volume = {667},
          eid = {A62},
        pages = {A62},
          doi = {10.1051/0004-6361/202243067},
archivePrefix = {arXiv},
       eprint = {2201.02635},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
