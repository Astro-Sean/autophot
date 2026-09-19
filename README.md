![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT: Automated Photometry Of Transients

AutoPhOT is a photometry pipeline for following up transients and variable sources. It is built on [Photutils](https://photutils.readthedocs.io/) and [Astropy](https://www.astropy.org/), and provides aperture and PSF photometry, catalogue calibration, WCS solving, and optional template subtraction.

Most reduction pipelines stack data from a single instrument. AutoPhOT does the opposite: it takes FITS frames from any telescope, filter, and pixel scale, finds the target in each frame, and produces a single calibrated light curve.

- Conda package: [anaconda.org/astro-sean/autophot](https://anaconda.org/astro-sean/autophot)
- Paper: [A&A 667, A62 (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)
- Issues: [GitHub Issues](https://github.com/Astro-Sean/autophot/issues)

> [!NOTE]
> I am the sole developer and maintainer of AutoPhOT and also a [full-time researcher](https://astro-sean.github.io/index.html) at [MPE](https://www.mpe.mpg.de/person/144270/1302618).
> Please open issues on GitHub and I will do my best to resolve them as soon as possible.

---

## Table of Contents

- [Installation](#installation)
- [Testing](#testing)
- [Quick Start](#quick-start)
- [CLI Entry Points](#cli-entry-points)
- [Optional Dependencies](#optional-dependencies)
- [Alignment Methods](#alignment-methods)
- [PSF Photometry](#psf-photometry)
- [Template Subtraction](#template-subtraction)
- [Limiting Magnitudes](#limiting-magnitudes)
- [Supported Catalogs](#supported-catalogs)
- [Post-Processing](#post-processing)
- [Environment Variables](#environment-variables)
- [Example Usage](#example-usage)
- [Citation](#citation)

---

## What the pipeline does

1. Sorts frames by telescope, instrument, and filter, then checks or re-solves the WCS using `astrometry.net` with Gaia DR3 cross-matching. SIP and TPV distortion are handled.
2. Prepares each image: cosmic-ray rejection, satellite-streak detection, background estimation, and FWHM measurement. Thresholds adapt for sparse and crowded fields.
3. Runs aperture and PSF photometry at the target position. The PSF model is built from in-frame stars using `photutils` ePSFBuilder. Three fitters are available: least-squares (default), Poisson likelihood, and MCMC (`emcee`).
4. Calibrates against a catalogue (Gaia DR3, Pan-STARRS, SDSS, APASS, 2MASS, Legacy Survey, SkyMapper, and others). Different catalogues can be assigned to different filter groups in a single run, and Gaia XP synthetic photometry is supported.
5. Optionally subtracts a template using SFFT, HOTPANTS, or ZOGY, with automatic fallback between methods. On difference images a fading source can be recovered as a negative PSF dip.
6. Measures limiting magnitudes by injecting and recovering artificial sources.

---

## Installation

### Conda (recommended)

AutoPhOT needs the `conda-forge` channel and has 20+ conda dependencies. Install the fast libmamba solver first to avoid slow dependency resolution:

```bash
# One-time: install the fast solver
conda install -n base -c conda-forge conda-libmamba-solver
conda config --set solver libmamba
```

```bash
# Install into an existing env, or create a dedicated one
conda install -c conda-forge -c astro-sean autophot

conda create -n autophot -c conda-forge -c astro-sean python=3.11 autophot
conda activate autophot
```

> [!NOTE]
> `sfft` and `sip_tpv` are not on conda channels. After installing AutoPhOT, also run:
> ```bash
> pip install sfft==1.7.3 sip_tpv==1.1
> ```

Check the install worked:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

### From source (developer)

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
pip install -e .
pip install sfft==1.7.3 sip_tpv==1.1  # not on conda
```

For a reproducible environment, `environment.yml` pins all dependency versions (including the pip-only ones):

```bash
conda env create -f environment.yml
conda activate autophot
pip install -e .
```

---

## Testing

The test suite covers the core functions, data validation, PSF validation, uncertainty calibration, MCMC diagnostics, injection/recovery, quality flags, and regressions. Everything runs on synthetic data, so no external images are needed.

```bash
pip install -e ".[test]"    # install test dependencies
pytest                      # run all tests
pytest -m "not slow and not mcmc and not injection"  # fast tests only
```

---

## Quick Start

```python
from autophot import AutomatedPhotometry

config = AutomatedPhotometry.load()
config["fits_dir"] = "/path/to/your/images"
config["target_name"] = "SN2024A"
config["target_ra"] = 123.456789
config["target_dec"] = -12.345678

output_file = AutomatedPhotometry.run_photometry(default_input=config)
print(f"Results saved to: {output_file}")
```

To list every configurable parameter:

```python
from autophot import list_parameters
list_parameters()
```

---

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `autophot-main` | Run the full photometry pipeline |
| `autophot-driver` | Interactive driver script with template setup |
| `autophot-gaia-curves` | Build a Gaia custom catalog from transmission curves |
| `autophot-inspect-telescope` | Inspect and verify telescope header keywords |

---

## Optional Dependencies

### Astrometry.net (`solve-field`)

Needed for WCS solving when the FITS headers have no astrometry:

```bash
conda install conda-forge::astrometry
# or: sudo apt install astrometry.net
```

Index files can be downloaded from the [astrometry.net website](https://astrometry.net/data.html).

### Astromatic Suite (SExtractor, SCAMP, SWarp)

```bash
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp
```

### SFFT (template subtraction)

```bash
pip install sfft==1.7.3
```

### HOTPANTS (template subtraction)

```bash
conda install -c conda-forge cfitsio make gcc
git clone https://github.com/acbecker/hotpants
cd hotpants && make
```

---

## Alignment Methods

Before subtraction the template has to be aligned to the science image. Six methods are available; `spalipy` is the default and usually gives the best sub-pixel accuracy. If you set `alignment_method` to a specific method it is tried first, and the pipeline falls back to the rest of the cascade if it fails.

| Method | `alignment_method` | Install | Typical RMS |
|--------|---------------------|---------|-------------|
| **spalipy** (default) | `spalipy` | `pip install spalipy>=3.5` | 0.05-0.2 px |
| **SWarp** (SCAMP+SWarp) | `swarp` | Astromatic suite | 0.1-0.5 px |
| **WCS Reproject** | `reproject` | bundled | 0.1-0.3 px |
| **AstroAlign** | `astroalign` | bundled | 0.2-1.0 px |
| **tweakwcs** | `tweakwcs` | `pip install tweakwcs>=0.8` | 0.1-0.5 px |
| **chi2_shift** | `chi2_shift` | `pip install image-registration>=0.2` | 0.5-2.0 px |

Each method is checked against offset, RMS, and p95 alignment-quality gates scaled to the image FWHM; a method that fails is rejected and the next one is tried. The gate thresholds are under `template_subtraction` in the config (`alignment_max_offset_px`, `alignment_max_rms_px`, `alignment_max_p95_px`).

Install all optional alignment methods at once:

```bash
pip install -e ".[spalipy,tweakwcs,chi2-shift]"
```

---

## PSF Photometry

AutoPhOT builds an empirical ePSF model from in-frame stars using `photutils` ePSFBuilder. For undersampled images (FWHM < 2.5 px) the oversampling factor is increased automatically. PSF stars are selected from a SExtractor detection run with cuts on saturation, elongation, isolation, FWHM consistency, and CLASS_STAR, plus an FFT-based check for close companions.

### Fitters

| Fitter | Config key | Use case |
|--------|-----------|----------|
| **Least-squares** (default) | - | Fast, general-purpose |
| **Poisson likelihood** | `use_poisson_likelihood_fitter: True` | Low-count regime; behaves better than chi2 (Fermilab TM-2543-AE) |
| **MCMC (emcee)** | `perform_emcee_fitting_s2n: 10` | Bayesian uncertainties; runs when the target S/N drops below the threshold |

The emcee fitter is adaptive: the chain is extended until the autocorrelation time stabilises, burn-in is discarded, and the chain is thinned. Chain length, walker count, and thinning are configurable under `photometry` (`emcee_nwalkers`, `emcee_nsteps`, `emcee_thin`, and related keys). With `emcee_store_samples` enabled, a corner plot is saved as `PSF_Corner_*.{png,svg}`.

### Inverted-fit detection

On difference images a fading transient shows up as a negative residual. Setting `photometry.check_inverted_image: True` fits the target on a sign-flipped copy of the image; these results are flagged with an `_inverted_fit` column.

---

## Template Subtraction

### Subtraction backends

| Method | `method` value | Install | Notes |
|--------|---------------|---------|-------|
| **SFFT** | `sfft` | `pip install sfft==1.7.3` | Default; supports noise decorrelation, B-spline kernel, variable-star rejection |
| **HOTPANTS** | `hotpants` | Build from source | Classic kernel-matching algorithm |
| **ZOGY** | `zogy` | Auto-downloaded from [pmvreeswijk/ZOGY](https://github.com/pmvreeswijk/ZOGY) | PSF-matched subtraction; propagates noise correctly |

Key SFFT options under `template_subtraction`:

- `kernel_order`: polynomial degree of the spatially varying kernel, or `"auto"`
- `forceconv`: `REF` (default, convolve the reference to the science PSF), `SCI`, or `AUTO`
- `sfft_decorrelate_noise`, `sfft_use_bspline_kernel`, `sfft_bg_order`: optional SFFT tuning

Saturated star cores in the template can be inpainted before subtraction (`inpaint_template_cores`) so they do not leave artifacts in the difference image.

---

## Limiting Magnitudes

Limiting magnitudes are measured by injecting artificial sources into the image and checking which ones are recovered, at one or more S/N thresholds. The defaults produce `Limit_3p0S2N` and `Limit_5p0S2N` columns. Thresholds, injection strategy, and site count are configurable under `limiting_magnitude`.

---

## Supported Catalogs

| Catalog | `use_catalog` value | Notes |
|---------|---------------------|-------|
| Gaia DR3 + XP | `gaia` | Default for most filters |
| Pan-STARRS | `pan_starrs` / `ps1` | DR1/DR2 |
| SDSS | `sdss` | |
| APASS | `apass` | |
| 2MASS | `2mass` | Infrared |
| Legacy Survey | `legacy` | DR8+ |
| SkyMapper | `skymapper` | Southern sky |
| RefCAT2 | `refcat` | Requires MAST CasJobs credentials |
| TIC | `tic` | TESS Input Catalog |
| Custom CSV | `custom` | Set `catalog.catalog_custom_fpath` |
| Gaia + custom curves | `gaia_custom` | User-provided transmission curves |

Different catalogs can be assigned to different filter groups in one run:

```yaml
catalog:
  use_catalog:
    griz: refcat
    u: gaia
    UBVRI: apass
    default: gaia
```

For non-standard filters, provide transmission curve files and use `gaia_custom`:

```yaml
catalog:
  use_catalog:
    gri: gaia_custom
  transmission_curve_map:
    g: /path/to/g_band.dat
    r: /path/to/r_band.dat
    i: /path/to/i_band.dat
```

---

## Post-Processing

```python
from lightcurve import (
    plot_lightcurve, plot_variability_check,
    generate_photometry_table, check_detection_plots,
)

# Lightcurve plot with detections and limits
plot_lightcurve(output_file, snr_limit=3, method="PSF")

# Variability check: target vs reference-star ensemble (no fitting)
plot_variability_check(output_file, method="PSF")

# ASCII photometry table (MJD, Date, Mag, Error, Filter, Limit)
generate_photometry_table(output_file, snr_limit=3, method="PSF")

# Sort detection plots into organised folders
check_detection_plots(detections_loc, method="PSF")
```

A few notes on the outputs:

- The default CSV (`lightcurve_output.csv`) is long-form: one row per image with a `filter` column.
- Multi-S/N limit columns (e.g. `Limit_3p0S2N`, `Limit_5p0S2N`) are generated automatically.
- Inverted-fit results are flagged with an `_inverted_fit` boolean column.
- Lightcurve x-axes are in MJD by default; for data spanning less than a day the axis switches to minutes or hours since the first observation.
- `plot_variability_check` compares the target against the reference-star ensemble after removing each epoch's common-mode instrumental drift, separating real variability from instrumental or atmospheric trends.

---

## Environment Variables

Needed for TNS lookups and RefCAT2 access. Do not hard-code these in scripts:

```bash
export MASTCASJOBS_WSID="..."
export MASTCASJOBS_PWD="..."
export TNS_BOT_ID="..."
export TNS_BOT_NAME="..."
export TNS_BOT_API="..."
```

---

## Example Usage

> [!IMPORTANT]
> FITS images **must** have `TELESCOP`, `INSTRUME`, and a bandpass keyword (e.g., `FILTER`). Images without these will be ignored.

```python
#!/usr/bin/env python3
"""Example AutoPhOT driver script."""
import os
from autophot import AutomatedPhotometry, prepare_template_directory

config = AutomatedPhotometry.load()
config["nCPU"] = 4
config["outdir_name"] = "REDUCED"
config["wdir"] = "/path/to/working/directory"
config["fits_dir"] = "/path/to/images"

# Target
config["target_name"] = "SN2024A"
config["target_ra"] = 123.456789
config["target_dec"] = -12.345678

# Per-filter catalog routing
config["catalog"]["use_catalog"] = {
    "griz": "refcat",
    "u": "gaia",
    "UBVRI": "apass",
}

# Processing options
config["cosmic_rays"]["remove_cmrays"] = False
config["wcs"]["redo_wcs"] = True
config["photometry"]["perform_emcee_fitting_s2n"] = 10
config["photometry"]["check_inverted_image"] = True

# Template subtraction
config["template_subtraction"]["do_subtraction"] = True
config["template_subtraction"]["method"] = "sfft"
config["template_subtraction"]["alignment_method"] = "spalipy"
config["template_subtraction"]["kernel_order"] = 1

# Optional TNS credentials from environment
for key in ("TNS_BOT_ID", "TNS_BOT_NAME", "TNS_BOT_API"):
    if os.getenv(key):
        config["wcs"][key] = os.getenv(key)

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

### Preparing template-subtracted photometry

1. Set `do_subtraction = True` and choose a `method` (`sfft`, `hotpants`, `zogy`) and `alignment_method` (`spalipy` by default).
2. Call `prepare_template_directory(...)` to create the folder structure.
3. Put one template FITS per filter in `fits_dir/templates/<filter>_template/`.
4. Run photometry.

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
