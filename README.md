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
5. Optionally subtracts a template using SFFT, HOTPANTS, or ZOGY, with automatic fallback between methods.
6. Measures limiting magnitudes by injecting and recovering artificial sources at 3σ and 5σ.

Other things worth knowing:

- Data from many facilities can be combined into one light curve.
- Up to six alignment methods (spalipy, SWarp, reproject, AstroAlign, tweakwcs, chi2_shift) are tried in turn, with quality gates scaled to the image FWHM.
- On difference images, a fading source can be picked up as a negative PSF dip (inverted-fit detection).
- A Poisson-likelihood fitter is available for low-count photometry, where it behaves better than χ² (Fermilab TM-2543-AE).
- The emcee fitter adapts its chain length on the fly and extracts the full parameter covariance.
- Calibration can be done directly against Gaia DR3 XP spectra with custom transmission curves.
- Target coordinates and redshifts can be looked up automatically from the Transient Name Server.

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
# Method 1: install into an existing env
conda install -c conda-forge -c astro-sean autophot

# Method 2: create a dedicated environment
conda create -n autophot -c conda-forge -c astro-sean python=3.11 autophot
conda activate autophot
```

> [!NOTE]
> `sfft` and `sip_tpv` are not on conda channels. After installing AutoPhOT, also run:
> ```bash
> pip install sfft==1.7.3 sip_tpv==1.1
> ```

Alternatively, use `mamba`:

```bash
conda install -c conda-forge mamba
mamba create -n autophot -c conda-forge -c astro-sean python=3.11 autophot
conda activate autophot
```

Check the install worked:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

### From source (developer)

**Option A: editable pip install**

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
pip install -e .
pip install sfft==1.7.3 sip_tpv==1.1  # not on conda
```

**Option B: full conda environment (reproducible)**

`environment.yml` pins all dependency versions:

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
conda env create -f environment.yml
conda activate autophot
pip install -e .
```

> [!NOTE]
> `environment.yml` includes `sfft==1.7.3` in the pip section, so pip-only
> dependencies are installed automatically.

---

## Testing

There are 186 tests covering the core functions, data validation, PSF validation, uncertainty calibration, MCMC diagnostics, injection/recovery, quality flags, and regressions. Everything runs on synthetic data, so no external images are needed.

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

| Method | `alignment_method` | Install | Typical RMS | Speed |
|--------|---------------------|---------|-------------|-------|
| **spalipy** (default) | `spalipy` | `pip install spalipy>=3.5` | 0.05-0.2 px | Medium |
| **SWarp** (SCAMP+SWarp) | `swarp` | Astromatic suite | 0.1-0.5 px | Fast |
| **WCS Reproject** | `reproject` | bundled | 0.1-0.3 px | Fast |
| **AstroAlign** | `astroalign` | bundled | 0.2-1.0 px | Medium |
| **tweakwcs** | `tweakwcs` | `pip install tweakwcs>=0.8` | 0.1-0.5 px | Medium |
| **chi2_shift** | `chi2_shift` | `pip install image-registration>=0.2` | 0.5-2.0 px | Fast |

```yaml
template_subtraction:
  alignment_method: spalipy  # or swarp, reproject, astroalign, tweakwcs, chi2_shift
```

Install all optional alignment methods at once:

```bash
pip install -e ".[spalipy,tweakwcs,chi2-shift]"
```

There are also quality gates on the alignment: if a method exceeds any of these, it is rejected and the next method is tried.

```yaml
template_subtraction:
  alignment_max_offset_px: 0.5
  alignment_max_rms_px: 0.75
  alignment_max_p95_px: 1.5
  alignment_min_sources_for_field_gate: 20
  post_swarp_verify: True
```

---

## PSF Photometry

AutoPhOT builds an empirical ePSF model from in-frame stars using `photutils` ePSFBuilder. For undersampled images (FWHM < 2.5 px) the oversampling factor is increased automatically.

### Fitters

| Fitter | Config key | Use case |
|--------|-----------|----------|
| **Least-squares** (default) | — | Fast, general-purpose |
| **Poisson likelihood** | `use_poisson_likelihood_fitter: True` | Low-count regime; behaves better than χ² (Fermilab TM-2543-AE) |
| **MCMC (emcee)** | `perform_emcee_fitting_s2n: 10` | Bayesian uncertainties; runs when the target S/N drops below the threshold |

### MCMC configuration

The emcee fitter is adaptive: the chain is extended until the autocorrelation time stabilises (up to 50,000 steps), then 10×τ is discarded as burn-in and the chain is thinned.

```yaml
photometry:
  perform_emcee_fitting_s2n: 10   # run MCMC when target S/N < this
  emcee_nwalkers: 32
  emcee_nsteps: null              # null = adaptive
  emcee_burnin_frac: 0.3
  emcee_thin: 10
  emcee_adaptive_tau_target: 50
  emcee_min_autocorr_N: 100
  emcee_store_samples: False      # store chains for corner plots
  emcee_threads: 1
```

With `emcee_store_samples` enabled, a corner plot is saved as `PSF_MCMC_corner_*.png`.

### Inverted-fit detection

On difference images a fading transient shows up as a negative residual. AutoPhOT can fit the target on a sign-flipped copy of the image:

```yaml
photometry:
  check_inverted_image: True
```

These results are flagged with an `_inverted_fit` column.

### PSF star selection

PSF stars come from a SExtractor detection run, followed by quality cuts on saturation, elongation, isolation, FWHM fraction, and CLASS_STAR. An FFT-based check removes stars with close companions.

```yaml
photometry:
  psf_min_candidates: 8
  psf_saturate_fraction: 0.90
  psf_elongation_max: 1.5
  psf_isolation_radius_fwhm: 3.0
  psf_fwhm_min_frac: 0.5
  psf_fwhm_max_frac: 2.5
  psf_class_star_min: 0.4
  psf_fft_rejection: True
  undersampled_fwhm_threshold: 2.5
  psf_auto_oversample_undersampled: True
```

---

## Template Subtraction

### Subtraction backends

| Method | `method` value | Install | Notes |
|--------|---------------|---------|-------|
| **SFFT** | `sfft` | `pip install sfft==1.7.3` | Default; supports noise decorrelation, B-spline kernel, variable-star rejection |
| **HOTPANTS** | `hotpants` | Build from source | Classic kernel-matching algorithm |
| **ZOGY** | `zogy` | Auto-downloaded from [pmvreeswijk/ZOGY](https://github.com/pmvreeswijk/ZOGY) | PSF-matched subtraction; propagates noise correctly |

### SFFT configuration

```yaml
template_subtraction:
  method: sfft
  kernel_order: "auto"             # 0=constant, 1=linear, 2=quadratic, 3=cubic, or "auto"
  kernel_hw_fwhm_multiplier: 2.5   # kernel half-width as FWHM multiplier
  forceconv: REF                   # REF (default, always convolve reference), SCI, or AUTO
  sfft_decorrelate_noise: False    # noise decorrelation (SFFT v1.7.3+)
  sfft_save_decorrelated: False    # save decorrelated diff image separately
  sfft_use_bspline_kernel: False   # B-spline kernel (requires CUDA/Cupy)
  sfft_bg_order: 0                 # background spatial polynomial order
  sfft_crowded_auto: False         # auto-enable crowded-field tuning
  sfft_use_post_anomaly_feedback: True
```

### Template inpainting

Saturated star cores in the template can be inpainted before subtraction to stop them leaving artifacts in the difference image:

```yaml
template_subtraction:
  inpaint_template_cores: False
  inpaint_method: biharmonic       # biharmonic or telea
  inpaint_saturate_frac: 0.90
  inpaint_dilate_radius: 6         # px
```

---

## Limiting Magnitudes

Limiting magnitudes are measured by injecting artificial sources into the image and checking which ones are recovered, at one or more S/N thresholds. The defaults produce `Limit_3p0S2N` and `Limit_5p0S2N` columns.

```yaml
limiting_magnitude:
  snr_thresholds: [3, 5]           # S/N thresholds for limit columns
  recovery_method: auto            # auto=match transient; or PSF, AP, EMCEE
  injection_strategy: ring_quiet   # ring_quiet or annulus_random
  injection_n_sites: 25
  inject_min_radius_fwhm: 2.0
  inject_max_radius_fwhm: 6.0
  plot_injection_recovery: False
```

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

```yaml
catalog:
  use_catalog: gaia
```

### Per-filter catalog routing

```yaml
catalog:
  use_catalog:
    griz: refcat
    u: gaia
    UBVRI: apass
    default: gaia
```

### Gaia custom transmission curves

For non-standard filters, provide transmission curve files:

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
- Lightcurve x-axes are in MJD by default. For data spanning less than a day the axis switches to minutes or hours since the first observation (e.g. `Time since 9th August 9:00pm UTC [hr]`).
- `plot_variability_check` writes `VariabilityCheck_<method>.png` next to the photometry CSV. It reads the per-image `Calib_*.csv` catalogs, subtracts each epoch's reference-ensemble mean instrumental magnitude (common-mode instrumental/atmospheric drift) from the target and the reference stars, and plots the residuals. If the target residuals sit inside the reference scatter, the apparent variability was instrumental rather than real.

---

## Environment Variables

Needed for TNS lookups and RefCAT2 access. Don't hard-code these in scripts:

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

# Optional credentials from environment
if os.getenv("TNS_BOT_ID"):
    config["wcs"]["TNS_BOT_ID"] = os.getenv("TNS_BOT_ID")
if os.getenv("TNS_BOT_NAME"):
    config["wcs"]["TNS_BOT_NAME"] = os.getenv("TNS_BOT_NAME")
if os.getenv("TNS_BOT_API"):
    config["wcs"]["TNS_BOT_API"] = os.getenv("TNS_BOT_API")

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
