![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT: Automated Photometry Of Transients

AutoPhOT is a photometric pipeline for transient and variable source follow-up. Built on [Photutils](https://photutils.readthedocs.io/) and [Astropy](https://www.astropy.org/), it provides automated aperture and PSF photometry, catalogue calibration, WCS solving, and optional template subtraction.

Unlike general-purpose reduction pipelines that stack data from a single instrument, AutoPhOT is **target-centric**: it ingests FITS frames from any telescopes, filters, and pixel scales, locates the target in each, and produces a single calibrated light curve.

## Quick Links

- **Conda Package**: [anaconda.org/astro-sean/autophot](https://anaconda.org/astro-sean/autophot)
- **Paper**: [A&A 667, A62 (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)
- **Issues**: [GitHub Issues](https://github.com/Astro-Sean/autophot/issues)

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

## Pipeline Stages

1. **Ingestion & WCS** — Sort frames by telescope/instrument/filter. Verify or re-solve WCS using `astrometry.net` with Gaia DR3 cross-matching. Handles SIP/TPV distortion.
2. **Image preparation** — Cosmic-ray rejection, satellite-streak detection, background estimation, and FWHM measurement per frame. Adaptive thresholds for sparse and crowded fields.
3. **Photometry** — Aperture and PSF photometry at the target position. PSF models built from in-frame stars via `photutils` ePSFBuilder. Three fitters: least-squares (default), Poisson likelihood, and MCMC (`emcee`).
4. **Calibration** — Zero points from user-selectable catalogs (Gaia DR3, Pan-STARRS, SDSS, APASS, 2MASS, Legacy Survey, SkyMapper, and more). Per-filter catalog routing and Gaia XP synthetic photometry.
5. **Template subtraction** (optional) — Align science and template images, then subtract using SFFT, HOTPANTS, or ZOGY with automatic fallback.
6. **Limiting magnitudes** — Source injection and recovery at multiple S/N thresholds (default 3σ and 5σ) for robust upper limits.

### Key features

- **Multi-instrument** — Combine data from many facilities into one light curve.
- **Cascaded alignment** — Six methods (spalipy, SWarp, reproject, AstroAlign, tweakwcs, chi2_shift) with automatic fallback and FWHM-scaled quality gates.
- **Inverted-fit detection** — Detects fading sources as negative PSF dips in difference images.
- **Poisson likelihood fitting** — Statistically superior to χ² for low-count photometry (Fermilab TM-2543-AE).
- **Adaptive MCMC** — emcee with adaptive chain length, principled burn-in, and full covariance extraction.
- **Gaia XP synthetic photometry** — Calibrate directly against Gaia DR3 XP spectra with custom transmission curves.
- **TNS integration** — Automatic coordinate and redshift lookup from the Transient Name Server.
- **Per-filter catalog routing** — Assign different catalogs to different filter groups in a single run.

---

## Installation

### Conda (Recommended)

AutoPhOT requires the `conda-forge` channel and has 20+ conda dependencies. Install the fast libmamba solver first to avoid slow dependency resolution:

```bash
# One-time: install the fast solver
conda install -n base -c conda-forge conda-libmamba-solver
conda config --set solver libmamba
```

```bash
# Method 1: Install into an existing env
conda install -c conda-forge -c astro-sean autophot

# Method 2: Create a dedicated environment
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

Verify:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

### From source (developer)

**Option A: Editable pip install**

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
pip install -e .
pip install sfft==1.7.3 sip_tpv==1.1  # not on conda
```

**Option B: Full conda environment (reproducible)**

The `environment.yml` pins all dependency versions:

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

The repository includes 186 tests covering core functions, data validation, PSF validation, uncertainty calibration, MCMC diagnostics, injection/recovery, quality flags, and regression. Tests use synthetic data (no external data required).

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

List all configurable parameters:

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

For WCS solving when FITS headers lack astrometry:

```bash
conda install conda-forge::astrometry
# or: sudo apt install astrometry.net
```

Download index files from the [astrometry.net website](https://astrometry.net/data.html).

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

AutoPhOT aligns the template to the science image before subtraction. Six methods are available. The default is `spalipy` (best sub-pixel accuracy). Setting `alignment_method` to a specific method uses only that method, with fallback to the cascade on failure.

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

Configurable alignment quality gates (reject and fall back to next method if exceeded):

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

AutoPhOT builds an empirical ePSF model from in-frame stars using `photutils` ePSFBuilder, with adaptive oversampling for undersampled images (FWHM < 2.5 px).

### Fitters

| Fitter | Config key | Use case |
|--------|-----------|----------|
| **Least-squares** (default) | — | Fast, general-purpose |
| **Poisson likelihood** | `use_poisson_likelihood_fitter: True` | Low-count regime; superior to χ² (Fermilab TM-2543-AE) |
| **MCMC (emcee)** | `perform_emcee_fitting_s2n: 10` | Bayesian uncertainties; triggered when target S/N < threshold |

### MCMC configuration

The emcee fitter runs adaptively — it extends the chain until autocorrelation time stabilises (up to 50,000 steps), then applies principled burn-in (10×τ discard) and thinning.

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

When `emcee_store_samples` is enabled, a corner plot is saved as `PSF_MCMC_corner_*.png`.

### Inverted-fit detection

For difference images where the transient appears as a negative residual (fading source), AutoPhOT can fit the target on a sign-flipped image:

```yaml
photometry:
  check_inverted_image: True
```

Results are flagged with an `_inverted_fit` column.

### PSF star selection

PSF stars are selected via SExtractor detection with quality cuts on saturation, elongation, isolation, FWHM fraction, and CLASS_STAR. FFT-based rejection removes stars with close companions.

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
| **SFFT** | `sfft` | `pip install sfft==1.7.3` | Default; noise decorrelation, B-spline kernel, variable-star rejection |
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

Saturated star cores in the template can be inpainted before subtraction to prevent artifacts:

```yaml
template_subtraction:
  inpaint_template_cores: False
  inpaint_method: biharmonic       # biharmonic or telea
  inpaint_saturate_frac: 0.90
  inpaint_dilate_radius: 6         # px
```

---

## Limiting Magnitudes

AutoPhOT computes limiting magnitudes via source injection and recovery at multiple S/N thresholds. The default produces `Limit_3p0S2N` and `Limit_5p0S2N` columns.

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

Output notes:
- The default CSV (`lightcurve_output.csv`) is **long-form**: one row per image with a `filter` column.
- Multi-S/N limit columns (e.g., `Limit_3p0S2N`, `Limit_5p0S2N`) are generated automatically.
- Inverted-fit results are flagged with an `_inverted_fit` boolean column.
- Lightcurve x-axes are MJD by default; for data spanning < 1 day the axis
  automatically switches to minutes or hours since the first observation
  (e.g. `Time since 9th August 9:00pm UTC [hr]`).
- `plot_variability_check` writes `VariabilityCheck_<method>.png` next to the
  photometry CSV. It reads the per-image `Calib_*.csv` catalogs, subtracts
  each epoch's reference-ensemble mean instrumental magnitude (common-mode
  instrumental/atmospheric drift) from the target and reference stars, and
  plots the residuals — flat target residuals within the reference scatter
  mean the apparent variability was instrumental.

---

## Environment Variables

For TNS lookups and RefCAT2 access (do not hard-code in scripts):

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

1. Set `do_subtraction = True`, choose `method` (`sfft`, `hotpants`, `zogy`) and `alignment_method` (`spalipy` default).
2. Call `prepare_template_directory(...)` to create the folder structure.
3. Place one template FITS per filter in `fits_dir/templates/<filter>_template/`.
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
