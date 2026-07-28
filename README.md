![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Relative Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_relative_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT: Automated Photometry Of Transients

**AutoPhOT** is a comprehensive photometric pipeline built on [Photutils](https://photutils.readthedocs.io/) and [Astropy](https://www.astropy.org/). It provides automated aperture and PSF photometry for transients and variable sources, including catalogue calibration, WCS solving, and optional template subtraction.

## Description

AutoPhOT performs precision photometry of a transient or variable source at a fixed sky position across a heterogeneous set of imaging data. Unlike general-purpose reduction pipelines that build stacked mosaics from a single instrument, AutoPhOT is target-centric: it ingests an arbitrary collection of FITS frames (different telescopes, filters, pixel scales, and WCS solutions), locates the target in each, and produces a self-consistent, calibrated light curve.

### Pipeline stages

1. **Ingestion & WCS** — Frames are sorted by telescope/instrument/filter. WCS is verified or re-solved using `astrometry.net` with Gaia DR3 cross-matching, handling SIP/TPV distortion and independently plate-solved images.
2. **Image preparation** — Cosmic-ray rejection (with satellite-streak detection), background estimation, and FWHM measurement are performed per frame, with adaptive detection thresholds for sparse and crowded fields.
3. **Photometry** — Both aperture and PSF photometry are computed at the target position. PSF models are built empirically from in-frame stars using `photutils` ePSFBuilder, with optional adaptive oversampling for undersampled data. Three fitters are available: least-squares (default), Poisson likelihood (Fermilab TM-2543-AE), and MCMC (`emcee`) with adaptive convergence for principled uncertainty estimation.
4. **Calibration** — Photometric zero points are derived against a user-selectable catalog (Gaia DR3 with XP spectra, Pan-STARRS, SDSS, APASS, 2MASS, Legacy Survey, SkyMapper, and more), with per-filter catalog assignment and synthetic-photometry support for Gaia XP.
5. **Template subtraction** (optional) — When a reference template is available, AutoPhOT aligns science and template images using a cascaded alignment pipeline with six methods, then performs difference imaging using SFFT, HOTPANTS, or PyZOGY with automatic fallback.
6. **Limiting magnitudes** — Source injection and recovery at multiple S/N thresholds (default 3σ and 5σ) using logistic-emcee fitting to produce robust upper limits for non-detections.

### Unique aspects

- **Multi-instrument, target-focused** — Designed from the ground up for follow-up campaigns that combine data from many facilities into one light curve, rather than reducing a single instrument's dataset.

- **Adaptive astrometric alignment** — A cascaded alignment pipeline (SCAMP+SWarp → WCS-based reproject → AstroAlign) with source-matched WCS refinement, per-quadrant verification, and FWHM-scaled quality gates that adapt to sparse fields, large distortion, and pixel-scale mismatches. Optional methods (spalipy spline-warp achieving ~0.05 px RMS, tweakwcs STScI WCS tweaking, chi2_shift cross-correlation) extend coverage to distorted wide-field, HST/JWST, and extended-source-dominated fields. All source detection uses SExtractor for consistency across the pipeline.

- **Inverted-fit detection** — Optionally fits the target on a sign-flipped (×−1) image to detect negative PSF dips — critical for fading sources in template-subtracted images where the transient may appear as a deficit relative to the template. The pipeline automatically replaces poor normal fits with inverted fits when appropriate, and flags results with an `_inverted_fit` column.

- **Poisson likelihood PSF fitting** — Implements the Poisson likelihood fitter from Fermilab TM-2543-AE, which is statistically superior to χ² methods for low-count PSF photometry. Uses analytic gradient and Hessian computation for fast convergence.

- **Adaptive MCMC with principled convergence** — emcee-based Bayesian PSF fitting with adaptive chain length (runs until integrated autocorrelation time stabilises), principled burn-in (10×τ discard), and full covariance matrix extraction from the posterior chain. Produces corner plots with posterior contours when `emcee_store_samples` is enabled.

- **Gaia XP synthetic photometry** — Calibrate optical/NIR frames directly against Gaia DR3 XP spectra, avoiding cross-filter transformations when standard catalogs are unavailable. Supports custom transmission curves via `gaia_custom` catalog mode.

- **Robust difference imaging** — Multiple subtraction backends (SFFT, HOTPANTS, PyZOGY) with automatic fallback, kernel-order auto-selection, noise decorrelation (SFFT v1.5.0+), PSF-source pool supplementation for sparse fields, and optional inpainting of saturated template star cores.

- **Multi-S/N limiting magnitudes** — Generates limiting magnitude columns at multiple signal-to-noise thresholds (e.g., `Limit_3p0S2N`, `Limit_5p0S2N`) in a single run, using source injection with quiet-site selection and logistic-emcee recovery fitting.

- **TNS integration** — Automatic target coordinate and redshift lookup via the Transient Name Server when a target name is provided.

- **Per-filter catalog routing** — Different reference catalogs can be assigned to different filter groups (e.g., `griz` → RefCAT2, `u` → Gaia, `UBVRI` → APASS) within a single run.

- **Comprehensive diagnostic plots** — Source check overlays with distortion vectors and heatmap, WCS-vs-PSF offset quiver plots, alignment offset diagnostics with SExtractor centroid error bars, spalipy match visualisation, PSF residual and oversampled-ePSF panels, MCMC corner plots, and injection-recovery diagnostic plots.

## Quick Links
- **Conda Package**: [https://anaconda.org/astro-sean/autophot](https://anaconda.org/astro-sean/autophot)
- **Paper**: [A&A 667, A62 (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)
- **Issues**: [GitHub Issues](https://github.com/Astro-Sean/autophot/issues)

> [!NOTE]
> I am the sole developer and maintainer of AutoPhOT and also a [full-time researcher](https://astro-sean.github.io/index.html) at [MPE](https://www.mpe.mpg.de/person/144270/1302618).
> Please open issues on GitHub and I will do my best to resolve them as soon as possible.

---



## Installation

### Conda (Recommended)

**Important**: AutoPHOT requires the `conda-forge` channel for dependency resolution.

```bash
# Method 1: Install with conda-forge (recommended)
conda install -c conda-forge -c astro-sean autophot

# Method 2: Add conda-forge permanently
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c astro-sean autophot

# Method 3: Create dedicated environment
conda create -n autophot -c conda-forge -c astro-sean python=3.11
conda activate autophot
conda install -c astro-sean autophot
```

> [!NOTE]
> `sfft` and `sip_tpv` are not available on conda channels. After installing
> AutoPhOT via conda, also run:
> ```bash
> pip install sfft==1.7.3 sip_tpv==1.1
> ```

If conda struggles to resolve the environment, prefer `mamba`:

```bash
conda install -c conda-forge mamba
mamba create -n autophot -c conda-forge -c astro-sean python=3.11 autophot
conda activate autophot
```

Verify installation:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

### Install from source (developer / latest)

If you are running from a cloned repository, install it in editable mode so
internal modules are importable:

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
pip install -e .
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

### Listing All Parameters

```python
from autophot import list_parameters
list_parameters()
```

---

## CLI Entry Points

AutoPhOT provides four command-line tools:

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
```

or install via apt (with sudo rights):

```bash
sudo apt install astrometry.net
```

You will also need to get some index files for astrometry.net. You can download them from the [astrometry.net website](https://astrometry.net/data.html)


### Astromatic Suite (SExtractor, SCAMP, SWarp)

```bash
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp
```

### SFFT

For SFFT-based template subtraction:

```bash
pip install sfft
```

### HOTPANTS

For template subtraction with the `hotpants` method:

```bash
conda install -c conda-forge cfitsio make gcc
git clone https://github.com/acbecker/hotpants
cd hotpants && make
```

### Alignment Methods

AutoPhOT aligns the template image to the science image before difference
imaging.  Six methods are available, each with different strengths.  The
default cascade tries `swarp` → `reproject` → `astroalign` in sequence and
returns the first successful result.  Setting `alignment_method` to a
specific method skips the cascade and uses only that method (with fallback
to the cascade on failure).

#### Method overview

| Method | `alignment_method` | Install | Typical RMS | Speed |
|--------|---------------------|---------|-------------|-------|
| **SWarp** (SCAMP+SWarp) | `swarp` | Astromatic suite (required) | 0.1–0.5 px | Fast |
| **WCS Reproject** | `reproject` | `reproject` (bundled) | 0.1–0.3 px | Fast |
| **AstroAlign** | `astroalign` | `astroalign` (bundled) | 0.2–1.0 px | Medium |
| **spalipy** | `spalipy` | `pip install spalipy>=3.5` | 0.05–0.2 px | Medium |
| **tweakwcs** | `tweakwcs` | `pip install tweakwcs>=0.8` | 0.1–0.5 px | Medium |
| **chi2_shift** | `chi2_shift` | `pip install image-registration>=0.2` | 0.5–2.0 px | Fast |

#### SWarp (SCAMP + SWarp) — `swarp` (default)

Runs SExtractor on both images, cross-matches the source catalogs with
SCAMP to derive a per-frame astrometric solution, then resamples the
template onto the science pixel grid with SWarp.

**Pros**
- Industry-standard Astromatic toolkit; robust and well-tested.
- Handles large pixel-scale mismatches and rotations natively.
- Produces a refined WCS header as a by-product.
- Fast (subprocess-based, multi-threaded).

**Cons**
- Requires SExtractor, SCAMP, and SWarp binaries on `PATH`.
- Writes temporary FITS catalogs and resampled images to disk.
- SCAMP can fail when the field has too few catalog-matched stars
  (sparse fields, narrow filters, Galactic poles).
- Assumes a smooth (polynomial) WCS distortion model; cannot correct
  high-order or localised distortion that deviates from the fit.

#### WCS Reproject — `reproject`

Uses the existing WCS headers from both images to reproject the template
onto the science frame via `reproject` (interpolation or adaptive
resampling).  No source matching is performed.

**Pros**
- No source detection needed — works even on empty or extended fields.
- Very fast (pure Python, no subprocess).
- Handles arbitrary WCS projections (SIP, TPV, DW).
- No external binaries required.

**Cons**
- Accuracy is limited by the quality of the existing WCS headers.
  If either header has a systematic offset (common for older
  reductions), the aligned image will inherit that offset.
- Cannot correct for sub-pixel WCS errors or distortion residuals.
- Not suitable when the science and template have significantly
  different pixel scales or rotations (interpolation degrades).

#### AstroAlign — `astroalign`

Python-only image registration using triangle-based asterism matching
(similar to the classic "point-pattern matching" algorithm).  Detects
stars in both images, matches asterism triangles, and fits an affine
transform.

**Pros**
- Pure Python; no external binaries.
- Works well for images with similar pointings and scales.
- Fast for small images.

**Cons**
- Only fits a single affine transform (6 parameters) — no correction
  for spatially-varying distortion.
- Triangle matching can fail in crowded fields (too many false
  triangles) or sparse fields (too few stars).
- Does not update the WCS header.
- Typically achieves 0.2–1.0 px RMS, which may be insufficient for
  sharp PSFs or sub-pixel photometry.

#### spalipy — `spalipy` (recommended for wide-field)

Quad-based asterism matching for an initial affine transform, followed
by 2D thin-plate spline fitting to the residual source-position field.
This corrects non-homogeneous optical distortion that a single affine
or polynomial cannot capture.

**Pros**
- **Sub-pixel accuracy** (typically 0.05–0.2 px RMS) — the most
  accurate method when enough matched sources are available.
- Spline warp corrects spatially-varying distortion (optical
  aberrations, focal-plane distortions, differential atmospheric
  refraction).
- Handles reflected templates automatically (detects `det(A) < 0`
  from WCS and flips the template before alignment).
- Includes a sub-pixel shift correction after spline warping to
  remove residual constant offsets.
- Per-quadrant verification catches localised alignment failures.
- Source detection uses SExtractor for consistency with the rest of
  the pipeline.

**Cons**
- Requires `pip install spalipy>=3.5` (not on conda).
- Needs ≥ 20 matched sources for reliable spline fitting; sparse
  fields may fall back to affine-only (still good, but less accurate).
- Spline fitting can over-fit in regions with few matched stars
  (mitigated by `sub_tile` adaptive subdivision).
- Cannot perform reflections itself — relies on the pre-flip logic
  in the pipeline (which handles both x- and y-reflections via a
  180° rotation trick).
- Slightly slower than SWarp/reproject due to spline computation.

#### tweakwcs — `tweakwcs`

STScI-style WCS tweaking: detects sources in both images, cross-matches
them, and fits a corrective WCS transform (linear or polynomial) with
sigma-clipped outlier rejection.  The template is then resampled with
`reproject` using the tweaked WCS.

**Pros**
- Designed for HST/JWST-grade astrometric refinement.
- Produces a corrected WCS header (useful for downstream analysis).
- Sigma-clipped fitting is robust to false matches and outliers.
- Handles SIP/TPV distortion corrections natively.

**Cons**
- Requires `pip install tweakwcs>=0.8` (not on conda).
- Fits a global WCS correction (polynomial), not a local spline —
  cannot correct highly localised distortion as well as spalipy.
- Needs a reasonable initial WCS to converge; will not work if the
  headers are completely wrong.
- Slower than pure reproject due to the matching + fitting loop.

#### chi2_shift — `chi2_shift`

DFT-based cross-correlation shift measurement.  Does not detect point
sources; instead it correlates the full 2D image arrays to find the
global (dx, dy) shift that best aligns them.  Only a translation is
fit — no rotation, scale, or distortion correction.

**Pros**
- Works on extended-source-dominated fields (galaxy hosts, nebulae)
  where point-source detection fails.
- Very fast (FFT-based).
- No source detection or catalog matching required.
- Pure Python (`image-registration` package).

**Cons**
- Only corrects a constant (dx, dy) translation — no rotation, scale,
  or distortion handling.
- Accuracy is limited to ~0.5–2.0 px (cross-correlation peak width).
- Can be confused by bright edges, saturated stars, or large
  background gradients.
- Not suitable for images with significant rotation or pixel-scale
  differences.
- Requires `pip install image-registration>=0.2`.

#### Choosing a method

| Scenario | Recommended method |
|----------|--------------------|
| General use, good WCS headers | `swarp` (default) |
| Good WCS, no SExtractor/SCAMP available | `reproject` |
| Wide-field, sub-pixel accuracy needed | `spalipy` |
| HST/JWST data, WCS refinement needed | `tweakwcs` |
| Galaxy-host-dominated, few point sources | `chi2_shift` |
| Sparse field, no point sources | `reproject` or `chi2_shift` |
| Reflected template (det(A) < 0) | `spalipy` (auto-flips) |

Set the method in your YAML configuration:
```yaml
template_subtraction:
  alignment_method: spalipy  # or swarp, reproject, astroalign, tweakwcs, chi2_shift
```

Or install all optional alignment methods at once:
```bash
pip install -e ".[spalipy,tweakwcs,chi2-shift]"
```

#### Alignment quality diagnostics

After alignment, AutoPhOT computes a robust alignment RMS using mutual
nearest-neighbour source matching (SExtractor detections in both
images).  The following quality gates can be configured:

```yaml
template_subtraction:
  alignment_max_offset_px: 0.5   # reject if median offset > 0.5 px
  alignment_max_rms_px: 0.75     # reject if RMS > 0.75 px
  alignment_max_p95_px: 1.5      # reject if 95th percentile > 1.5 px
  alignment_min_sources_for_field_gate: 20  # min sources before gates apply
  post_swarp_verify: True        # cross-match SExtractor detections after SWarp
```

When an alignment is rejected, the pipeline falls back to the next
method in the cascade.  Two diagnostic plots are produced:
- **`Alignment_Offset_*.png`** — quiver plot of per-source (dx, dy)
  offsets with error bars from SExtractor centroid uncertainties.
- **`SpalipyMatch_*.png`** — side-by-side view of matched sources
  (spalipy method only).

---

## PSF Photometry

AutoPhOT builds an empirical ePSF model from in-frame stars using
`photutils` ePSFBuilder, with optional adaptive oversampling for
undersampled images (FWHM < 2.5 px).  Three fitters are available:

### Fitters

| Fitter | Config key | Use case |
|--------|-----------|----------|
| **Least-squares** (default) | — | Fast, general-purpose |
| **Poisson likelihood** | `use_poisson_likelihood_fitter: True` | Low-count regime; statistically superior to χ² (Fermilab TM-2543-AE) |
| **MCMC (emcee)** | `perform_emcee_fitting_s2n: 10` | Bayesian uncertainty estimation; triggered when target S/N < threshold |

### MCMC configuration

The emcee fitter runs adaptively — it extends the chain until the
integrated autocorrelation time stabilises (up to 50,000 steps), then
applies principled burn-in (10×τ discard) and thinning (τ/2).  Key
parameters:

```yaml
photometry:
  perform_emcee_fitting_s2n: 10   # run MCMC when target S/N < this
  emcee_nwalkers: 32              # number of walkers
  emcee_nsteps: null              # null = adaptive; or set fixed step count
  emcee_burnin_frac: 0.3          # burn-in fraction (overridden by 10×τ rule)
  emcee_thin: 10                  # thinning factor
  emcee_adaptive_tau_target: 50   # target autocorrelation time for convergence
  emcee_min_autocorr_N: 100       # min chain length before autocorr checks
  emcee_store_samples: False      # store full chains for corner plots
  emcee_threads: 1                # parallel threads
```

When `emcee_store_samples` is enabled (or for the target source), a
corner plot of the posterior is saved as `PSF_MCMC_corner_*.png`.

### Inverted-fit detection

For template-subtracted images where the transient may appear as a
negative residual (fading source), AutoPhOT can fit the target on a
sign-flipped image:

```yaml
photometry:
  check_inverted_image: True
```

When enabled, the pipeline fits both the normal and inverted (×−1)
images, and replaces poor normal fits with inverted fits when
appropriate.  Results are flagged with an `_inverted_fit` column in
the output table.

### PSF star selection

PSF model stars are selected using SExtractor detection with quality
cuts on saturation, elongation, isolation, FWHM fraction, and
CLASS_STAR.  FFT-based rejection removes stars with close companions
or residual structure.  Key parameters:

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
| **SFFT** | `sfft` | `pip install sfft` | Default; supports noise decorrelation, B-spline kernel, variable-star rejection |
| **HOTPANTS** | `hotpants` | Build from source | Classic kernel-matching algorithm |
| **PyZOGY** | `zogy` | Bundled | Optimal for PSF-matched subtraction; propagates noise correctly |

### SFFT advanced features

```yaml
template_subtraction:
  method: sfft
  kernel_order: 0                  # 0=constant, 1=linear, 2=quadratic, 3=cubic, or "auto"
  kernel_hw_fwhm_multiplier: 2.5   # kernel half-width as FWHM multiplier
  sfft_forceconv: AUTO             # AUTO, REF, or SCI — which image to convolve
  sfft_decorrelate_noise: True     # apply noise decorrelation (SFFT v1.5.0+)
  sfft_save_decorrelated: True     # save decorrelated difference image separately
  sfft_use_bspline_kernel: False   # B-spline kernel (requires CUDA/Cupy)
  sfft_bg_order: 1                 # background spatial polynomial order
  sfft_crowded_auto: False         # auto-enable crowded-field tuning
  sfft_use_post_anomaly_feedback: True  # extra pass using post-anomaly sources
```

### Template inpainting

Saturated star cores in the template can be inpainted before
subtraction to prevent artifacts:

```yaml
template_subtraction:
  inpaint_template_cores: False    # enable inpainting
  inpaint_method: biharmonic       # algorithm: biharmonic or telea
  inpaint_saturate_frac: 0.90      # saturation threshold for core mask
  inpaint_dilate_radius: 6         # mask dilation radius (px)
```

---

## Limiting Magnitudes

AutoPhOT computes limiting magnitudes via source injection and
recovery at multiple S/N thresholds.  The default produces `Limit_3p0S2N`
and `Limit_5p0S2N` columns in the output table.

```yaml
limiting_magnitude:
  snr_thresholds: [3, 5]           # S/N thresholds for limit columns
  recovery_method: logistic_emcee  # recovery fitting method
  injection_strategy: ring_quait   # injection placement: ring_quiet or uniform
  injection_n_sites: 25            # number of injection sites
  inject_min_radius_fwhm: 2.0      # min injection radius from target
  inject_max_radius_fwhm: 6.0      # max injection radius from target
  plot_injection_recovery: False   # generate InjectionRecovery diagnostic plot
```

---

## Supported Catalogs

AutoPHOT supports the following photometric reference catalogs (case-insensitive):

- **gaia** - Gaia DR3 with XP spectra (default for most use cases)
- **pan_starrs** / **panstarrs** / **ps1** - Pan-STARRS DR1/DR2
- **sdss** - Sloan Digital Sky Survey
- **apass** - AAVSO Photometric All-Sky Survey
- **2mass** - 2-Micron All Sky Survey (infrared)
- **legacy** - Legacy Survey (DR8+)
- **skymapper** - SkyMapper Southern Sky Survey
- **refcat** - PS1-based reference catalog (requires MAST CasJobs credentials)
- **tic** - TESS Input Catalog
- **custom** - User-provided CSV catalog (set `catalog.catalog_custom_fpath`)
- **gaia_custom** - Gaia DR3 with user-provided transmission curves

Set the catalog in your YAML configuration:
```yaml
catalog:
  use_catalog: gaia  # or any of the above
```

### Per-filter catalog routing

Different catalogs can be assigned to different filter groups:
```yaml
catalog:
  use_catalog:
    griz: refcat
    u: gaia
    UBVRI: apass
    default: gaia
```

### Gaia custom transmission curves

For non-standard filters, provide transmission curve files and use
`gaia_custom`:
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

## Diagnostic Plots

AutoPhOT generates several diagnostic plots (saved as PNG alongside
the reduced data):

| Plot | Filename pattern | Description |
|------|-------------------|-------------|
| Source check | `SourceCheck_*.png` | WCS overlay with detected sources, distortion residual vectors, and optional heatmap |
| WCS vs PSF offset | `WCS_vs_PSF_Offset_*.png` | Quiver plot of PSF-fit vs WCS-catalog position offsets |
| Alignment offset | `Alignment_Offset_*.png` | Per-source (dx, dy) offsets after template alignment with error bars |
| Spalipy matches | `SpalipyMatch_*.png` | Side-by-side matched-source visualisation (spalipy method) |
| PSF model | `PSF_Target_*.png` | Science image, PSF model, residuals, and oversampled ePSF |
| MCMC corner | `PSF_MCMC_corner_*.png` | Posterior corner plot from emcee fitting |
| Cosmic rays | `CosmicRay_*.png` | Before/after cosmic-ray cleaning |
| Injection recovery | `InjectionRecovery_*.png` | Injected-source recovery diagnostic |
| Subtraction check | `SubtractionCheck_*.png` | Science, template, and difference image panels |

### Source check distortion overlays

The source check plot can overlay residual distortion vectors and a
gridded distortion-magnitude heatmap:

```yaml
alignment:
  plot_source_check_distortion_vectors: True
  plot_source_check_distortion_grid_map: False
  plot_source_check_distortion_grid_colorbar: True
  plot_source_check_distortion_grid_alpha: 0.35
  plot_source_check_distortion_grid_cmap: viridis
  plot_source_check_max_sep_pix: 6.0
  plot_source_check_max_vectors: 300
```

---

### Post-Processing Products

```python
from lightcurve import plot_lightcurve, generate_photometry_table, check_detection_plots

# Lightcurve plot with detections and limits
plot_lightcurve(output_file, snr_limit=3, method="PSF")

# ASCII photometry table
# Output: lightcurve_PSF.dat with columns:
# MJD, Date, Mag, Error, Filter, Limit
generate_photometry_table(output_file, snr_limit=3, method="PSF")

# Sort detection plots into organised folders
check_detection_plots(detections_loc, method="PSF")
```

Notes on outputs:
- The pipeline's default photometry CSV (`lightcurve_output.csv`) is **long-form**: one row per image with a per-row `filter` column and uniform columns like `mag_psf`, `zp_psf`, etc.
- `plot_lightcurve` / `generate_photometry_table` understand this format and will group points by the `filter` column.
- Multi-S/N limiting magnitude columns (e.g., `Limit_3p0S2N`, `Limit_5p0S2N`) are generated automatically based on `snr_thresholds` config.
- Inverted-fit results are flagged with an `_inverted_fit` boolean column.

---


### Environment Variables

For TNS lookups and catalog access (i.e. for RefCAT2), set these (do not hard-code):

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
> FITS images **must** have `TELESCOP` and `INSTRUME` header keywords, plus a bandpass keyword (e.g., `FILTER`). Images without these will be ignored.


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

# PSF photometry with MCMC
config["photometry"]["perform_emcee_fitting_s2n"] = 10
config["photometry"]["check_inverted_image"] = True

# Enable template subtraction
config["template_subtraction"]["do_subtraction"] = True
config["template_subtraction"]["method"] = "sfft"
config["template_subtraction"]["alignment_method"] = "spalipy"  # sub-pixel accuracy
config["template_subtraction"]["kernel_order"] = 1

# Create template directories
prepare_template_directory(
    fits_dir=config["fits_dir"],
    include_legacy_p_folders=False,
    confirm_before_continue=True,
)

# Run photometry
output = AutomatedPhotometry.run_photometry(default_input=config, do_photometry=True)

# Generate plots and tables
from lightcurve import plot_lightcurve, check_detection_plots, generate_photometry_table
plot_lightcurve(output, snr_limit=3, method="PSF")
generate_photometry_table(output, snr_limit=3, method="PSF")
```


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

    # Optional credentials from environment (do not hard-code secret keys):
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
    autophot_input["template_subtraction"]["alignment_method"] = "swarp"
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
   - `autophot_input["template_subtraction"]["alignment_method"] = "swarp"` (recommended; SCAMP+SWarp) or `"spalipy"` (sub-pixel accuracy)
   - `autophot_input["template_subtraction"]["method"] = "sfft"` (or `hotpants`, `zogy`)
2. Create template directories:
   - Call `prepare_template_directory(...)`.
  - AutoPhOT prints where folders were created and asks if you want to continue.
3. Place template FITS files:
   - Put one usable template per filter in `fits_dir/templates/<filter>_template/`.
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
