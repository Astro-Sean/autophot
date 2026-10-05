![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT: Automated Photometry Of Transients

AutoPhOT is a photometry pipeline for following up transients and variable sources, built on [Photutils](https://photutils.readthedocs.io/) and [Astropy](https://www.astropy.org/). It takes FITS frames from any telescope, filter, and pixel scale, finds the target in each frame, and produces one calibrated light curve.

- Conda package: [anaconda.org/astro-sean/autophot](https://anaconda.org/astro-sean/autophot)
- Paper: [A&A 667, A62 (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)
- Issues: [GitHub Issues](https://github.com/Astro-Sean/autophot/issues)

> [!NOTE]
> I am the sole developer and maintainer of AutoPhOT and also a [full-time researcher](https://astro-sean.github.io/index.html) at [MPE](https://www.mpe.mpg.de/person/144270/1302618).
> Please open issues on GitHub and I will do my best to resolve them as soon as possible.

## What it does

1. Sorts frames by telescope, instrument, and filter; checks or re-solves the WCS with astrometry.net and Gaia DR3. SIP and TPV distortion are handled.
2. Prepares each image: cosmic-ray rejection, satellite-trail detection, background estimation, FWHM measurement.
3. Runs aperture and PSF photometry at the target position, with an empirical ePSF built from in-frame stars.
4. Calibrates against a catalog (Gaia DR3, Pan-STARRS, SDSS, APASS, 2MASS, Legacy Survey, SkyMapper, and more), with per-filter routing and Gaia XP synthetic photometry.
5. Optionally subtracts a template (SFFT, HOTPANTS, or ZOGY) and scores every difference image for quality.
6. Measures limiting magnitudes by injecting and recovering artificial sources.

## Installation

### Conda (recommended)

```bash
conda install -n base -c conda-forge conda-libmamba-solver
conda config --set solver libmamba

conda create -n autophot -c conda-forge -c astro-sean python=3.11 autophot
conda activate autophot

pip install sfft==1.7.3 sip_tpv==1.1  # not on conda channels
```

Check the install:

```bash
python -c "from autophot import AutomatedPhotometry; print('OK')"
autophot-main -h
```

### From source

```bash
git clone https://github.com/Astro-Sean/autophot.git
cd autophot
pip install -e .
pip install sfft==1.7.3 sip_tpv==1.1
```

`environment.yml` pins every dependency for a reproducible setup:

```bash
conda env create -f environment.yml
conda activate autophot
pip install -e .
```

## Quick start

```python
from autophot import AutomatedPhotometry

config = AutomatedPhotometry.load()
config["fits_dir"] = "/path/to/your/images"
config["target_name"] = "SN2024A"
config["target_ra"] = 123.456789
config["target_dec"] = -12.345678

output_file = AutomatedPhotometry.run_photometry(default_input=config)
```

A few things worth knowing up front:

- `from autophot import list_parameters; list_parameters()` prints every config key and its default.
- FITS headers must carry `TELESCOP`, `INSTRUME`, and a bandpass keyword (e.g. `FILTER`); images without them are skipped.
- `config["nCPU"] = 4` (or `AUTOPHOT_NCPU=4`) processes images in parallel, each with its own log. Worker settings like `photometry.aperture_n_jobs` multiply `nCPU`, so keep the product at or below your core count.
- `python batch_main.py -f im1.fits im2.fits -c config.yml --jobs 4` runs a file list without a driver script.

## Command-line tools

| Command | Purpose |
|---------|---------|
| `autophot-main` | Run the full photometry pipeline |
| `autophot-driver` | Interactive driver with template setup |
| `autophot-gaia-curves` | Build a Gaia custom catalog from transmission curves |
| `autophot-inspect-telescope` | Inspect telescope header keywords |

## Optional dependencies

- **astrometry.net** (`solve-field`): needed when frames have no WCS. `conda install conda-forge::astrometry`; index files from [astrometry.net](https://astrometry.net/data.html). [repo](https://github.com/dstndstn/astrometry.net) - [Lang et al. 2010, AJ 139, 1782](https://ui.adsabs.harvard.edu/abs/2010AJ....139.1782L/abstract)
- **Astromatic suite** (SExtractor, SCAMP, SWarp): `conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp` - [astromatic.net](https://www.astromatic.net/) - SExtractor: [Bertin & Arnouts 1996](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract); SCAMP: [Bertin 2006](https://ui.adsabs.harvard.edu/abs/2006ASPC..351..112B/abstract); SWarp: [Bertin et al. 2002](https://ui.adsabs.harvard.edu/abs/2002ASPC..281..228B/abstract)
- **SFFT**: `pip install sfft==1.7.3` - [repo](https://github.com/thomasvrussell/sfft) - [Hu et al. 2022, ApJ 936, 157](https://ui.adsabs.harvard.edu/abs/2022ApJ...936..157H/abstract)
- **HOTPANTS**: needs cfitsio; `git clone https://github.com/Astro-Sean/hotpants && cd hotpants && make` - [repo](https://github.com/Astro-Sean/hotpants) (fork of acbecker/hotpants with the build fixed for macOS and modern GCC/Clang) - [Becker 2015, ascl:1504.004](https://ui.adsabs.harvard.edu/abs/2015ascl.soft04004B/abstract); method from [Alard & Lupton 1998, ApJ 503, 325](https://ui.adsabs.harvard.edu/abs/1998ApJ...503..325A/abstract)
- **ZOGY**: built in (self-contained numpy implementation, nothing to install) - based on [pmvreeswijk/ZOGY](https://github.com/pmvreeswijk/ZOGY) - [Zackay, Ofek & Gal-Yam 2016, ApJ 830, 27](https://ui.adsabs.harvard.edu/abs/2016ApJ...830...27Z/abstract)
- **MaxiMask**: CNN defect mask that catches contaminants the heuristic masks miss - worthwhile in complex/crowded fields. Inference is slow on CPU (order a minute per frame) but can run on GPU (`maximask_allow_gpu`). `pip install maximask-and-maxitrack tensorflow`, enable with `use_maximask: True` - [repo](https://github.com/mpaillassa/MaxiMask) - [Paillassa, Bertin & Bouy 2020, A&A 634, A49](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..49P/abstract)

## Alignment

The template is registered to the science image before subtraction. `spalipy` is the default; the pipeline checks each result against offset, RMS, and p90 gates scaled to the image FWHM, and falls through the remaining methods on failure.

| Method | `alignment_method` | Install |
|--------|--------------------|---------|
| spalipy (default) | `spalipy` | `pip install spalipy>=3.5` |
| SCAMP + SWarp | `swarp` | Astromatic suite |
| WCS reproject | `reproject` | bundled |
| AstroAlign | `astroalign` | bundled |
| tweakwcs | `tweakwcs` | `pip install tweakwcs>=0.8` |
| chi2_shift | `chi2_shift` | `pip install image-registration>=0.2` |

Gate thresholds live under `template_subtraction` (`alignment_max_offset_px`, `alignment_max_rms_px`, `alignment_max_p90_px`). Install all optional methods with `pip install -e ".[spalipy,tweakwcs,chi2-shift]"`.

## Template subtraction

| Method | `method` | Notes |
|--------|----------|-------|
| SFFT (default) | `sfft` | Spatially varying kernel; variable-star rejection; optional B-spline refinement |
| HOTPANTS | `hotpants` | Classic kernel matching; build from source |
| ZOGY | `zogy` | PSF-matched subtraction; built-in implementation, no external package needed |

Shared options under `template_subtraction`:

- `forceconv`: `REF` (default), `SCI`, or `AUTO`. REF convolves the reference to the science PSF, so the science ePSF applies directly to the difference image. For ZOGY the matching kernel is probed in both directions first; a direction that would deconvolve (a kernel with strong negative sidelobes) is vetoed, flipped to the cleaner direction, or replaced by canonical ZOGY. The direction actually used is written to the `CONVD` header card.
- `kernel_order`: SFFT kernel polynomial degree, or `"auto"` (default).
- `inpaint_template_cores`: fill saturated template cores before subtraction.

Sources on masked pixels are dropped before the kernel fit, and variable sources and the target itself are excluded from scale and PSF estimation. Every difference image gets a structured quality check - dipoles, bright-star residuals, background variation, autocorrelation, edge artifacts - written to the FITS header (`DIFFQUAL`, `DIFFQSCR`) and a `diff_quality_<base>.json` manifest.

Set `do_subtraction: True`, then place one template FITS per filter in `fits_dir/templates/<filter>_template/` (`prepare_template_directory()` builds the layout for you).

## Photometry

The PSF is an empirical ePSF built with photutils; oversampling is raised automatically on undersampled images (FWHM < 2.5 px). PSF stars are cut on saturation, elongation, isolation, FWHM consistency, and concentration, plus an FFT check for close companions.

| Fitter | Config | When to use |
|--------|--------|-------------|
| Least-squares | default | General use |
| Poisson likelihood | `use_poisson_likelihood_fitter: True` | Low-count regime |
| emcee (MCMC) | `perform_emcee_fitting_s2n: 10` | S/N below threshold; Bayesian uncertainties, adaptive chains |

On difference images, `photometry.check_inverted_image: True` recovers a fading source as a negative PSF dip; those rows carry an `_inverted_fit` flag.

## Catalogs

| Catalog | `use_catalog` | Magnitude system | Notes |
|---------|---------------|------------------|-------|
| Gaia DR3 + XP | `gaia` | ugriz `abmag`, BVRI `vegamag` | Default for most filters; synthetic mags from XP spectra (SDSS_Std = AB, JKC_Std = Vega) |
| Pan-STARRS | `pan_starrs` / `ps1` | `abmag` | DR1/DR2 |
| SDSS | `sdss` | `abmag` | |
| APASS | `apass` | BV `vegamag`, gri `abmag` | Johnson BV are Vega, Sloan bands are AB |
| 2MASS | `2mass` | `vegamag` | Infrared |
| Legacy Survey | `legacy` | `abmag` | DR8+ |
| SkyMapper | `skymapper` | `abmag` | Southern sky |
| RefCAT2 | `refcat` | griz `abmag`, JHK `vegamag` | Needs MAST CasJobs credentials; optical ATLAS-derived, IR from 2MASS |
| TIC | `tic` | ugriz `abmag`; BV, JHK, G, T `vegamag` | TESS Input Catalog; mixes source systems |
| Custom CSV | `custom` | user-defined | Set `catalog.catalog_custom_fpath`; reported as `unknown` |
| Gaia + custom curves | `gaia_custom` | `abmag` | User transmission curves; magnitudes synthesized in AB |
| Auto-select | `auto` | per resolved catalog | Picks the best catalog per band |

The system of the calibrating magnitudes is resolved per band for the mixed catalogs above, printed in the run log next to the catalog choice and the zeropoint table, and written to `output.csv` as `magsys` (`abmag`, `vegamag`, `mixed`, or `unknown`). For a `custom` catalog the system cannot be inferred, so `magsys` reports `unknown` unless you are certain of the convention your columns follow.

Different catalogs can serve different filters in one run:

```yaml
catalog:
  use_catalog:
    griz: refcat
    u: gaia
    UBVRI: apass
    default: gaia
```

Mapping keys take single bands (`u`), families (`UBVRI`, `JHK`), mixes (`uRI`), or lists (`u, RI`). Case matters: `r` is SDSS r, `R` is Johnson-Cousins R. Bad keys fall back to `default` with a warning.

With `use_catalog: auto`, the pipeline scans every image footprint once before processing, downloads each feasible catalog, and keeps the one with the most usable calibrators per band. The resolved mapping is cached under `<wdir>/catalog_queries/` so restarts skip the rescan; Gaia stays available as a fallback but is excluded from the bulk scan to protect the archive servers. For non-standard filters, supply curve files via `gaia_custom` and `catalog.transmission_curve_map`.

## Limiting magnitudes

Measured by injecting artificial sources and checking recovery, at one or more S/N thresholds. Defaults write `Limit_3p0S2N` and `Limit_5p0S2N` columns; thresholds and injection strategy live under `limiting_magnitude`.

## Post-processing

```python
from lightcurve import (
    plot_lightcurve, plot_variability_check,
    generate_photometry_table, check_detection_plots,
)

plot_lightcurve(output_file, snr_limit=3, method="PSF")      # detections + limits
plot_variability_check(output_file, method="PSF")            # target vs reference stars
generate_photometry_table(output_file, snr_limit=3, method="PSF")  # ASCII table
check_detection_plots(detections_loc, method="PSF")          # sort plots into folders
```

The output CSV is long-form: one row per image with a `filter` column. Lightcurve x-axes default to MJD and switch to minutes or hours for spans under a day. `plot_variability_check` removes each epoch's common-mode drift before comparing the target to the reference ensemble.

## Environment variables

Needed for TNS lookups and RefCAT2;

```bash
export MASTCASJOBS_WSID="..."
export MASTCASJOBS_PWD="..."
export TNS_BOT_ID="..."
export TNS_BOT_NAME="..."
export TNS_BOT_API="..."
```


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
