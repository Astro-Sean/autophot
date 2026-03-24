AutoPHoT Logo

[Anaconda Version](https://anaconda.org/astro-sean/autophot)  
[Latest Release Date](https://anaconda.org/astro-sean/autophot)  
[Platforms](https://anaconda.org/astro-sean/autophot)  
[License](https://anaconda.org/astro-sean/autophot)  
[Downloads](https://anaconda.org/astro-sean/autophot)

---

# AutoPHoT

**The Automated Photometry of Transients Pipeline**

AutoPHoT is a Python pipeline designed for publication-quality photometry of transients and variable sources. It processes CCD/NIR FITS images through WCS solving, cosmic-ray removal, and background subtraction. The pipeline builds or uses reference catalogues (e.g., Gaia, Pan-STARRS), measures aperture and PSF photometry, and calibrates zeropoints with robust fitting and optional colour terms. Optional template subtraction (SFFT by default, or HOTPANTS/ZOGY) produces difference images for transient detection. AutoPHoT reports target magnitudes, errors, limiting magnitudes (via injection/recovery), and can output light curves and detection-limit plots.

**Key Features:**

- Aperture and PSF photometry
- Template subtraction (SFFT, HOTPANTS, ZOGY)
- Light curve and detection-limit plots
- Parallel processing support

**Project Links:**

- [GitHub Repository](https://github.com/Astro-Sean/autophot)
- [Publication (ADS)](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)

---

## Table of Contents

1. [Installation](#installation)
2. [Optional External Tools](#optional-external-tools)
3. [Quick Start](#quick-start)
4. [Template Subtraction Setup](#template-subtraction-setup)
5. [Citation](#citation)

---

## Installation

### Using Conda

Install AutoPHoT from the `astro-sean` Conda channel:

```bash
conda install astro-sean::autophot
```

For best results, create a fresh environment:

```bash
conda create -n autophot python=3.11 -y
conda activate autophot
conda install astro-sean::autophot
```

To update AutoPHoT:

```bash
conda update astro-sean::autophot
```

**Verify Installation:**

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPHoT import successful')"
autophot-main -h
```

---

## Optional External Tools

### PyZOGY (for ZOGY Subtraction)

```bash
git clone https://github.com/dguevel/PyZOGY
cd PyZOGY
python setup.py install
cd ..
```

### Astrometry.net (`solve-field`) for WCS Solving

```bash
conda install -c conda-forge astrometry-net
solve-field --help
```

- Download index files from [Astrometry.net](https://data.astrometry.net/) and set the `ASTROMETRY_NET_DATA_DIR` environment variable.

### Astromatic Tools (SExtractor/SCAMP/SWarp)

```bash
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp
```

### HOTPANTS for Template Subtraction

```bash
conda install -c conda-forge cfitsio make gcc
git clone https://github.com/acbecker/hotpants
cd hotpants
make
```

---

## Quick Start

### Python Driver Script Example

```python
#!/usr/bin/env python3
from autophot import AutomatedPhotometry

# Load default settings
autophot_input = AutomatedPhotometry.load()
autophot_input["outdir_name"] = "REDUCED"
autophot_input["wdir"] = "/path/to/autophot_db"
autophot_input["fits_dir"] = "/path/to/images/"

# Set target coordinates
autophot_input["target_name"] = "SNXXXXabc"
autophot_input["target_ra"] = 123.456789
autophot_input["target_dec"] = -12.345678

# Configure catalogues, photometry, and WCS
autophot_input["catalog"]["use_catalog"] = {"griz": "refcat", "u": "gaia", "UBVRI": "apass"}
autophot_input["photometry"]["psf_oversample"] = 2
autophot_input["wcs"]["redo_wcs"] = True

# Enable template subtraction
autophot_input["template_subtraction"]["do_subtraction"] = True
autophot_input["template_subtraction"]["method"] = "sfft"

# Run AutoPHoT
loc = AutomatedPhotometry.run_photometry(default_input=autophot_input)
```

---

## Template Subtraction Setup

### Directory Structure

Templates must be organised into per-filter subfolders under `fits_dir/templates/`:

```
my_field/
├── science_2024_01_15_r.fits
└── templates/
    ├── r_template/
    │   └── r_template.fits
    └── K_template/
        └── K_template.fits
```

### Configuration

```python
autophot_input["template_subtraction"]["do_subtraction"] = True
autophot_input["template_subtraction"]["alignment_method"] = "reproject"  # or "swarp", "astroalign"
autophot_input["template_subtraction"]["method"] = "sfft"  # or "hotpants", "zogy"
```

---

## Citation

If you use AutoPHoT in your research, please cite the following publication:

- [ADS Link](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)

**BibTeX Entry:**

```bibtex
@ARTICLE{2022A&A...667A..62B,
       author = {{Brennan}, S.~J. and {Fraser}, M.},
        title = "{The Automated Photometry of Transients pipeline (AUTOPHOT)}",
      journal = {\aap},
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