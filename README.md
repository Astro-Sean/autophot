<p align="center">
  <img src=https://github.com/Astro-Sean/autophot/blob/master/logo.png>
</p>



:warning: **New version available**: I have taken some time to update the code, integrating new functions from Photutils and Astropy, as well as spending some time on template subtraction. This branch contains the updated code, with limited documentation for now. If you encounter any issues, please contact [me](mailto:sean.brennan@astro.su.se?subject=New_feature_in_AutoPhOT) or open an issue.


<div align="center">
I am currently the sole developer and maintainer of the AutoPhOT code.

Your pateince is appreciated will I address bugs and implement changes. If you would like to collaborate, do get in touch — Seán
</div>



## Introduction

[AutoPhOT](https://arxiv.org/abs/2201.02635) is a novel, automated pipeline designed for rapid, publication-quality photometry of transients.The pipeline is built from the ground up using Python 3, with no dependencies on legacy software. Some features of the code include

- Automatic aperture and Point Spread Function (PSF) photometry (using ePSF modeling from Photutils)
- Template subtraction using HotPants, ZOGY, or SFFT
- Limiting magnitude measurement
- Integrated with SDSS, PanSTARRS, 2MASS, Skymapper, APASS, or REFCAT2 catalogs
- 'One-click' operation


## Installation

We suggest creating a new environment for AutoPhOT. This can be done using [conda](https://docs.conda.io/en/latest/) by running:

```bash
conda create -n autophot_env python=3.7
```

then to activate this environment, run:

```bash
conda activate autophot_env
```


Before installing autophot, you'll need conda-forge and astropy in your conda channels. See [here](https://github.com/Astro-Sean/autophot/issues/1) for help.

There was issues installing the code due to conflict packages from various independant software packages. With this updated version, please follow the script given below.

## How to use AutoPhOT

Below is a script that will execute AutoPhOT on all FITS images in a given folder. While providing limited functionality (compared to version 1 of the code), the following script will be suitable in most cases.

```Python

import sys

# Add the path to the autophot library to the system path
sys.path.append('/home/seanbrennan/Documents/autophot_object')

# Import necessary modules from autophot
from autophot import automated_photometry
import autophot_tokens

# Load default input parameters for autophot
autophot_input = automated_photometry.load()

# Set the name of the output directory where results will be saved
autophot_input['outdir_name'] = 'REDUCED'

# Set the working directory where calibration files will be saved
autophot_input['wdir'] = '/home/seanbrennan/Desktop/autophot_db'

# Set the directory where the FITS files (astronomical images) are located
autophot_input['fits_dir'] = 'path to directory of fits files'

# Specify the target's RA (Right Ascension) and Dec (Declination) in degrees
# You can also specify the target by name using the 'target_name' parameter (shown below)
autophot_input['target_ra'], autophot_input['target_dec'] = 123.00, 456.00

# Choose the catalog to be used for astrometry and photometry.
# Options: 'pan_starrs', 'sdss', 'skymapper', 'apass', 'refcat', or '2mass'
autophot_input['catalog']['use_catalog'] = 'refcat'

# If using 'refcat', set the API details for MAST Casjobs. Leave as None if not available
autophot_input['catalog']['MASTcasjobs_wsid'] = None
autophot_input['catalog']['MASTcasjobs_pwd'] = None

# Set whether to remove cosmic rays from the images (optional)
autophot_input['cosmic_rays']['remove_cmrays'] = False

# Set WCS (World Coordinate System) options:
# - 'remove_wcs': Whether to remove the existing WCS from the FITS header
# - 'redo_wcs': Whether to redo the WCS using Astrometry.net (recommended)
# - 'guess_scale': Set to False unless you're unsure of the image scale
autophot_input['wcs']['remove_wcs'] = True
autophot_input['wcs']['redo_wcs'] = True
autophot_input['wcs']['guess_scale'] = False

# Enable automatic determination of the optimum aperture radius for photometry
autophot_input['photometry']['find_optimum_radius'] = True

# Set whether to perform only aperture photometry (set to False to include PSF photometry)
autophot_input['photometry']['do_ap_phot'] = False

# Optionally, specify a target name to use instead of RA/Dec, useful for automated lookup
autophot_input['target_name'] = '2024ABC'

# Set API credentials for TNS (Transient Name Server) BOT if available, to retrieve coordinates
autophot_input['wcs']['TNS_BOT_ID'] = autophot_tokens.TNS_BOT_ID
autophot_input['wcs']['TNS_BOT_NAME'] = autophot_tokens.TNS_BOT_NAME
autophot_input['wcs']['TNS_BOT_API'] = autophot_tokens.TNS_BOT_API

# Set the path to the solve-field executable used for WCS solving
autophot_input['wcs']['solve_field_exe_loc'] = 'filepath for the solve_field executable'

# Template subtraction parameters:
# - 'get_panstarrs_templates': Whether to download Pan-STARRS templates
# - 'panstarrs_templates_size': Size of the image in pixels (with a pixel scale of 0.25"/pixel)
autophot_input['template_subtraction']['get_panstarrs_templates'] = False
autophot_input['template_subtraction']['panstarrs_templates_size'] = 1500

# Perform template subtraction and set method details:
# - 'do_subtraction': Command to perform template subtraction
# - 'use_astroalign': Use astroalign (True) or reproject_interp (False) to align images
# if you image is undersmpled I suggest using useing astroalign
autophot_input['template_subtraction']['do_subtraction'] = True
autophot_input['template_subtraction']['use_astroalign'] = True

# Set the method to use for template subtraction: 'sfft', 'zogy', or 'hotpants'
autophot_input['template_subtraction']['method'] = 'sfft'

# Set the path to the hotpants executable, if using the 'hotpants' method for subtraction
autophot_input['template_subtraction']['hotpants_exe_loc'] = 'filepath for the hotpants executable'

# Run the photometry process with the specified input parameters
loc = automated_photometry.run_photometry(default_input=autophot_input, do_photometry=1)

# Import plotting functions for light curve analysis
from lightcurve import plot_lightcurve, check_detection_plots

# Generate the light curve plot with signal-to-noise ratio (SNR) limits and PSF photometry
detections_loc = plot_lightcurve(loc, show_limits=True, snr_limit=3, method='PSF', format='png')

# Check the detection plots to visually verify the PSF photometry results
check_detection_plots(detections_loc, method='PSF')
```


## Template subtraction

AutoPhOT is designed for transient photometry, where it is often necessary to remove "reference" flux using template subtraction. While AutoPhOT will handle the template subtraction, it needs the templates. You can use the code to download images from Pan-STARRS or provide your own.

To provide your own, you will need to have the correct directory structure as follows:



```
├── rband_image.fits
└── templates
    └── rp_template
        └── rp_template.fits
```


**Note:** You need the templates folder within the directory set by *fits_dir*, and each subfolder needs to be named X_template, where X can be *up, rp, gp, ip, or zp* for Sloan filters, or *U, B, R, I, or Z* for Johnson-Cousins filters.


## Additional functionality


<h4>Astrometry.Net</h4>


AutoPhOT relies on [Astrometry.net](https://arxiv.org/abs/0910.2233) by Dustin Lang to solve for WCS. While the code can be downloaded/installed [here](http://astrometry.net/doc/readme.html) and [here](http://astrometry.net/doc/build.html#build), we suggest using [Homebrew](https://brew.sh/) to install Astrometry.net.



```bash
brew install astrometry-net
```

To make sure everything is setup correctly, we can run the following in the terminal:

```bash
solve-field
```

In order for Astometry.net to run successfully, it requires pre-indexed files for calibration. Firstly, we can create a new directory called "data".

```bash
mkdir data
cd data/
```

Next, while in the data directory, we can run the following to obtain these index files (~40Gb):

```bash
wget -r -np http://broiler.astrometry.net/~dstn/4200/
```

This will download all the index files to the *data/* folder. Once this download is completed, this *data* folder must be placed in the correct location so that Astrometry.net can find it.

We can search for the location of the solve-field command using the following:

```bash
which solve-field
```

It should be something similar to /usr/local/Cellar/astrometry-net/0.85_1/solve-field (although maybe not exactly). Move our *data* folder to the parent directory using:

```bash
cd ../
cp -R data /usr/local/Cellar/astrometry-net/0.85_1/
```

To update AutoPhOT on the location of Astrometry.Net,  update (if needed) 'solve_field_exe_loc' in the autophot_input dictionary (see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb) for example).

If the user trusts their WCS this step can be ignored as Astrometry.net is not used.

<h4>SFFT</h4>

AutoPhOT is now integrated with  the [saccadic fast Fourier transform [SFFT](https://iopscience.iop.org/article/10.3847/1538-4357/ac7394/meta) code. This is a novel template subtraction code that (in testing) is quite reliable

SFFTd will need to be installed, see the [SFFT github](https://github.com/thomasvrussell/sfft) for details

<h4>ZOGY</h4>

AutoPhOT can also use [Zogy](https://arxiv.org/abs/1601.02655) which has a python wrapper and can be found [here](https://github.com/dguevel/PyZOGY). We can install this straight from Github. Make sure the correct environment is activated. we can clone the Github repository to the current directory:

```bash
git clone https://github.com/dguevel/PyZOGY
```

and we can install it by running:
```bash
cd PyZOGY/
python setup.py install
```

no further action is required.


<h4>HOTPANTS</h4>

AutoPhOT can use [HOTPANTS](http://www.ascl.net/1504.004) by Andy Becker which can be found [here](https://github.com/acbecker/hotpants).

We can download the HotPants code from Github using:

```bash
git clone https://github.com/acbecker/hotpants
cd hotpants\
```
Next we need to modify the *Makefile*. HOTPANTS requires *CFITSIO* to be already installed. We can install this using [Homebrew](https://formulae.brew.sh/formula/cfitsio):

 ```bash
brew install cfitsio
 ```

Which will install the library in a directory similar to */usr/local/Cellar/cfitsio/4.0.0* (although maybe not exactly). In this directory there should be two folders, *include* and *bin*.

We need to update the *Makefile* for HOTPANTS to work correctly. This file can be opened using a text editor. We need to update the CFITSIOINCDIR and LIBDIR variables to point towards the *include* and *bin* directories respectively.

```
CFITSIOINCDIR=/usr/local/Cellar/cfitsio/4.0.0/include
LIBDIR=/usr/local/Cellar/cfitsio/4.0.0/lib
```

Finally we can compile the code by running the following in the *hotpants/* directory

```bash
make
```

Once installed, locate the *hotpants* executable and update 'hotpants_exe_loc' in autophot_input (see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb) for example) .

**Known error with installation of HOTPANTS**

There is a [known bug](https://github.com/acbecker/hotpants/issues/4) with the HOTPANTS installation on MacOS - if upon installation you get 'malloc.h' file not found, replace:

```c
#include <malloc.h>
```
with
 ```c
 #if !defined(  MACH  )
 #include <malloc.h>
 #endif
 #if defined(  MACH  )
 #include <stdlib.h>
 #endif
```
to every .c file. Then you can run the *make* command.

If the user has no need for image subtraction or wants to use Zogy only, this step can be ignored.


<h4>ZOGY</h4>

AutoPhOT can also use [Zogy](https://arxiv.org/abs/1601.02655) which has a python wrapper and can be found [here](https://github.com/dguevel/PyZOGY). We can install this straight from Github. Make sure the correct environment is activated. we can clone the Github repository to the current directory:

```bash
git clone https://github.com/dguevel/PyZOGY
```

and we can install it by running:
```bash
cd PyZOGY/
python setup.py install
```

no further action is required.


## Referencing & Attribution

If you use results from AutoPhOT in a publication, please cite the follows

```
@ARTICLE{Brennan2022,
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

**All shared data will be kept confidential**.
