<p align="center">
  <img src=https://github.com/Astro-Sean/autophot/blob/master/logo.png >
</p>
<div align="center">

[![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot ) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/installer/conda.svg)](https://conda.anaconda.org/astro-sean)

</div>

## Introduction

The Automated Photometry Of Transients (AutoPhOT) pipeline allows for rapid, automatic analysis of fits images for transient events.

The novel pipeline is built from the ground up, based on Python3 and makes extensive use of Astropy and Numpy packages. No instance of IRAF or Python2 software is used. AutoPhOT is able to handle homogenised data from different telescopes and applies techniques such as image calibration, image subtraction, and novel PSF fitting in an automated and intelligent way.

**Project is undergoing active development. AutoPhoT will include the Python executable code presented here, an interactive website with an accompanying API. Documentation is currently being written. Feedback is welcome.
email: sean.brennan2@ucdconnect.ie**

<p align="center">
  <a href="https://www.buymeacoffee.com/astrosean" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>

## Developer notes

 * Colour terms still in testing/development and **not** in current build.
 * Currently no airmass correction.
 * Astrometry.net and HOTPANTS needs to be installed by User (see below for details).
 * PSF selection will look for bright isolated sources, however this can lead to sources being selected near the image boundaries which can cause large errors, especially in IR bands.
 * Telescope header script can fail on some image types - If fits image has TELESCOP and INSTRUME it should execute okay.

 * Image subtraction is somewhat crude - pipeline can produce clean subtractions reliably but we suggest users to check all subtractions.



## Installation

* Some packages require conda-forge in channel list:

```bash
conda config --add channels conda-forge.
```
as well as the astropy channel:

```bash
conda config --add channels astropy
```

* Install AutoPhOT via conda istall:

```bash
conda install -c astro-sean autophot
```

* Image alignment can use [astroalign](https://www.sciencedirect.com/science/article/pii/S221313372030038X) over WCS alignment from Astropy (using [reproject_interp](https://reproject.readthedocs.io/en/stable/api/reproject.reproject_interp.html)). Install via

```bash
pip install astroalign
```

* Code relies on [Astrometry.net](https://arxiv.org/abs/0910.2233) by Dustin Lang to solve for WCS. Code can be downloaded/installed [here](http://astrometry.net/doc/readme.html) and [here](http://astrometry.net/doc/build.html#build.).

Once installed, locate the solve-field executable [default location: /usr/local/astrometry/bin/solve-field] and update (if needed) 'solve_field_exe_loc' in syntax (see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb)).

**If the user trusts their WCS this step can be ignored as Astrometry.net is not used.**

* Image subtraction uses a local instance of [HOTPANTS](http://www.ascl.net/1504.004) by Andy Becker - HOTPANTS can be found [here](https://github.com/acbecker/). Once installed, locate the hotpants executable and update 'hotpants_exe_loc' in syntax see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb).

**If the user has no need for image subtraction this step can be ignored.**

**Known error with installation of HOTPANTS**

if installing on MacOS - if upon installation you get 'malloc.h' file not found, replace

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
to every .c file.

## Usage

* Check out my Jupyter Notebooks the get started with AutoPhOT [here](https://github.com/Astro-Sean/autophot/tree/master/example_notebooks)

## Referencing

* As this code is very much still under development and currently being developed and tested, you should be very cautious before using the results from AutoPhOT in a publication. This code is completely available to the community but please speak to me first.

* The AutoPhOT paper is currently being written, for the time being please cite this [paper](https://arxiv.org/abs/2102.09572).

## Testing and Debugging

* If you experience errors with a particular file, the most effective means of debug is to share the file with a developer for diagnostic. Once bugs have been addressed all files will be deleted. **All shared data will be kept confidential**.



<p align="center">
  <img src="https://media.giphy.com/media/ehkn6JLlzuayjGarJX/giphy.gif"  />
</p>
<div align="center">

