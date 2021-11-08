<p align="center">
  <img src=https://github.com/Astro-Sean/autophot/blob/master/logo.png>
</p>
<div align="center">

[![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot ) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/installer/conda.svg)](https://conda.anaconda.org/astro-sean)

</div>

## Introduction

The Automated Photometry Of Transients (AutoPhOT) pipeline allows for rapid, automatic analysis of fits images for transient events.

The novel pipeline is built from the ground up, based on Python3 and makes extensive use of Astropy and Numpy packages. No instance of IRAF or Python2 software is used. AutoPhOT is able to handle homogenised data from different telescopes and applies techniques such as image calibration, image subtraction, and novel PSF fitting in an automated and intelligent way.

Feedback is welcome. Email: sean.brennan2@ucdconnect.ie

<p align="center">
  <a href="https://www.buymeacoffee.com/astrosean" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>


## Installation

* We suggest creating a new enviroment for AutoPhOT. This can be done using conda by:

```bash
conda create -n autophot_env python=3.7
```

and then to activate this enviroment run:
```bash
conda activate autophot_env
```


* Install AutoPhOT via conda istall:

```bash
conda install -c astro-sean autophot
```

## Additional functionality

<h3>Astrometry.Net</h3>
To fully utilise the AutoPhot Code, several additional softwares may be used:

* Code relies on [Astrometry.net](https://arxiv.org/abs/0910.2233) by Dustin Lang to solve for WCS. Code can be downloaded/installed [here](http://astrometry.net/doc/readme.html) and [here](http://astrometry.net/doc/build.html#build.). Alternatively [Homebrew](https://brew.sh/) can be used to install Astometry.net.

```bash
brew install astrometry-net
```
In order for Astometry.net to run successfully, it require pre-index files for calibration. First we can create a new directory called "data".

```bash
mkdir data
cd data/
```

Next, while in the data directory we can run thoe following to download these files (~40Gb):

```bash
wget -r -np http://broiler.astrometry.net/~dstn/4200/
```

Once this download is complete this data folder must be correct location so that astrometry can find it.

```bash
which solve-feld
```
which will give the location of the "solve-field" command, it should be something similar to /usr/local/Cellar/astrometry-net/0.85_1. Finally we move this data folder to this directory:

```bash
cd ../
cp data /usr/local/Cellar/astrometry-net/0.85_1/.
```

To update AutoPhot on the location of Astrometry.Net,  update (if needed) 'solve_field_exe_loc' in the autophot_input dictionary (see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb) for example).

**If the user trusts their WCS this step can be ignored as Astrometry.net is not used.**

<h3>HOTPANTS</h3>
* AutoPhOT can use  [HOTPANTS](http://www.ascl.net/1504.004) by Andy Becker - HOTPANTS can be found [here](https://github.com/acbecker/). Once installed, locate the hotpants executable and update 'hotpants_exe_loc' in autophot_input (see [here](https://github.com/Astro-Sean/autophot/blob/master/autophot_example.ipynb) for example) .

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

<h3>ZOGY</h3>

* AutoPhOT can also use [Zogy](https://arxiv.org/abs/1601.02655) Which has a python wrapper which can be found [here](https://github.com/dguevel/PyZOGY). We can install this from Github. First we can clone the Github repository to the current directory:

```bash
git clone https://github.com/dguevel/PyZOGY
```

and we can install it by running:
```bash
cd PyZOGY
python setup.py install
```

no further action is required.


## Usage

* Check out my Jupyter Notebooks the get started with AutoPhOT [here](https://github.com/Astro-Sean/autophot/tree/master/example_notebooks)

## Referencing

* The AutoPhOT paper is currently being written, for the time being please cite this [paper](https://arxiv.org/abs/2102.09572).

## Testing and Debugging

* If you experience errors with a particular file, the most effective means of debug is to share the file with a developer for diagnostic. Once bugs have been addressed all files will be deleted. **All shared data will be kept confidential**.
