<p align="center">
  <img src=https://github.com/Astro-Sean/autophot/blob/master/logo.png>
</p>
<div align="center">

[![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/AutoPhOT ) [![Anaconda-Server Badge](https://anaconda.org/astro-sean/autophot/badges/installer/conda.svg)](https://conda.anaconda.org/astro-sean)

</div>

:warning: **Project under new development**: The code is currently being rewritten to improve performance, reliability, accuracy and usability. Please be patient and inform [me](mailto:sean.brennan@astro.su.se?subject=New feature in AutoPhOT) of any new features you would like to see availale in the code.
Seán


## Introduction


AutoPhOT is novel automated pipeline that is designed for rapid, publication-quality photometry of transients.

The pipeline is built from the ground up using Python 3 - with no dependencies on legacy software. Capabilities of AutoPhOT include aperture and PSF-fitting photometry, template subtraction, and calculation of limiting magnitudes through artificial source injection. AutoPhOT is also capable of calibrating photometry against either survey catalogs (e.g. SDSS, PanSTARRS), or using a custom set of local photometric standards.

You can find the AutoPhOT paper [here](https://arxiv.org/abs/2201.02635). Feedback is welcome. Email [me](https://github.com/Astro-Sean) at [sean.brennan2@ucdconnect.ie](mailto:sean.brennan@astro.su.se?subject=AutoPhOT)

<p align="center">
  <a href="https://www.buymeacoffee.com/astrosean" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>


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

Install AutoPhOT using the conda install:

```bash
conda install -c astro-sean autophot
```

If you want to update AutoPhOT, you can do so using:

```bash
conda update -c astro-sean autophot
```
## Additional functionality

To fully utilise the AutoPhOT Code, several additional softwares may be used.

<h4>Astrometry.Net</h4>

AutoPhOT relies on [Astrometry.net](https://arxiv.org/abs/0910.2233) by Dustin Lang to solve for WCS. While the code can be downloaded/installed [here](http://astrometry.net/doc/readme.html) and [here](http://astrometry.net/doc/build.html#build.) we suggest using [Homebrew](https://brew.sh/) to install Astometry.net.

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


## How to use AutoPhOT

I've designed AutoPhOT to be a user friendly pipeline, meaning you need very little prior knowledge on the art of photometry.

&#8594; I've put together some *Jupyter* notebooks the get you started with AutoPhOT, click [here](https://github.com/Astro-Sean/autophot/tree/master/example_notebooks).

&#8594; If you wish to use the packages within AutoPhOT outside of the examples given, documentation on each package is given [here](https://autophot.readthedocs.io/en/latest/).

&#8594; A list of settings used in AutoPhOT can be found [here](https://autophot.readthedocs.io/en/latest/instructions.html).

&#8594; If you need an example of how to use specific functions in AutoPhOT, please open an issue [here](https://github.com/Astro-Sean/autophot/issues).

## Referencing & Attribution

If you use results from AutoPhOT in a publication, please cite [Brennan & Fraser (2022)](https://arxiv.org/abs/2201.02635). The AutoPhOT code is released under a GPL3 licence and you are free to reuse the code as you wish. If you modify AutoPhOT or use it in a strange fashion (or even if you use it normally), we make no guarantee that your photometry will be valid.

## Testing and Debugging

If you experience errors with a particular file, the most effective means of debugging is to share the file with me ([Seán Brennan](mailto:sean.brennan2@ucdconnect.ie?subject=AutoPhOT)) for diagnostic. Once bugs have been addressed all files will be deleted.

**All shared data will be kept confidential**.
