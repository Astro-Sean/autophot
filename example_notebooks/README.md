<p align="center">
  <img src=https://github.com/Astro-Sean/autophot/blob/master/logo.png>
</p>

# Example Notebooks for using AutoPhOT

We provide several Jupyter Notebooks on how to use AutoPHoT. If there is an aspect of AutoPhOT that is confusing, feel free to email me ([Seán Brennan](mailto:sean.brennan2@ucdconnect.ie?subject=AutoPhOT Notebooks)) or open an issue [here](https://github.com/Astro-Sean/autophot/issues).

## Basic Operations
For a notebook example on the basic operations of AutoPHoT, click  [here](https://github.com/Astro-Sean/autophot/blob/master/example_notebooks/basic_example.ipynb)

This notebook demonstrates:
  * How to load in and execute AutoPHoT on an example dataset
  * Perform photometry on a target using it's RA and Dec.

## Using AutoPHoT with Template subtraction

AutoPhOT is integrated with several popular template subtraction packages, click [here](https://github.com/Astro-Sean/autophot/blob/master/example_notebooks/template_subtraction_and_WCS.ipynb)!

This notebook demonstrates:

* Updating AutoPhOT to correct WCS values using Astrometry.Net
* Correct directory structure to template images.
* How to setup AutoPHoT to find the necessary template subtraction pipelines.
* Perform several preprocessing steps on the template images

## Telescope.yml

For insights on how AutoPhOT saves telescope and instrument information, click [here](https://github.com/Astro-Sean/autophot/blob/master/example_notebooks/example_call_database.ipynb)!

This notebook demonstrates:
* The overall structure of the *telescope.yml* file
* Check new images for correct header keywords


## Using your own catalog of sequence stars

For a example of how to use your own set of sequence stars for photometric calibration, click [here](https://github.com/Astro-Sean/autophot/blob/master/example_notebooks/add_your_catalog_example.ipynb)!

This notebook demonstrates:

* How to update AutoPhOT to accept your own add_your_catalog_example
* Check that the given catalog file is in the correct structure.
