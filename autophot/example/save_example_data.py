#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Load example data and save it
to users desktop by default

'''

import os
from pathlib import Path
from functools import reduce
from shutil import copyfile

def save_fits_to_desktop(new_path = os.path.join(str(Path.home()),'Desktop')):

    # parent directory
    filepath = os.path.dirname(os.path.abspath(__file__))

    # default name - don't fix
    example_fits_name = 'example.fits'

    # Location of example.fits
    example_fits_path = reduce(os.path.join,[filepath,example_fits_name])

    # create folder on dekstop called autophot example
    example_directory_new_path = os.path.join(new_path,'autophot_example')

    # create directory on dekstop if not already created
    os.makedirs(example_directory_new_path , exist_ok=True)

    # copy example.fits to desktop
    copyfile(example_fits_path,
             os.path.join(example_directory_new_path,example_fits_name))

    # Check that file is written to correct place
    if os.path.isfile(os.path.join(example_directory_new_path,example_fits_name)):
        print('Successful copy of %s written to:\n%s' % (example_fits_name,os.path.join(example_directory_new_path,example_fits_name)))

    return os.path.join(example_directory_new_path,example_fits_name)
