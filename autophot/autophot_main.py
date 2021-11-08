#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

try:
    import autophot
    # print('test')
    # print('AutoPhOT Version: %s' % autophot.__version__ )
except ModuleNotFoundError:
    sys.stderr.write('\nAutoPhot Package not installed!\n\n')
    sys.stderr.write('Download the latest version by running:\n\nconda install -c astro-sean autophot\n\nin Python3 enviroment\n')
    sys.exit()

# Check user is running Python3, if not exit
if sys.version_info<(3,0,0):
  from platform import python_version
  sys.stderr.write("\nYou need Python3 or later to run AutoPHoT\n")
  sys.stderr.write("Your Version: %s\n" % python_version())
  sys.exit()


def run_automatic_autophot(autophot_input):
    '''
    
    Main package to run AutoPhoT Pipeline
    
    :param autophot_input: Input Dictionary file
    :type autophot_input: dict
    :return: Produces calibration transient phototmetry in Output folder  
    :rtype: N\A

    '''
    
    print(r"""
        _       _       ___ _  _    _____
       /_\ _  _| |_ ___| _ \ || |__|_   _|
      / _ \ || |  _/ _ \  _/ __ / _ \| |
     /_/ \_\_,_|\__\___/_| |_||_\___/|_|
    
     ---------------------------------------
        Automated Photometry of Transients
        S. J. Brennan et al. 2021 
        Please provide feedback/bugs to:
        Email: sean.brennan2@ucdconnect.ie
    ---------------------------------------""")
    
    # AutoPHoT Specific packages
    from autophot.packages.recover_output import recover
    from autophot.packages.run import run_autophot

    start = time.time()
    
    #  Run AutoPhOT with instructurions given by autophot_input dictionary
    if autophot_input['fits_dir']:
        
        print('Directory of fits file: %s'  % autophot_input['fits_dir'] )
        
    elif autophot_input['fname']:
        
        print('Working on: %s'  % autophot_input['fname'] )
    else:
        print('No files found: update fits_dir or fname')
        return False

    # Run complete autophot package for automatic photometric reduction
    run_autophot(autophot_input)

    
    # Go through output filepath and see what has already been done and produce a human readbale output
    recover(autophot_input)

    print('\nDone - Time Taken: %.1f' %  float(time.time() - start))
