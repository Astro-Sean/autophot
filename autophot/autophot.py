#!/usr/bin/env python3
# -*- coding: utf-8 -*-


print(r"""
    _       _       ___ _  _    _____
   /_\ _  _| |_ ___| _ \ || |__|_   _|
  / _ \ || |  _/ _ \  _/ __ / _ \| |
 /_/ \_\_,_|\__\___/_| |_||_\___/|_|

 ---------------------------------------
         AutoPhOT Beta
         S.J.Brennan et al. 2020 in prep
         Please provide feedback/bugs to:
         Email: sean.brennan2@ucdconnect.ie
---------------------------------------


""")

import sys

try:
    import autophot
    print('AutoPhOT Version: %s' % autophot.__version__ )
except ModuleNotFoundError:
    sys.stderr.write('\nAutoPhot Package not installed!\n\n')
    sys.stderr.write('Download the latest version by running:\n\nconda install -c astro-sean autophot\n\nin Python3 enviroment\n')
    sys.exit()



# Check user is running Python3 if not exit
if sys.version_info<(3,0,0):
  from platform import python_version
  sys.stderr.write("\nYou need Python3 or later to run AutoPHoT\n")
  sys.stderr.write("Your Version: %s\n" % python_version())
  sys.exit()





# Standard packages
import time
from urllib.request import urlopen
import matplotlib.pyplot as plt



# Turn off plots so nothing is shown
plt.ioff()


# Check internet connection
def internet_on():
    print('Checking internet connection...')
    try:
        urlopen('https://www.google.com/', timeout=10)
        return True
    except:
        return False

if not internet_on():
    print('Not connected to internet - some packages may not work\n')
else:
    print('Connected to internet\n')


# Run autophot
def run_automatic_autophot(syntax):

    # AutoPHoT Specific packages
    from autophot.packages.recover_output import recover
    from autophot.packages.run import run_autophot


    #  Run AutoPhOT with instructurions given by syntax dictionary
    start = time.time()

    if syntax['fits_dir']:
        print('Directory of fits file: %s'  % syntax['fits_dir'] )
    else:
        print('Working on: %s'  % syntax['fname'] )


    # Run complete autophot package for automatic photometric reduction
    run_autophot(syntax)

    '''
    Go through output filepath and see what has already
    been done and produce a human readbale output
    '''
    recover(syntax)

    print('\nDone - Time Taken: %.1f' %  float(time.time() - start))
