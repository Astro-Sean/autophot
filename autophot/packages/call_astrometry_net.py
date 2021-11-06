#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:24:59 2018

@author: seanbrennan
"""

'''
- Inputs:
    File: path to file
    syntax: dictionary use for settings


- Output:
    Filename of new, downloaded fits file with updated WCS

'''
def AstrometryNetLOCAL(file, syntax = None):

    import subprocess
    import os
    import shutil
    import numpy as np
    from autophot.packages.functions import getheader
    import signal
    import time
    import logging

    try:

        logger = logging.getLogger(__name__)

        #Open file and get header information
        headinfo = getheader(file)

        if syntax == None:
            logger.critical('Astrommetry needs synatx Yaml file')
            exit()

        # get filename from filepath used to name new WCS fits file contained WCS header with values
        base = os.path.basename(file)
        base = os.path.splitext(base)[0]

        parent_dir = os.path.dirname(file)

        # new of output.fits file
        wcs_file = base + "_WCS.fits"


        # location of executable [/Users/seanbrennan/AutoPhot_Development/AutoPHoT/astrometry.net/bin/solve-field]
        exe = syntax['solve_field_exe_loc']

        # Create input for subprocess:



        # Guess image scale if f.o.v is not known
        if syntax['scale_type'] == None or syntax['try_guess_wcs']:
            scale = [(str("--guess-scale"))]

        elif syntax['guess_scale']:
            scale = [(str("--guess-scale"))]

        else:
            scale = [("--scale-units=" ,str(syntax['scale_type'])),
                    ("--scale-low="    , str(syntax['scale_low'])),
                    ("--scale-high="   , str(syntax['scale_high']))]


        if not syntax['ignore_pointing']:

            if syntax['target_name'] != None or syntax['target_ra'] != None and syntax['target_dec'] != None:
                try:

                    tar_args = [("--ra="     , str(syntax['target_ra'])), # target location on sky, used to narrow down cone search
                                ("--dec="    , str(syntax['target_dec'])),
                                ("--radius=" , str(syntax['search_radius'])) # radius of search around target for matching sources to index deafult 0.5 deg
                                ]
                    scale = scale + tar_args
                except:
                    pass

        # if 'NAXIS1' in headinfo:
        #     scale = scale+ [("--height="      , str(syntax['NAXIS1'])),
        #                     # ("--width="      ,  str(syntax['NAXIS2']))]
        # if not syntax['ignore_pointing']:







        include_args = [

            ('--no-remove-lines'),
            ('--uniformize=' , str(0)),
            	("--overwrite"),
            ("--downsample="  , str(syntax['downsample']) ),  # Downsample image - good for large images
            ("--new-fits="    , str(None)), # Don't download new fits file with updated wcs
            ("--cpulimit="   ,  str(syntax['solve_field_timeout'])), # set time limit on subprocess
            ("--wcs="         , str(wcs_file)), # filepath of wcs fits header file
            ("--index-xyls="  , str(None)),# don't need all these files
            ("--axy="         , str(None)),
            ("--scamp="       , str(None)),
            ("--corr="        , str(None)),
            ("--rdl="        ,  str(None)),
            ("--match="      ,  str(None)),
            ("--solved="     ,  str(None)),
            ("--height="      , str(syntax['NAXIS1'])), #set image height and width
            ("--width="      ,  str(syntax['NAXIS2'])),
            ("--no-plots"),
            ("--no-verify"),
            # ("--tweak-order"), str(2)
            ]

        # and scale commands
        include_args = include_args + scale

        # build command to run astrometry.net
        args= [str(exe) + ' ' + str(file) + ' ' ]
        for i in include_args:
            if isinstance(i,tuple):
                for j in i:
                    args[0] += j
            else:
                args[0] += i
            args[0] += ' '

        start = time.time()

        with open(syntax['write_dir'] + base + '_astrometry.log', 'w') as FNULL:

            '''
            Call processto run astronmetry.net using solve-field command and list of keywords
            given in args list or tuples

            print all output from astrometry to a .log file to help with any debugging
            '''

            pro = subprocess.Popen(args,
                                   shell=True,
                                   stdout=FNULL,
                                   stderr=FNULL,
                                   preexec_fn=os.setsid)

            # Timeout command - will only run command for this long - ~30s works fine
            pro.wait(syntax['solve_field_timeout'])

            try:
                # Try to kill process to avoid memory errors / hanging process
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            except:
                pass

        logger.info('ASTROMETRY finished: %ss' % round(time.time() - start) )

        # check if file is there - if so return filepath if not return nan
        if os.path.isfile(os.path.join(os.getcwd(), wcs_file)):

            # Move file into new file directory
            shutil.move(os.path.join(os.getcwd(), wcs_file),
                        os.path.join(parent_dir,  wcs_file))

            # astrometry creates this none file - delete it
            os.remove(os.path.join(os.getcwd(), 'None'))

            return  os.path.join(parent_dir, wcs_file)


        else:
            logger.warning("-> FILE CHECK FAILURE - Return NAN <-")
            logger.debug(args)

        return np.nan

    except Exception as e:
        logger.exception(e)
        return np.nan

