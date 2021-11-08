#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def AstrometryNetLOCAL(file,
                       NAXIS1,
                       NAXIS2,
                       solve_field_exe_loc = None,
                       pixel_scale = None,
                       try_guess_wcs = False,
                       ignore_pointing = False,
                       target_ra = None,
                       target_dec = None,
                       search_radius = 1,
                       downsample = 2,
                       cpulimit = 180,
                       solve_field_timeout = 180
                       ):
    '''
    
    :param file: DESCRIPTION
    :type file: TYPE
    :param NAXIS1: DESCRIPTION
    :type NAXIS1: TYPE
    :param NAXIS2: DESCRIPTION
    :type NAXIS2: TYPE
    :param solve_field_exe_loc: DESCRIPTION, defaults to None
    :type solve_field_exe_loc: TYPE, optional
    :param pixel_scale: DESCRIPTION, defaults to None
    :type pixel_scale: TYPE, optional
    :param try_guess_wcs: DESCRIPTION, defaults to False
    :type try_guess_wcs: TYPE, optional
    :param ignore_pointing: DESCRIPTION, defaults to False
    :type ignore_pointing: TYPE, optional
    :param target_ra: DESCRIPTION, defaults to None
    :type target_ra: TYPE, optional
    :param target_dec: DESCRIPTION, defaults to None
    :type target_dec: TYPE, optional
    :param search_radius: DESCRIPTION, defaults to 1
    :type search_radius: TYPE, optional
    :param downsample: DESCRIPTION, defaults to 2
    :type downsample: TYPE, optional
    :param cpulimit: DESCRIPTION, defaults to 180
    :type cpulimit: TYPE, optional
    :param solve_field_timeout: DESCRIPTION, defaults to 180
    :type solve_field_timeout: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import subprocess
    import os
    import shutil
    import numpy as np
    from autophot.packages.functions import getheader
    import signal
    import time
    import logging

    try:
        
        if solve_field_exe_loc is None:
            raise Exception('Please enter "solve_field_exe_loc" to solve for WCS')

        logger = logging.getLogger(__name__)

        #Open file and get header information
        headinfo = getheader(file)

        # get filename from filepath used to name new WCS fits file contained WCS header with values
        base = os.path.basename(file)
        base = os.path.splitext(base)[0]

        write_dir = os.path.dirname(file)

        # new of output.fits file
        wcs_file = base + ".wcs.fits"

        # location of executable [/Users/seanbrennan/AutoPhot_Development/AutoPHoT/astrometry.net/bin/solve-field]
        exe = solve_field_exe_loc

        # Guess image scale if f.o.v is not known
        if pixel_scale == None or try_guess_wcs:
            scale = [(str("--guess-scale"))]

        elif try_guess_wcs:
            scale = [(str("--guess-scale"))]

        else:
            scale = [("--scale-units=" , 'arcsecperpix'),
                    ("--scale-low="    , str(pixel_scale*0.25)),
                    ("--scale-high="   , str(pixel_scale*1.25))]


        if not ignore_pointing:

            if target_ra != None and target_dec != None:
                try:

                    tar_args = [("--ra="     , str(target_ra)), # target location on sky, used to narrow down cone search
                                ("--dec="    , str(target_dec)),
                                ("--radius=" , str(search_radius)) # radius of search around target for matching sources to index deafult 0.5 deg
                                ]
                    scale = scale + tar_args
                except:
                    pass




        include_args = [

            ('--no-remove-lines'),
            ('--uniformize=' , str(0)),
            ("--overwrite"),
            ("--downsample="  , str(downsample) ),  # Downsample image - good for large images
            ("--new-fits="    , str(None)), # Don't download new fits file with updated wcs
            ("--cpulimit="   ,  str(cpulimit)), # set time limit on subprocess
            ("--wcs="         , str(wcs_file)), # filepath of wcs fits header file
            ("--index-xyls="  , str(None)),# don't need all these files
            ("--axy="         , str(None)),
            ("--scamp="       , str(None)),

            ("--corr="        , str(None)),
            ("--rdl="        ,  str(None)),
            ("--match="      ,  str(None)),
            ("--solved="     ,  str(None)),
            ("--height="      , str(NAXIS1)), #set image height and width
            ("--width="      ,  str(NAXIS2)),
            ("--no-plots"),
            ("--no-verify")
 
            # ("--no-verify"),
            # ("--crpix-center"),
            # ("--no-tweak")
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
        
        astrometry_log_fpath = os.path.join(write_dir,base + '_astrometry.log')

        with open(astrometry_log_fpath, 'w') as FNULL:

            '''
            Call processto run astronmetry.net using solve-field command and list of keywords
            given in args list or tuples

            print all output from astrometry to a .log file to help with any debugging
            '''
            logger.info('ASTROMETRY started...' )
            pro = subprocess.Popen(args,
                                   shell=True,
                                   stdout=FNULL,
                                   stderr=FNULL,
                                   preexec_fn=os.setsid)

            # Timeout command - will only run command for this long - ~30s works fine
            try:
                pro.wait(solve_field_timeout)
            except:
                print(args)


                return np.nan


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
                        os.path.join(write_dir,  wcs_file))

            # astrometry creates this none file - delete it
            os.remove(os.path.join(os.getcwd(), 'None'))

            return  os.path.join(write_dir, wcs_file)


        else:
            logger.warning("-> FILE CHECK FAILURE - Return NAN <-")
            logger.debug(args)
            print(args)

        return np.nan

    except Exception as e:
        logger.exception(e)
        return np.nan

