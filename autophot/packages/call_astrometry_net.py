#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def AstrometryNetLOCAL(file,
                       solve_field_exe_loc = None,
                       pixel_scale = None,
                       target_ra = None,
                       target_dec = None,
                       search_radius = 1,
                       downsample = 2,
                       cpulimit = 180
                       ):
    '''
    
    Call a locally installed version of `Astrometry.net
    <https://astrometry.net/use.html>`_ to redo World Coordinates system (WCS)
    values for a given image. Astrometry.net must be installed on your machine with
    the correct index files for package to work sucessfully. Instructions on how do
    do so may be found `here <https://github.com/Astro-Sean/autophot>`_.
    
    :param file: Filepath of *FITS* file which needs corrected WCS values
    :type file: str
    :param solve_field_exe_loc: Filepath location of the *solve-field* executable,
    see `here <https://github.com/Astro-Sean/autophot>`_ for instruction on how to
    find this path, defaults to None
    :type solve_field_exe_loc: str, optional
    :param pixel_scale: Pixel scale in :math:`arcseconds / pixel`. We recommend you
    include this variable as it can greatly speed up computation times and lower
    the risk of failures, defaults to None
    :type pixel_scale: float, optional
    :param target_ra: Right ascension (RA) of target in degrees i.e. where are we
    looking on the sky. Telling Astrometry approximately where it is looking can
    greatly speed up computation times and lower the risk of failures , defaults to
    None
    :type target_ra: float, optional
    :param target_dec: Declination ascension (RA) of target in degrees, defaults to
    None
    :type target_dec: float, optional
    :param search_radius: Radius in degrees around target RA and Dec to search for
    correct location., defaults to 1
    :type search_radius: TYPE, optional
    :param downsample: For larger images it may be useful to downsample them to
    speed up computation times, defaults to 2
    :type downsample: int, optional
    :param cpulimit: Execution time limit in seconds for the local version of
    Astrometry.net. If the code run time exceeds this value, the script will exit,
    defaults to 180
    :type cpulimit: float, optional
    :raises Exception: If no output WCS file is created an excepted is raise
    :return: Returns the filepath of the WCS file created from the image saved to
    the same parent directory as the image.
    :rtype: str

    '''

    

    import subprocess
    import os
    import numpy as np
    from autophot.packages.functions import getheader,getimage
    import signal
    import time
    import logging
    
  
    try:
 
        logger = logging.getLogger(__name__)

        #Open file and get header information
        headinfo = getheader(file)

        # get filename from filepath used to name new WCS fits file contained WCS header with values
        base = os.path.basename(file)
        base = os.path.splitext(base)[0]
      

        dirname = os.path.dirname(file)

        # new of output.fits file
        wcs_file = os.path.join(dirname,base.replace('sources_','') + ".wcs.fits")

        # location of executable [/Users/seanbrennan/AutoPhot_Development/AutoPHoT/astrometry.net/bin/solve-field]
        exe = str(solve_field_exe_loc)
        
        
        if exe == str(None):
            raise Exception('Please enter "solve_field_exe_loc" to solve for WCS')
            # return np.nan


        # Guess image scale if f.o.v is not known
        if pixel_scale == None:
            scale = [(str("--guess-scale"))]

        else:
            scale = [("--scale-units=" , 'arcsecperpix'),
                    ("--scale-low="    , str(pixel_scale*0.25)),
                    ("--scale-high="   , str(pixel_scale*1.25))]

        if target_ra != None and target_dec != None:
            try:

                tar_args = [("--ra="     , str(target_ra)), # target location on sky, used to narrow down cone search
                            ("--dec="    , str(target_dec)),
                            ("--radius=" , str(search_radius)) # radius of search around target for matching sources to index deafult 0.5 deg
                            ]
                scale = scale + tar_args
                
            except:
                pass


        if 'NAXIS1' in headinfo and 'NAXIS' in headinfo:
            NAXIS1 = headinfo['NAXIS1']
            NAXIS2 = headinfo['NAXIS2']
        else:
            image = getimage(file)
            NAXIS1 = image.shape[0]
            NAXIS2 = image.shape[1]




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
        
        astrometry_log_fpath = os.path.join(dirname,base + '_astrometry.log')

        with open(astrometry_log_fpath, 'w') as FNULL:
 
            logger.info('ASTROMETRY started...' )
            pro = subprocess.Popen(args,
                                   shell=True,
                                   stdout=FNULL,
                                   stderr=FNULL,
                                   preexec_fn=os.setsid)

            # Timeout command - will only run command for this long - ~30s works fine
            try:
                pro.wait(cpulimit)
                
            except:
                
                return np.nan


            try:
                # Try to kill process to avoid memory errors / hanging process
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            except:
                pass

        logger.info('ASTROMETRY finished: %ss' % round(time.time() - start) )

        # check if file is there - if so return filepath if not return nan
        if os.path.isfile(wcs_file):

            # astrometry creates this none file - delete it
            os.remove(os.path.join(os.getcwd(), 'None'))

            return   wcs_file


        else:
            logger.warning("-> FILE CHECK FAILURE - Return NAN <-")
            logger.debug(args)
            print(args)

        return np.nan

    except Exception as e:
        logger.exception(e)
        return np.nan

