#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def remove_cosmic_rays(image_with_CRs,
                     gain = 1,
                     use_lacosmic = False):
    '''

    Function to remove Cosmic Rays from an image. Cosmic Rays (CRs) are high energy
    particles that impact the CCD detector and can result in bright points or
    streaks on the CCD image. For images with long exposure times, CRs can be
    problematic as they may lie on top of regions or sources of interest This
    fucntion can using either `Astroscrappy
    <https://astroscrappy.readthedocs.io/en/latest/#functions>`_ or `LACosmic
    <https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html>`_
    and returns an image cleaned of cosmic rays
    
    :param image_with_CRs: File Path to *fits* image that is contaminated by cosmic
    rays.
    :type image_with_CRs: str
    :param gain: Gain of the image (electrons / ADU). We always need to work in
    electrons for cosmic ray detection., defaults to 1
    :type gain: float, optional
    :param use_lacosmic: If True, use LAComic from CCDProc rather that
    astroscrappy, defaults to False
    :type use_lacosmic: boolean, optional
    :return: Returns an image that has been cleaned of cosmic rays
    :rtype: 2D array    

    '''


    import logging
    import astroscrappy
    import numpy as np

    try:

        logger = logging.getLogger(__name__)

        logger.info('Detecting/removing cosmic ray sources')

        if not use_lacosmic:
            
    

            print('Starting Astroscrappy ... ',end = '')

            cray_free_image = astroscrappy.detect_cosmics(image_with_CRs.data,sigclip=4.5, sigfrac=0.3,
                                                    objlim=5.0, gain=gain,
                                                    satlevel=65535.0, 
                                                    pssl=0.0, 
                                                    niter=4,
                                                    sepmed=True, 
                                                    cleantype='meanmask', 
                                                    fsmode='median',
                                                    psfmodel='gauss', 
                                                    psffwhm=2.5, 
                                                    psfsize=7,
                                                    psfk=None, 
                                                    psfbeta=4.765, 
                                                    verbose=False)
                                                    
            clean_image = cray_free_image[1]
            CR_mask = cray_free_image[0]

        else:

            from ccdproc import cosmicray_lacosmic


            clean_image,CR_mask = cosmicray_lacosmic(image_with_CRs.data,sigclip=4.5, 
                                                     sigfrac=0.3,
                                                     objlim=5.0, gain=gain,
                                                     satlevel=65535.0, 
                                                     pssl=0.0, niter=4,
                                                     sepmed=True, 
                                                     cleantype='meanmask', 
                                                     fsmode='median',
                                                     psfmodel='gauss', 
                                                     psffwhm=2.5, 
                                                     psfsize=7,
                                                     psfk=None, 
                                                     psfbeta=4.765,
                                                     verbose=False)
            # end_time = time.time() -  cray_time
            
            
        logger.info('Contaminated pixels with Cosmic rays removed: %d' % np.sum(CR_mask))


        return clean_image


    except Exception as e:
        logger.info('Could not remove Cosmic Rays!\n->%s\m Returning original image' % e)
        logger.exception(e)
        return image_with_CRs