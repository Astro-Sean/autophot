#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def run_astroscrappy(image_with_CRs,
                     gain = 1,
                     use_astroscrappy = True,
                     use_lacosmic = False):
    '''
    
    :param image_with_CRs: DESCRIPTION
    :type image_with_CRs: TYPE
    :param gain: DESCRIPTION, defaults to 1
    :type gain: TYPE, optional
    :param use_astroscrappy: DESCRIPTION, defaults to True
    :type use_astroscrappy: TYPE, optional
    :param use_lacosmic: DESCRIPTION, defaults to False
    :type use_lacosmic: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import time
    import logging
    import astroscrappy
    import numpy as np

    try:

        logger = logging.getLogger(__name__)

        logger.info('Detecting/removing cosmic ray sources')

        if use_astroscrappy:
            
            # Function to call a instance of astroscrappy by Curtis McCully

            # link: https://astroscrappy.readthedocs.io/en/latest/#functions

            print('Starting Astroscrappy ... ',end = '')

            # Note start time
            cray_time = time.time()


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

        elif use_lacosmic:

            from ccdproc import cosmicray_lacosmic
            cray_time = time.time()

            clean_image,CR_mask = cosmicray_lacosmic(image_with_CRs.data,sigclip=4.5, sigfrac=0.3,
                                                     objlim=5.0, gain=gain,
                                                     satlevel=65535.0, pssl=0.0, niter=4,
                                                     sepmed=True, cleantype='meanmask', fsmode='median',
                                                     psfmodel='gauss', psffwhm=2.5, psfsize=7,
                                                     psfk=None, psfbeta=4.765, verbose=False)
            end_time = time.time() -  cray_time
            
            
        logger.info('Contaminated pixels with Cosmic rays removed: %d' % np.sum(CR_mask))


        return clean_image


    except Exception as e:
        logger.info('Could not remove Cosmic Rays!\n->%s\m Returning original image' % e)
        logger.exception(e)
        return image_with_CRs