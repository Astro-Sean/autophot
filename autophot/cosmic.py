#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:24:07 2022

@author: seanbrennan
"""

class remove_cosmic_rays:
    
    def __init__(self,input_yaml,fpath,image,header,use_lacosmic =  False):
        self.input_yaml = input_yaml
        self.fpath = fpath
        self.image = image
        self.header =header
        self.use_lacosmic = use_lacosmic 
        
        pass
    
    
    def remove(self,bkg = None,mask = None,satlevel = None,gain = 1):
        
        import logging
        import astroscrappy
        import warnings
        import numpy as np
        
        import gc
        from astropy.io import fits
        
        from functions import (border_msg)
        
        logger = logging.getLogger(__name__)
        
        image_copy = self.image.copy()
        
        
        # from ccdproc import cosmicray_lacosmic
        
        try:
            
            # if not (bkg is None):
            # image_copy = image_copy + bkg 
            if gain ==0:
                gain = 1
            

            logger.info(border_msg('Detecting/removing cosmic ray sources with Astroscrappy'))
            
            # Using this catch as then is a depracted wanring with astroscrappy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
    
                cray_free_image = astroscrappy.detect_cosmics(image_copy,
                                                               gain = gain,
                                                                sigclip=5,  # Higher to avoid artifacts
                                                               sigfrac=0.3,  # More conservative detection
                                                               objlim=3.0,   # Helps with bright objects
                                                               niter=1,
                                                             # satlevel = np.inf,
                                                                # satlevel=satlevel,
                                                                inbkg=bkg,
                                                               # verbose = True
                                                               # sepmed = False
                                                                # psfsize = 15,
                                                                # psffwhm=5,
                                                               # fsmode = 'convolve',
                                                              # sigclip=6, 
                                                              # sigfrac=0.3,
                                                             # readnoise = 1,
                                                             # cleantype = 'median',
                                                              # sepmed = False
                                                               )      # Prevent overcorrection

                                                    
            clean_image = cray_free_image[1]
            CR_mask = cray_free_image[0]
            
            logger.info(f'> Contaminated pixels cleaned: {np.sum(CR_mask)}')

            return clean_image
        
        
        except Exception as e:
            logger.info('Could not remove Cosmic Rays!\n->%s\m Returning original image' % e)
            logger.exception(e)
            return image_copy 
                
                
                
                
                