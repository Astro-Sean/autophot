#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:58:34 2022

@author: seanbrennan
"""

class background():
    
    def __init__(self, input_yaml,image):
        
        self.input_yaml = input_yaml
        self.image = image
        
        
    def remove(self,bkgLevel = 3,
               xc = None,yc = None,
               removeSurface= False,
               removeMedian = True,
               removePoly = False,
               removePoly_degree = 2):
        
        import warnings
        import numpy as np
        
        from astropy.stats import SigmaClip
        from astropy.stats import sigma_clipped_stats
        from astropy.modeling import models, fitting
        from astropy.stats import mad_std
        
        from photutils.aperture import CircularAperture,CircularAnnulus
        from photutils.background import (Background2D,
                               SExtractorBackground,
                               BkgZoomInterpolator,
                               MedianBackground)
        
        
        fwhm = self.input_yaml['fwhm']
        

        
        try:
            # assumed size (radius) of the source  = area to be masked
            source_size = 1.75*fwhm
            
            # box and filter size for  surface fitting
            box = int(np.ceil(fwhm))

            filter_size = int(np.ceil(fwhm)*2)
            
            if filter_size % 2 ==0: filter_size+=1
    
            if xc is None and yc is None:
                
                yc = int(self.image.shape[1]/2)
                xc = int(self.image.shape[0]/2)
                

            positions = [xc,yc]

            # mask array excluding center location
            aperture = CircularAperture(positions,r=source_size)
            masks = aperture.to_mask(method='center')
            mask_array = masks.to_image(shape=self.image.shape).astype(bool)
            
            
            annulus_aperture = CircularAnnulus(positions, r_in=source_size+2, r_out=source_size+3)
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_mask_array = annulus_masks.to_image(shape=self.image.shape).astype(bool)
         
            if removeSurface:
                
                sigma_clip = SigmaClip(sigma = bkgLevel,
                                       cenfunc = np.nanmedian,
                                       stdfunc = mad_std)
            
                
                # Fit surface background
                background = Background2D(self.image,
                                            box_size = box,
                                            mask = mask_array,
                                            filter_size = filter_size,
                                            sigma_clip = sigma_clip,
                                            bkg_estimator =  MedianBackground(),
                                            # exclude_percentile=5
                                            )
        
                # fitted surface
                surface = background.background
        
                # background free self.image
                image_background_free = self.image - surface
                
            elif removePoly and not removeMedian:
                
                # fitted 2D polynomial to surface
                surface_function_init = models.Polynomial2D(degree=removePoly_degree)
        
                fit_surface = fitting.LevMarLSQFitter()
        
                x = np.arange(0,self.image.shape[1])
                y = np.arange(0,self.image.shape[0])
                xx, yy = np.meshgrid(x,y)
        
                with warnings.catch_warnings():
                    # Ignore model linearity warning from the fitter
                    warnings.simplefilter('ignore')
                    
                    surface_fit = fit_surface(surface_function_init, xx, yy, self.image)
        
                surface = surface_fit(xx,yy)
        
                
                image_background_free = self.image - surface
                
            elif removeMedian:
                
                _, background_value, _ = sigma_clipped_stats(self.image,
                                                             sigma=bkgLevel,
                                                             mask = annulus_mask_array,
                                                             cenfunc = np.nanmedian,
                                                             stdfunc = mad_std
                                                   )
        
                surface = np.ones(self.image.shape) * background_value     
                
                image_background_free = self.image - surface
                
        
            # mask out target area in surface
            backgroundfree_image_outside_aperture = np.ma.array(image_background_free,
                                                                mask=~annulus_mask_array)
        
            
            # Get noise/STD of the surface, excluding the location of the target
            _, _, noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                             cenfunc = np.nanmean,
                                             stdfunc = np.nanstd,
                                             sigma = bkgLevel)
            
        
            
            # Find the mean value of the surface inside the target location
            surface_inside_aperture = np.ma.array(surface,mask=~mask_array)
            
            _, mean_surface, _= sigma_clipped_stats(surface_inside_aperture,
                                                      cenfunc = np.nanmean,
                                                      stdfunc = np.nanstd,
                                                      sigma = bkgLevel)
        
            
            return image_background_free, surface , mean_surface, noise
        
        except Exception as e:
            import sys,os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            
            
