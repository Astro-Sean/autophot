#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:03:09 2022

@author: seanbrennan
"""

class limits():

    def __init__(self,input_yaml):
        self.input_yaml = input_yaml
        
        
    def getCutout(self,image):
        
        import numpy as np
        
        target_x_pix=self.input_yaml['target_x_pix']
        target_y_pix=self.input_yaml['target_y_pix']
        
        expand_scale = int(np.ceil((self.input_yaml['limiting_magnitude']['inject_source_location'] * self.input_yaml['fwhm']) + self.input_yaml['scale']) + 3*self.input_yaml['fwhm'])
       
        close_up_expand = image[int(target_y_pix - expand_scale): int(target_y_pix + expand_scale),
                                int(target_x_pix - expand_scale): int(target_x_pix + expand_scale)]
        
        return close_up_expand 
        
    def getProbableLimit(self,cutout,
                          bkg_level = 3,
                          detection_limit=3,
                          useBeta = True,
                          beta =0.5,
                          plot = True,
                          unityPSF = None,
                          residualTable = None,
                          functionParams = None):
        
        import logging
        import warnings
        import numpy as np
        import random,os
        from photutils.detection import DAOStarFinder
        from background import background
        from photutils.aperture import CircularAperture
        from scipy.optimize import curve_fit
        
        from functions import gauss_1d
        from functions import beta_value,fluxUpperlimit
        from functions import mag
        
        logger = logging.getLogger(__name__)
        
        logger.info('Finding probable magnitude upper limit')
        try:
            fpath = self.input_yaml['fpath']
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            base = os.path.splitext(base)[0]
            
            
            # Source is good if we get here
            cutout, bkg_surface, bkg_median, bkg_std = background(input_yaml=self.input_yaml, 
                                                                                image = cutout).remove()
            
            daofind = DAOStarFinder(fwhm = self.input_yaml['fwhm'],
                                    threshold = bkg_level*bkg_std,
                                    sharplo   =  0.2,sharphi = 1.0,
                                    roundlo   = -1.0,roundhi = 1.0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore no sources warning
                sources = daofind(cutout)
                
            positions = []  
            if not (sources is None) :
    
                positions = list(zip(np.array(sources['xcentroid']),np.array(sources['ycentroid'])))
        
                    # Add center of image
            positions.append((cutout.shape[0]/2,cutout.shape[1]/2))
            
            
            # "size" of source, set to the aperture size
            source_size =  self.input_yaml['photometry']['ap_size']*self.input_yaml['fwhm']
            
            aperture_area = int(np.pi*source_size**2)
            
            # Mask out target region
            mask_ap  = CircularAperture(positions,r = source_size/2)
            mask = mask_ap.to_mask(method='center')
            mask_sumed = [i.to_image(cutout.shape) for i in mask]
            
            if len(mask_sumed) !=1:
                mask_sumed = sum(mask_sumed)
            else:
                mask_sumed = mask_sumed[0]
    
            #  Get measureable pixels
            mask_sumed[mask_sumed>0] = 1
            
            # Mask out center region
            mask_image  = (cutout) * (1-mask_sumed)
            
            # get pixels that are excluded and included
            excluded_points = np.where(mask_image >= 0.)
    
            exclud_x = excluded_points[1]
            exclud_y = excluded_points[0]
    
            exclud_zip = list(zip(exclud_x,exclud_y))
    
            included_points = np.where(mask_image != 0)
    
            includ_x = list(included_points[1])
            includ_y = list(included_points[0])
    
            includ_zip = list(zip(includ_x,includ_y))
    
            number_of_points = 150
       
            fake_points = {}
    
            # Failsafe - if there isn't enough pixels just use everything
            if len(includ_zip) < aperture_area:
            
                includ_zip=includ_zip+exclud_zip
            
            # get random sample of pixels and sum them up
            for i in range(number_of_points):
            
                fake_points[i] = []
            
                random_pixels = random.sample(includ_zip,aperture_area)
            
                xp_ran = [i[0] for i in random_pixels]
                yp_ran = [i[1] for i in random_pixels]
            
                fake_points[i].append([xp_ran,yp_ran])
            
            fake_sum = {}
            
            for i in range(number_of_points):
                fake_sum[i] = []
                for j in fake_points[i]:
                    for k in range(len(j[0])):
                        fake_sum[i].append(cutout[j[0][k]][j[1][k]])
            
            
            fake_mags = {}
            for f in fake_sum.keys():
                fake_mags[f] = np.nansum(fake_sum[f])
                
                
            hist, bins = np.histogram(list(fake_mags.values()),
                                      bins = len(list(fake_mags.values())),
                                      density = True)
    
            center = (bins[:-1] + bins[1:]) / 2
    
            sigma_guess = np.nanstd(list(fake_mags.values()))
            mean_guess = np.nanmean(list(fake_mags.values()))
            A_guess = np.nanmax(hist)
    
            popt,pcov = curve_fit(gauss_1d,center,hist,
                                  p0=[A_guess,mean_guess,sigma_guess],
                                  absolute_sigma = True )
    
            mean = popt[1]
            std  = abs(popt[2])
            
            if useBeta:
    
                CountsUpperlimit = fluxUpperlimit(n = detection_limit, beta_p = beta, sigma = std)
    
            else:
                
                CountsUpperlimit = detection_limit*std
                
                
            fluxUpperlimit = CountsUpperlimit/self.input_yaml['exptime']
            magUpperlimit = mag(fluxUpperlimit)

            logger.info('Probable magnitude upper limit: %.3f' % magUpperlimit)
                 
            injectFakePSF = True
                
            # if unityPSF is None or residualTable is None :
            #     injectFakePSF = False
            #     sigma = self.input_yaml['fwhm'] / 2*np.sqrt(2*np.log(2))
            #     functionParams = {'sigma':sigma}
            #     residualTable = np.zeros([int(2*self.input_yaml['scale']),int(2*self.input_yaml['scale'])])
            #     mag2image = lambda x: (10**(x/-2.5)) * self.input_yaml['exptime'] /(2*np.pi*sigma**2)
            # else:
            #     mag2image = lambda x: self.input_yaml['exptime']/unityPSF * 10**(x/-2.5)
                
            # # from psf import psfModel
  
            # base_psfModel = psfModel(input_yaml=self.input_yaml,
            #                           residualTable = residualTable,
            #                           functionParams = functionParams)
            # from functions import PointsInCircum
            # random_sources = PointsInCircum(self.input_yaml['limiting_magnitude']['inject_source_location']*self.input_yaml['fwhm'],
            #                                 shape = cutout.shape,
            #                                 n = 4)
                
            # xran = [abs(i[0]) for i in random_sources]
            # yran = [abs(i[1]) for i in random_sources]
            
            # if plot:
                
            #     from functions import set_size
            #     import matplotlib.pyplot as plt
            #     from matplotlib.gridspec import  GridSpec
            #     from astropy.visualization import (ImageNormalize,LinearStretch,ZScaleInterval)
            #     from mpl_toolkits.axes_grid1 import make_axes_locatable
            #     import os
                              
            #     dir_path = os.path.dirname(os.path.realpath(__file__))
            #     plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
                
                
            #     ncols = 2
            #     nrows = 2
                
            #     heights = [0.75,1]
            #     gs = GridSpec(nrows, ncols ,
            #                   wspace=0.2 ,
            #                   hspace=0.4,
            #                   height_ratios=heights,
            #                     # width_ratios = widths
            #                     )
            #     plt.ioff()
            #     fig = plt.figure(figsize = set_size(330,aspect = 1.52))
                
            #     ax0 = fig.add_subplot(gs[0, :])
            #     ax1 = fig.add_subplot(gs[1, 0])
            #     ax2 = fig.add_subplot(gs[1, 1])
                
            #     ax1.scatter(exclud_x,exclud_y,
            #                 color ='red',
            #                 marker = 'x',
            #                 alpha = 0.1,
            #                 label = 'Encluded areas',
            #                 zorder = 2)
        
            #     # the histogram of the data
            #     n, bins, patches = ax0.hist(list(fake_mags.values()),
            #                                 density=True,
            #                                 bins = 'auto',
            #                                 facecolor='blue',
            #                                 histtype = 'step',
            #                                 align = 'mid',
            #                                 alpha=1,
            #                                 label = 'Pseudo-Counts\nDistribution')
                
            #     line_kwargs = dict(ymin = 0,ymax = 0.75, alpha=0.5,color='black',ls = '--')
        
            #     ax0.axvline(mean, alpha=0.5,color='black',ls = '--')
                
            #     ax0.axvline(mean + 1*std,**line_kwargs)
            #     ax0.text(mean + 1*std,np.max(n),r'$1\sigma_{bkg}$',
            #               rotation = -90,va = 'top',ha = 'center')
                
            #     ax0.axvline(mean + 2*std,**line_kwargs)
            #     ax0.text(mean + 2*std,np.max(n),r'$2\sigma_{bkg}$',
            #               rotation = -90,va = 'top',ha = 'center')
        
            #     ax0.axvline(mean + detection_limit*std,**line_kwargs)
            #     ax0.text(mean + detection_limit*std,np.max(n),r'$'+str(detection_limit)+r'\sigma_{bkg}$',
            #               rotation = -90,va = 'top',ha = 'center')
                
            #     if useBeta:
            #         ax0.axvline(mean+CountsUpperlimit,ymin = 0,ymax = 0.6, alpha=0.5,
            #                     color='black',ls = '--')
            #         ax0.text(mean+CountsUpperlimit,np.max(n),r'$F_{UL,\beta=%.2f}$ '%beta,
            #                   rotation = -90,va = 'top',ha = 'center')
                
        
        
            #     x_fit = np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 250)
               
            #     ax0.plot(x_fit, gauss_1d(x_fit,*popt),
            #               label = 'Gaussian Fit',
            #               color = 'red')
        
        
            #     ax0.set_xlabel('Pseudo-Counts [counts]')
            #     ax0.set_ylabel('Probability Distribution')
                
            #     # vmin = np.percentile(cutout, 0.5)
            #     # vmax = np.percentile(cutout, 99.5)
                

                
            #     fake_sources = np.zeros(cutout.shape)
            #     for i in range(4):
            #         faketarget_i = base_psfModel.getPSF(xran[i],yran[i],mag2image(magUpperlimit),0,padShape=fake_sources)
            #         fake_sources+=faketarget_i
                    
            #         ax2.scatter(xran[i],yran[i],marker = 'o',fc = 'none',s = 25,alpha = 0.3,color = 'red')
                    
                    
                    
            #     cutout_with_sources = cutout + fake_sources
                
            #     cutout_trim = cutout[int(0.5*cutout.shape[1]-self.input_yaml['scale']):int(0.5*cutout.shape[1]+self.input_yaml['scale']),
            #                           int(0.5*cutout.shape[0]-self.input_yaml['scale']):int(0.5*cutout.shape[0]+self.input_yaml['scale'])
            #                         ]
                
                
            #     cutout_with_sources_trim = cutout_with_sources[int(0.5*cutout.shape[1]-self.input_yaml['scale']):int(0.5*cutout.shape[1]+self.input_yaml['scale']),
            #                           int(0.5*cutout.shape[0]-self.input_yaml['scale']):int(0.5*cutout.shape[0]+self.input_yaml['scale'])
            #                         ]
                
        
            #     im1 = ax1.imshow(cutout,origin='lower',
            #                       aspect = 'auto',
            #                       interpolation = 'nearest',
            #                       vmin = np.percentile(cutout_trim, 0.5),
            #                       vmax = np.percentile(cutout_trim, 99.5)
            #                       )
                    
            #     im2 = ax2.imshow(cutout_with_sources,
            #                       aspect = 'auto',
            #                       origin = 'lower',
            #                       interpolation = 'nearest',
            #                       vmin = np.percentile(cutout_trim, 0.5),
            #                       vmax = np.percentile(cutout_trim, 99.5)
            #                       )
                
                
            #     for ax in [ax1,ax2]:
            #         ax.set_xlim(0.5*cutout.shape[1]-self.input_yaml['scale'],0.5*cutout.shape[1]+self.input_yaml['scale'])
                
            #         ax.set_ylim(0.5*cutout.shape[0]-self.input_yaml['scale'],0.5*cutout.shape[0]+self.input_yaml['scale'])
             
                    
            
                
                
                
            #     divider = make_axes_locatable(ax2)
            #     cax = divider.append_axes("right", size="5%", pad=0.05)
            #     cb = fig.colorbar(im2, cax=cax)
            #     cb.ax.set_ylabel('Counts', rotation=270,labelpad = 5)
            #     cb.update_ticks()
        
            #     ax1.set_title('Image - Surface')
            #     ax2.set_title('Image + fake source')
                
            #     save_loc = os.path.join(write_dir,'probableLimit_'+base+'.pdf')
                
                
            
            #     fig.savefig(save_loc)
                
            #     plt.close()
                            
        except Exception as e:
            import os,sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno,e)
            magUpperlimit  = np.nan
            

        return magUpperlimit 
    
# =============================================================================
#     
# =============================================================================
    
    def getInjectedLimit(self,cutout,
                         epsf_model = None,
                         initialGuess = -5,
                         detection_limit = 3,
                         useBeta = True,
                         beta = 0.5,
                         plot = True,
                         # unityPSF = None,
                         # residualTable = None,
                         # functionParams = None,
                         subtraction_ready = False):
        
        
        
        try:
            import os
            import logging
            import numpy as np
            from functions import border_msg            
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib.ticker as ticker
            from matplotlib.lines import Line2D
            from astropy.nddata.utils import Cutout2D
            
            if np.isnan(initialGuess):
                initialGuess = -5
                
            
                    
            # magUpperlimit
            logger = logging.getLogger(__name__)
            logger.info(border_msg('Finding limiting magnitude through artifical source injection'))
            
            if (epsf_model  is None):
                
                logger.info('PSF model not available - skipping')
                return np.nan
            from functions import PointsInCircum,beta_value,mag,fluxUpperlimit
            
            
            from background import background
   
            
            # Dictionaries to track sources 
            inserted_magnitude = {}
            injected_f_source = {}
            recovered_f_source = {}
            recovered_magnitude = {}
            recovered_magnitude_e = {}
            location_noise = {}
            recovered_fwhm = {}
            recovered_fwhm_e = {}
            recovered_sigma_detection = {}
            recovered_max_flux = {}
            recovered_sources_list  = {}
            recovered_pos = {}
            recovered_SNR={}
            delta_recovered_SNR={}
            SNR_gradient_change = {}
            beta_gradient_change = {}
            beta_probability = {}
            
            
     
            

            # cutout, bkg_surface, bkg_median, bkg_std = background(input_yaml=self.input_yaml,image = cutout).remove()
        

            height, width = cutout.shape
            fittingGrid = np.meshgrid(np.arange(width),np.arange(height))
            gridy = np.array(fittingGrid)[1,:]
            gridx = np.array(fittingGrid)[0,:]
            
            
            dmag = 0.2
            fine_dmag = 0.05
            sourceNum = 6
            
            
            start_mag = initialGuess     
            
  
            redo = 1
            
            # detectionCutoff = 0.8
            
            iter_stop_limit = 5
            iter_stop = 0
            
    
            from photutils.datasets.make import apply_poisson_noise
            random_sources = PointsInCircum(self.input_yaml['limiting_magnitude']['inject_source_location']*self.input_yaml['fwhm'],
                                            shape = cutout.shape,
                                            n = sourceNum)

            xran = [abs(i[0])for i in random_sources]
            yran = [abs(i[1])for i in random_sources]
            
            injection_df = pd.DataFrame([list(xran),list(yran)])
            injection_df = injection_df.transpose()
            injection_df.columns = ['x_pix','y_pix']
            injection_df.reset_index(inplace = True,drop = True)
            
# =============================================================================
#             Find low S/N areas for source injection
# =============================================================================
            

            from aperture import aperture
            initialAperture = aperture(input_yaml=self.input_yaml,image = cutout)
            

            injection_df  = initialAperture.measure(sources=injection_df,plot = False)
            faketarget_flux = float(injection_df['flux_AP'].iloc[0])
            
            # logger.info   injection_df)
            # logger.infoinjection_df ['SNR'].values)
            if sum(injection_df ['SNR'].values>3)>0:
                logger.info('Ignoring %d positions with SNR > 3' % (sum(injection_df ['SNR'].values>3)))
                injection_df  = injection_df [injection_df ['SNR'].values<3]
       
                              
            
            faketarget_noise = float(injection_df['noiseSky'].iloc[0])
            faketarget_height = float(injection_df['maxPixel'].iloc[0])
            
            
            faketarget_beta = beta_value(n = detection_limit,
                                       sigma = faketarget_noise ,
                                       f_ul  = faketarget_height)
            
                                
    
            injection_df ['initial_beta'] = faketarget_beta
            injection_df['initial_noise'] = faketarget_noise 
            injection_df['initial_peak_flux'] = faketarget_height
            injection_df['f_ul'] = fluxUpperlimit(n = detection_limit,
                                                  beta_p = beta,
                                                  sigma = faketarget_noise )
        
                
            
# =============================================================================
# 
# =============================================================================

            nsteps = 50
            # dmag = np.linspace(0,dmag*nsteps,int(nsteps+1))
            dmag_range = np.linspace(0,dmag*nsteps,int(nsteps+1))
            
            # Are sources getting brighter or fainter - start off with fainter - 
            # swap to negative gradient if sources are initial not detected
            gradient = -1
            
            # initialise limiting magnitude
            inject_lmag = None
            lmag_found = False
            
            # dmag fine scale - inital parameters
            use_dmag_fine = False
            fine_dmag_range = None
            discrepancy_count = 1
            discrepancy_limit = 3
            
            
            exp_time = self.input_yaml['exptime']
            mag2image = lambda x: (  exp_time / 1) * 10**(x/-2.5)
            # Inital counter
            ith = 0
            
            #  SLection criteria
            criteria = True
     
            explore = False
            detection_cutout = 0.8
            zeropoint = 0
            print_progress = True
            detection_limit=3
            recovered_criteria = {}
            
            
            # Perform photometry on the isolated sources
            from aperture import aperture

            
            while True:
                try:
                    
                    if ith>nsteps:
                        break
                    if use_dmag_fine and not explore:
                        # Use small magnitude step size
                        dmag_step = gradient * fine_dmag_range[ith]
                    else:
                        # Use larger step size
                        dmag_step = gradient * dmag_range[ith]
            
                    # step labels to keep track of everything
                    step_name = round(start_mag + dmag_step + zeropoint, 3)
                    
                    
     
                    inserted_magnitude[step_name] = {}
                    recovered_magnitude[step_name] = {}
                    recovered_magnitude_e[step_name] = {}
                    recovered_fwhm[step_name] = {}
                    recovered_fwhm_e[step_name] = {}
                    location_noise[step_name] = {}
                    injected_f_source[step_name] = {}
                    recovered_f_source[step_name] = {}
                    recovered_max_flux[step_name] = {}
                    recovered_SNR[step_name] = {}
                    beta_probability[step_name] = {}
                    
                    
                    
                    # Sources are put in one at a time to avoid contamination
                    for k in range(len(injection_df)):
                       
                        inserted_magnitude[step_name][k] = []
                        recovered_magnitude[step_name][k] = []
                        recovered_magnitude_e[step_name][k] = []
                        recovered_fwhm[step_name][k] = []
                        recovered_fwhm_e[step_name][k] = []
                        location_noise[step_name][k] = []
                        injected_f_source[step_name][k] = []
                        recovered_f_source[step_name][k] = []
                        recovered_max_flux[step_name][k] = []
                        recovered_SNR[step_name][k] = []
                        beta_probability[step_name][k] = []
                        
                        cutout_copy = cutout.copy()
            
                        for j in range(redo):
                            
                   
                            dx = np.random.uniform(-0.5,0.5) 
                            dy = np.random.uniform(-0.5,0.5) 
                            
                            if redo == 1: 
                                dx = 0
                                dy = 0
                            
                            x_inject = injection_df['x_pix'].values[k] + dx
                            y_inject = injection_df['y_pix'].values[k] + dy
                            
                            fake_source_on_target = epsf_model.evaluate(x = gridx, y=gridy, 
                                                                        flux = mag2image( start_mag + dmag_step ), 
                                                                        x_0 = x_inject,
                                                                        y_0 = y_inject)
                               
                            
     
                            

                            f_injected_source = np.nanmax(fake_source_on_target)/exp_time
                            
                            new_cutout =      cutout_copy + fake_source_on_target
                            
                            
                            
 
                            initialAperture = aperture(input_yaml=self.input_yaml,image = new_cutout)

                            aperture_fakesources = initialAperture.measure(sources= injection_df.iloc[[k]],plot = False)
                            
                            
                                                
                            faketarget_flux = float(aperture_fakesources['flux_AP'].iloc[0])
                            faketarget_noise = float(aperture_fakesources['noiseSky'].iloc[0])
                            faketarget_height = float(aperture_fakesources['maxPixel'].iloc[0])
                            faketarget_SNR = float(aperture_fakesources['SNR'].iloc[0])
                            
                                
                                                   
                       
                            fake_target_beta = beta_value(n = detection_limit,
                                                          sigma =  faketarget_noise ,
                                                          f_ul  = faketarget_height)
                   
                        
                            
                            
                            mag_recovered =  mag(faketarget_flux)
                            location_noise[step_name][k].append(faketarget_noise)
                            inserted_magnitude[step_name][k].append(float(start_mag+dmag_step))
                            recovered_magnitude[step_name][k].append(mag_recovered)
                            recovered_max_flux[step_name][k].append(faketarget_noise)              
                            recovered_SNR[step_name][k].append(faketarget_SNR)
                            beta_probability[step_name][k].append(1-fake_target_beta)
                            injected_f_source[step_name][k].append(f_injected_source)
                            recovered_max_flux[step_name][k].append(faketarget_height)
                                
     

                    # recovered_sources = np.concatenate([np.array(recovered_SNR[step_name][k]) >= detection_limit for k in range(len(injection_df))])
                    
                    
                    recovered_sources = np.concatenate([1 - np.array(beta_probability[step_name][k]) >= 0.75 for k in range(len(injection_df))])

                    recover_test = np.sum(recovered_sources)/len(recovered_sources) >=  1 - detection_cutout

                    recovered_criteria[step_name] = recover_test
                    
                    
                    # If the first source inject comes back negative on the first interation - flip the dmag steps i.e. fainter -> brighter
                    if recovered_criteria[step_name] == criteria and ith == 0 and not use_dmag_fine:
                        if print_progress:
                            logger.info('> Initial injection sources not recovered - injecting brighter sources')
                        
                        gradient *= -1
                        
                        criteria = not criteria 
                  
                    # else if the detection meetings the criteria i.e. sources are (not) detected 
                    elif recovered_criteria[step_name] == criteria:
                        
                        
                        if iter_stop == 0 and not use_dmag_fine:
                            if print_progress:
                                logger.info('\n\nApproximate limiting magnitude: %.3f mag - using finer scale\n' % (zeropoint+start_mag+dmag_step))
                            
                            use_dmag_fine = True
            
                            fine_nsteps = int(1/fine_dmag)+50
                            
                            # reset start magnitude
                            start_mag = start_mag+dmag_step
                            
                            fine_dmag_range =  np.linspace(0,fine_dmag*fine_nsteps,int(fine_nsteps))
                            
                            gradient *= -1
                            
                            criteria = not criteria 
                            
                            # Restart the count
                            ith = 0
                            
                        elif iter_stop == 0:
                            
                            if print_progress:
                               logger.info('\nLimiting mag found, checking overshoot...\n' )
                           
                            # First time finer scale meets criteria, not magnitude
                            inject_lmag = start_mag + dmag_step
                            
                            lmag_found = True
                        
                            iter_stop+=1
                            
                            
                            
                        else:
                            
                            iter_stop+=1
                        
                    else:
                        
                        if lmag_found:
                            if print_progress:
                                logger.info('\n\nOvershoot discrepancy[%d], resetting...\n' % discrepancy_count)
                                discrepancy_count+=1
                        if discrepancy_count>discrepancy_limit:
                            pass
                        else:
                        
                            # didn't meet criteria, keep going
                            lmag_found = False
                            iter_stop = 0
                        
                    if iter_stop > iter_stop_limit:
                        #Done
                        # if print_progress:
                        #     logger.info('\nLimiting magnitude: %.3f \n' % ( inject_lmag+zeropoint ))
                            
                        
                        break
                    
                    else:
                        
                        ith = ith + 1
                        
                        
                        
                except Exception as e:
                    
                    import os,sys
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    # logger.info(exc_type, fname, exc_tb.tb_lineno)
                    logger.info('\nInjection failed: ' + str(e) )
                    logger.info(exc_type, fname, exc_tb.tb_lineno)
                 
                    inject_lmag = np.nan
                    save_cutouts = False
                    
                    break
                    
                    
            if plot and False:
                
                
                
                fig = plt.figure()
                
                
                ax1 = fig.add_subplot(111)
                

                x = [np.mean(list(inserted_magnitude[i].values()))for i in inserted_magnitude.keys()]
                y = [1 - np.mean(list(beta_probability[i].values()))for i in inserted_magnitude.keys()]

                ax1.scatter(x,y,marker = 's',color = 'red')
                
                
                ax1.invert_xaxis()
                
                ax1.axvline(inject_lmag)
           
                ax1.set_xlabel('Injected Mag')
                ax1.set_ylabel('Beta Probability')
                
                
                plt.show()
                         
        except Exception as e:
            import os, sys
    
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno)
            logger.info('\nInjection failed: ' + str(e))
            inject_lmag = np.nan
            save_cutouts = False
    
 
            
            
        # break


        return  inject_lmag
    