#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:50:02 2022

@author: seanbrennan
"""

class find_fwhm:
    
    def __init__(self, input_yaml: dict):
        """
        Initialize the FindFWHM class with configuration from input_yaml.

        :param input_yaml: Configuration parameters for the FWHM calculation.
        """
        self.input_yaml = input_yaml

    
    def create_circular_mask(self,h, w, center=None, radius=None):
        '''
        Create mask centered within a self.image with height *h* and width *w* centered on
        *center* with a radius *radius*
        
        :param h: height of self.image
        :type h: int
        :param w: width of self.image
        :type w: int
        :param center: pixel location of mask, if none, mask out center of self.image defaults to None
        :type center: tuple with x, y pixel position, optional
        :param radius: radius of mask in pixels, if none, use the smallest distance between the center and self.image walls defaults to None
        :type radius: float, optional
        :return: self.image with shape h,w with masked regions
        :rtype: 2D array with height *h* and width *w*
    
        '''
        import numpy as np
    
        if center is None:
            # use the middle of the self.image
            center = (int(w/2), int(h/2))
            
        if radius is None: 
            # use the smallest distance between the center and self.image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])
    
        Y, X = np.ogrid[:h, :w]
        
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
        mask = dist_from_center <= radius
        
        
        return mask

    
    def combine(self,dictionaries):
            '''
            Combine list of dictionaies 
            
            :param dictionaries: List of dictionaries
            :type dictionaries: Dict
            :return: Combined dictionary with values equal to a list of original dictionary values
            :rtype: dict
        
            '''
            combined_dict = {}
            for dictionary in dictionaries:
                for key, value in dictionary.items():
                    combined_dict.setdefault(key, []).append(value)
            return combined_dict
        
        

        
        
    def plot_FWHM_distribution(self,dataframe,mean_fwhm = None):
        
        import os
        import matplotlib.pyplot as plt
        from functions import set_size,get_normalized_histogram
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
        
        save_loc = os.path.join(self.input_yaml['write_dir'],'fwhmDistribution_'+self.input_yaml['base']+'.pdf')
        
        
        plt.ioff()
        fig = plt.figure(figsize = set_size(330,1))
        
        ax1 = fig.add_subplot(111)
        
        
        bins,hist = get_normalized_histogram(dataframe['fwhm'].values, bins='auto')
        # print(bins,hist)
        ax1.stairs(bins,hist,
                 color = 'darkgrey',
                 label = 'All',zorder = 0,lw = 0.5)
        
        # print(bins,hist)
        # 
        # print(xlims)

        
        bins,hist = get_normalized_histogram(dataframe['fwhm'][dataframe['include_fwhm']].values, bins='auto')
        # print(bins,hist)
        ax1.stairs(bins,hist,
                 color = 'grey',
                 label = 'FWHM cutoff',
                 zorder = 1,lw = 0.5)
        
        
        
        # xlims =  np.nanpercentile(dataframe['fwhm'][dataframe['include_fwhm']].values, [0.1, 99.9])
        
        
        # ax1.stairs(dataframe['fwhm'][dataframe['include_fwhm']],
        #          histtype = 'stairs',
        #          color = 'orange',
        #          label = 'FWHM cutoff',
        #          density = True)
        
        
        bins,hist = get_normalized_histogram(dataframe['fwhm'][dataframe['include_median']].values, bins='auto')
        # print(bins,hist)
        ax1.stairs(bins,hist,
                 color = 'lightgrey',
                 label = 'Background cutoff',
                 zorder = 2,lw = 0.5)
        
        # ax1.hist(dataframe['fwhm'][dataframe['include_median']],
        #          histtype = 'stairs',
        #          color = 'green',
        #          label = 'Background cutoff',
        #          density = True)
        
                
        
        bins,hist = get_normalized_histogram(dataframe['fwhm'][(dataframe['include_fwhm']) & (dataframe['include_median'])].values, bins='auto')
        # print(bins,hist)
        ax1.stairs(bins,hist,
                 color = 'red',
                 label = 'Selected FWHM',
                 zorder = 3)
        
        
        
        # ax1.set_xlim(xlims[0],xlims[1])
# 
        
        ax1.set_xlabel('Full Width Half Maximum [pixels]')
        ax1.set_ylabel('N [Normalised]')
        
        ax1.legend(frameon = False)
        
        
        plt.savefig(save_loc)
        plt.close('all')
        
        
        
        
        return
        
# =============================================================================
#         
# =============================================================================
    
    def measure_object(self,object_df,image,functionParams,update_position = True):
        
        import logging
        from functions import border_msg
        import warnings
        from lmfit import Model
        import numpy as np
        
        from background import background
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Measure FWHM of object(s)'))
        
        # Filter out Astropy warnings
        from astropy.utils.exceptions import AstropyWarning
        warnings.filterwarnings('ignore', category=AstropyWarning)
        
        
        imageScale = self.input_yaml['scale']
        use_moffat = self.input_yaml['fitting']['use_moffat']
        default_moff_beta = self.input_yaml['fitting']['default_moff_beta']
        
        if 'sigma' in functionParams:
            useMoffat = False
        else:
            useMoffat = True
        
        #  Prepare the analytical mode for fitting the sources
        if useMoffat:
            from functions import Moffat,alphaMoffat,moffat_fwhm

            fitting_model_fwhm = moffat_fwhm
            fittingModel = Model(Moffat,independent_vars=['gridx','gridy'])
            
        else:
            from functions import Gaussian,sigmaGaussian,gauss_sigma2fwhm
            fittingModel = Model(Gaussian,independent_vars=['gridx','gridy'])
            fitting_model_fwhm = gauss_sigma2fwhm
        
        

        
        if useMoffat:
            
            fittingModel.set_param_hint('alpha',
                                        value = alphaMoffat(self.input_yaml['fwhm']),
                                        vary = True)
            fittingModel.set_param_hint('beta',
                                               value = default_moff_beta,
                                               min = 0,
                                               vary =False )
 
            
        else:
            
            fittingModel.set_param_hint('sigma',
                                        value = self.input_yaml['fwhm'] / 2.355,
                                        min = 1e-3,
                                        vary = True)
            
            
        fittingModel.set_param_hint('x0',
                                  value = imageScale,
                                  min   = imageScale - self.input_yaml['dx'],
                                  max   = imageScale + self.input_yaml['dx'], )
  
                
        fittingModel.set_param_hint('y0',
                                    value = imageScale,
                                    min   = imageScale - self.input_yaml['dy'],
                                    max   = imageScale + self.input_yaml['dy'], )

      
        for index, row in object_df.iterrows():
            try:
                
                xPix = float(row['x_pix'])
                yPix = float(row['y_pix'])
                
                
                        

                
                sourceCutout = image[int(yPix-imageScale):int(yPix+imageScale),
                                     int(xPix-imageScale):int(xPix+imageScale)]
                
                
                sourceCutout, _,_,_ = background(input_yaml=self.input_yaml, image = sourceCutout).remove()
                                                                    
                fittingPars = fittingModel.make_params()
                
                if len(sourceCutout) ==0: continue
            
            
                     
                
                if sourceCutout.size == 0: continue
            
                
            
     
    
                fittingModel.set_param_hint('A',
                                            value = 0.75 * np.nanmax(sourceCutout),
                                            min = 0.8*np.nanmin(sourceCutout),
                                            max = 1.5*np.nanmax(sourceCutout)
                                            )
                fittingModel.set_param_hint('sky',
                                            value = np.nanmedian(sourceCutout),
                                            vary = True
                                            )

                h, w = sourceCutout.shape
                x  = np.arange(0, w)
                y  = np.arange(0, h)
                grid = np.meshgrid(x, y)

                    
                fittingPars = fittingModel.make_params()
                

                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
        
                    
                    result = fittingModel.fit(data = sourceCutout,
                                              params = fittingPars,
                                              gridx = np.array(grid)[0],
                                              gridy = np.array(grid)[1],
                                              # weights = weights,
                                              nan_policy = 'omit',
                                              method = 'least_sqaure'
                                              )
                    
                    if use_moffat:

                         fwhm_fit = fitting_model_fwhm(dict(alpha=result.params['alpha'].value,
                                                            beta=result.params['beta'].value)
                                                       )

                    else:
                         fwhm_fit = fitting_model_fwhm(dict(sigma=result.params['sigma'].value))

                    xCentered = xPix  + result.params['x0'].value  - imageScale
                    yCentered = yPix  + result.params['y0'].value  - imageScale

                
            except Exception as e:
                import sys,os
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno,e)
                fwhm_fit = np.nan
                xCentered = np.nan
                yCentered = np.nan
                
                
            object_df.at[index,'fwhm'] = fwhm_fit   
            
            if update_position:
                object_df.at[index,'x_pix'] = xCentered
                object_df.at[index,'y_pix'] = yCentered
            
            
        return  object_df 
    
# =============================================================================
#     
# =============================================================================
    def fit_gaussian(self,data,x = None,y=None,dx = None,dy = None,sigma = None):
        
        from lmfit.models import Gaussian2dModel
        import numpy as np
        import warnings
        import matplotlib.pyplot as plt
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # print(data)
                height, width = data.shape
                fittingGrid = np.meshgrid(np.arange(width),np.arange(height))
                gridy = np.array(fittingGrid)[1,:]
                gridx = np.array(fittingGrid)[0,:]
                
                # plt.figure()
                # plt.imshow(data)
                # plt.show()
                if x is None:
                    x = data.shape[1]/2
                    y = data.shape[0]/2
                  
                    
                core = Gaussian2dModel() 
        
                params = core.make_params()
                params['centerx'].set(value = x,min = 1, max = data.shape[1]-1)
                params['centery'].set(value = y,min = 1, max = data.shape[0]-1)
                
                params['amplitude'].set(value = np.nanmax(data)*(sigma**2)/0.16 * 0.25,
                                        min = np.nanmin(data)*(sigma**2)/0.16 * 1e-6,
                                        max = np.nanmax(data)*(sigma**2)/0.16 * 1e6)
                
                params['sigmax'].set(value = sigma,
                                     min = 0.5 / (2 * np.sqrt(2 * np.log(2))),
                                     max = 20 / (2 * np.sqrt(2 * np.log(2))))
                
                
                params['sigmay'].set(value = sigma,
                                     min = 0.5 / (2 * np.sqrt(2 * np.log(2))),
                                     max = 30 / (2 * np.sqrt(2 * np.log(2))))
                
                params['sigmay'].set(expr = 'sigmax')
                # 
                # print(params)
                
                results = core.fit(data,x = gridx,y = gridy,params=params,
                                   nan_policy = 'omit',method = 'least_sqaure')
                    

                    
                xfit = results.params['centerx'].value
                yfit = results.params['centery'].value
                
                A =  results.params['height'].value
                fwhm = results.params['fwhmx'].value 
            
            return fwhm,xfit,yfit,A
        except: 
            return [np.nan]*4 
        
        
        
        
# =============================================================================
#         
# =============================================================================
        
        
        
    def measure_image(self,image,scale = 31,fwhm = None,sigma = None,fwhm_inital = 5,mask_sources_XY_R = [],
                      dontClean = False,DataFrame  = None,default_scale =15.5,mask = None,no_clean = False):

        from astropy.stats import sigma_clipped_stats
    
        from photutils.detection import IRAFStarFinder,DAOStarFinder
        
        import numpy as np
        from astropy.stats import mad_std
        import pandas as pd
        from astropy.stats import sigma_clip
        import logging
        import os,gc
        import warnings
    
        from random import sample
        from lmfit import Model
    
        from functions import gauss_sigma2fwhm,Gaussian,gauss_fwhm2sigma
        from functions import Moffat,moffat_fwhm
        from functions import pix_dist,border_msg
        from photutils.detection import find_peaks
                
        from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
        
        from background import background
        import matplotlib.pyplot as plt
        
            
        from astropy.nddata import Cutout2D
        
        try:
            image_copy = image.copy()
            
            
            
            if not (fwhm is None) and not (sigma is None):
                
                
                logging.info(border_msg(f'Performing source detection with FWHM: {fwhm:.1f}  px and sigma: {sigma:.1f}')) 
                
                image_mean, image_median, image_std = sigma_clipped_stats(image_copy, sigma=3.0)  
                            
                threshold_value =  sigma * image_std
           
             
                daofind = IRAFStarFinder(fwhm      = fwhm ,
                                          # sigma_radius=fwhm  / 2.35,
                                        threshold = threshold_value,
                                        # sharplo   =  0.3, sharphi = 1,
                                        # roundlo=0, roundhi=1,
                                        # brightest = 250,
                                        minsep_fwhm = 1,
                                        exclude_border = True,
                                        peakmax = self.input_yaml['saturate'] 
                                   )
                
                
                sources = daofind(image_copy- image_median,     mask = mask ).to_pandas()
                
                sources['x_pix'] = sources['xcentroid'] 
                sources['y_pix'] = sources['ycentroid'] 
                
                         
                logging.info(f'> Number of sources: {len(sources)}')
                
                
                return sources
            
            scale_multipler = self.input_yaml['source_detection']['scale_multipler']
            
                    
            # Filter out Astropy warnings
            from astropy.utils.exceptions import AstropyWarning
            warnings.filterwarnings('ignore', category=AstropyWarning)
                        
    
            
            from photutils.centroids import (centroid_1dg, centroid_2dg,
                                             centroid_com, centroid_quadratic)
    
            logger = logging.getLogger(__name__)
        
            logging.info(border_msg('Finding Full Width Half Maximum'))
            
            
    
            image_mean, image_median, image_std = sigma_clipped_stats(image_copy, sigma=3.0)  
            
            single_source = True
            
            threshold_value = detect_threshold(image_copy, nsigma=5) 
            if (fwhm is None):
                single_source = False
                # threshold_value = (5.0 * image_std) + image_median
                all_sources = find_peaks(image_copy,threshold_value, box_size=scale,
                                      border_width = int(scale), 
                                       npeaks = 100, 
                                      centroid_func = centroid_2dg,
                                      mask = mask ).to_pandas()
                
                
                
                
                
    
                all_sources['xcentroid'] =  all_sources['x_peak']
                all_sources['ycentroid'] =  all_sources['y_peak']
                all_sources['peak'] =  all_sources['peak_value']
                
                
                
                sources = all_sources.copy()
                
                if no_clean:
                    
                    sources['x_pix'] = sources['xcentroid'] 
                    sources['y_pix'] = sources['ycentroid'] 
                    
                    sources = sources[['x_pix','y_pix']]
                    return sources
            
                logging.info(f'> Initial number of sources: {len(sources)}')
                
                
                # Find isolated sources 
                isolation_distance = 11
                
                nearest_neightbour = []
                for idx,row in sources.iterrows():
                    
                    distance2sources = pix_dist(row['xcentroid'],all_sources['xcentroid'],row['ycentroid'],     all_sources['ycentroid'])
                    distance2sources[distance2sources<1] = np.nan
                    nearest_neightbour.append(np.nanmin(distance2sources))
                    
                sources['nearest_neighbour'] = nearest_neightbour
                 
                not_crowded = sources['nearest_neighbour']<isolation_distance
                if sum(not_crowded)>0:
                    sources = sources[not_crowded]
                    logging.info(f'> Number of isolated [> {isolation_distance} px] sources: {len(sources)}')
                
                saturated = sources['peak']>self.input_yaml['saturate']
         
                
                if sum(saturated) != len(sources):
                    sources = sources[~saturated]
                    logging.info(f'> Removed {sum(saturated)} saturated sources')
                
                
                
                # print(sources)
                fwhm_guess = []
                sigma = 3
                for idx,row in sources.iterrows():
                    
                    
                    xpos = float(row['xcentroid'] )
                    ypos = float(row['ycentroid'] )
                    position = (xpos, ypos)
                    size = 2 * scale  # Diameter of the aperture
        
                    # Create the Cutout2D object
                    cutout = Cutout2D(image_copy, position, size, mode='partial', fill_value=0).data
                    
                    
                    mean, median, std = sigma_clipped_stats(cutout, sigma=3.0)  
                    cutout -= median
        
                    fwhm,xfit,yfit,A = self.fit_gaussian(cutout,sigma = sigma)
        
                    
                    fwhm_guess.append(fwhm)
                    
                    
                              
                sources['fwhm'] = fwhm_guess
                
                sources = sources[sources['fwhm']>1]
        
                FWHM_mask = sigma_clip(sources['fwhm'].values,
                                  sigma=3,
                                  masked = True,
                                   cenfunc = np.nanmean,
                                   stdfunc = mad_std
                                  )
                
                logging.info(f'> Masking {sum(FWHM_mask.mask)} sources with irrelgular FWHM')
                sources = sources[~FWHM_mask.mask]
                        
                
                
                
                fwhm_inital = np.nanpercentile(sources['fwhm'].values, 50)
                logging.info(f'> Initial guess at FWHM: {fwhm_inital:.3f} px')
                
            else:
                fwhm_inital = fwhm
                
            
            
            if fwhm_inital<2:
                fwhm_inital = 3
                
            scale = int(np.ceil(scale_multipler * fwhm_inital) ) + 0.5
            
            
            scale = max(scale,default_scale)
            # threshold_value = 5
            
            threshold_value = np.nanmean(detect_threshold(image_copy- image_median, nsigma = 5) )
       
         
            daofind = IRAFStarFinder(fwhm      = fwhm_inital ,
                                      # sigma_radius=fwhm_inital  / 2.35,
                                    threshold = threshold_value,
                                    # sharplo   =  0.3, sharphi = 0.8,
                                    # roundlo=0, roundhi=3,
                                    brightest = 250,
                                    # minsep_fwhm = 2,
                                    exclude_border = True,
                                    peakmax = self.input_yaml['saturate'] 
                               )
                        
            
            # print(sources)
            
            DAOFIND_sources = daofind(image_copy- image_median,     mask = mask )
            
            if DAOFIND_sources is not None:
                
                sources = DAOFIND_sources.to_pandas()

                
                logging.info(f'\n> Initial number of sources: {len(sources)}')
            
            
                sharpness_mask = sigma_clip(sources['sharpness'].values,
                                  sigma = 3,
                                  masked = True,
                                    cenfunc = np.nanmean,
                                    stdfunc = mad_std,
                                  maxiters=10)
                
                if sum(sharpness_mask.mask)>0:
                    logging.info(f'> Masking {sum( sharpness_mask.mask)} sources with irrelgular sharpness')
                    sources = sources[~sharpness_mask.mask]
                    
                    
                roundness_mask = sigma_clip(abs(sources['roundness'].values),
                                  sigma = 3,
                                  masked = True,
                                  cenfunc = np.nanmean,
                                  stdfunc = mad_std,
                                  maxiters=10)
                
                if sum(roundness_mask.mask)>0:
                    logging.info(f'> Masking {sum( roundness_mask.mask)} sources with irrelgular roundness')
                    sources = sources[~roundness_mask.mask]
                    
                

            s2n_limit = 10
            sources = sources[sources['peak']/image_std>s2n_limit ]
            logging.info(f'> Number of high S/N  [> { s2n_limit }] sources: {len(sources)}')
                
            # logging.info(f'> Number of sources found: {len(sources)}')

            nearest_neightbour = []

            fwhm_guess = []
            xfits = []
            yfits = []
            s2n = []
            background = []
            eccentricity = []
            areas = []
            

            
            # sigma_initial  = fwhm_inital/ (2 * np.sqrt(2 * np.log(2)))
  
                

            for idx,row in sources.iterrows():
                
                xpos = float(row['xcentroid'] )
                ypos = float(row['ycentroid'] )
                
                
                A = float(row['peak'])
                fwhm = float(row['fwhm'])
                position = (xpos, ypos)
                size = 2 * scale  # Diameter of the aperture
    
                # Create the Cutout2D object
                cutout = Cutout2D(image_copy, position, size, mode='partial', fill_value=0).data
                mean, median, std = sigma_clipped_stats(cutout, sigma=3.0) 

                catalog = []
                if cutout.size != 0:   
                    
                    # threshold_value = (3 * std) + median
                    
                    threshold_value = detect_threshold(cutout, nsigma=4) 
                                # Detect sources above the threshold
    
                    
                    if not single_source:
                        
                        try:
                            from photutils.segmentation import SourceFinder,detect_threshold, detect_sources, deblend_sources, SourceCatalog
                            from astropy.stats import sigma_clipped_stats
                            from astropy.convolution import convolve, Gaussian2DKernel
                            
                            convolved_data = cutout

                            segment_map = detect_sources(convolved_data, threshold_value, npixels=5)
                            
                            
                            segm_deblend = deblend_sources(cutout, segment_map,
                               npixels=5, nlevels=32, contrast=0.001,
                               progress_bar=False)
                            
                            # Create a catalog of sources
                            catalog = SourceCatalog(convolved_data , segm_deblend)
                            
        
                            catalog = catalog.to_table().to_pandas()
                        except: catalog = []
    
                
                
                if len(catalog)!=1 or cutout.size == 0:
                    fwhm,xfit,yfit,A  = [np.nan]*4
                    std = np.nan
                    source_eccentricity =np.nan
                    area = np.nan
                    
       
                else:
                    
                    try:
                        source_eccentricity = float(catalog['eccentricity'].iloc[0])
                        area = float(catalog['area'].iloc[0])
                    except: 
                        # print(catalog)
                        source_eccentricity = np.nan
                        area = np.nan
                    
            # =============================================================================
            #                 
            # =============================================================================

                xfit = row['xcentroid'] #+ xfit- scale
                yfit = row['ycentroid'] #+ yfit- scale
                
                xfits.append(xfit)
                yfits.append(yfit)
                
                s2n.append(A/std)
                background.append(mean)
    
                fwhm_guess.append(fwhm)
                eccentricity.append(source_eccentricity)
                areas.append(area)
                
                
        # exit()
                      
            sources['fwhm'] = fwhm_guess
            sources['x_pix'] = xfits
            sources['y_pix'] = yfits
            sources['s2n'] = s2n
            sources['noise'] =     background
            sources['eccentricity'] = eccentricity
            sources['area'] = areas
     
        # print(sources['eccentricity'])
        
       
        
            sources = sources[np.isfinite(sources['fwhm']) & (np.isfinite(sources['eccentricity']))]
            logging.info(f'> Number of fitted sources: {len(sources)}')
            
            
 
            
                        
            
            saturated = sources['peak']>self.input_yaml['saturate']
            if sum(saturated) != len(sources):
                sources = sources[~saturated]
            
         
         
            # print(sources['noise'])
            from astropy.stats import sigma_clip
            

            eccentricity_mask = sigma_clip(sources['eccentricity'].values,
                              sigma = 5,
                              masked = True,
                                cenfunc = np.nanmean,
                                stdfunc = mad_std,
                              maxiters=10)
            
            
            if sum(eccentricity_mask.mask) >0:
                logging.info(f'> Masking {sum(eccentricity_mask.mask)} sources with irrelgular shapes')
                sources = sources[~eccentricity_mask.mask]
            
  

            
     
            # area_mask = sigma_clip(sources['area'].values,
            #                   sigma = 3,
            #                   masked = True,
            #                     cenfunc = np.nanmean,
            #                     # stdfunc = mad_std,
            #                   maxiters=10)
                
            # if sum(area_mask.mask)>0:
                
            #     logging.info(f'> Masking {sum(area_mask.mask)} sources with irrelgular areas')
            #     sources = sources[~area_mask.mask]
                
            # sky_mask = sigma_clip(sources['noise'].values,
            #                   sigma = 3,
            #                   masked = True,
            #                     cenfunc = np.nanmean,
            #                     stdfunc = mad_std,
            #                   maxiters=10)
            
            # if sum(sky_mask .mask)>0:
            #     logging.info(f'> Masking {sum(sky_mask .mask)} sources with irrelgular backgrounds')
            #     sources = sources[~sky_mask .mask]
            

            

            FWHM_mask = sigma_clip(sources['fwhm'].values,
                              sigma = 3,
                              masked = True,
                              cenfunc = np.nanmean,
                              stdfunc = mad_std,
                              maxiters=25)
            
            if sum(FWHM_mask.mask)>0:
                logging.info(f'> Masking {sum(FWHM_mask.mask)} sources with irrelgular FWHM')
                sources = sources[~FWHM_mask.mask]
                 

            nSources = 100
            if len(sources)>nSources:
                
          
                    
                logger.info(f'\nSelected {nSources} sources close to target location')
                dist = pix_dist(float(self.input_yaml['target_x_pix']), sources['x_pix'].values,
                                float(self.input_yaml['target_y_pix']), sources['y_pix'].values)
                
                sources['dist2target'] = dist
                
                sources.sort_values(by = 'dist2target',inplace = True)
                sources = sources.head(nSources)

            fwhm = np.nanpercentile(sources['fwhm'].values, 50)
            
            logging.info(f'\n> Image FWHM: {fwhm:.3f} [pixels]')
            
            scale = int(np.ceil(scale_multipler * fwhm) ) + 0.5
            scale = max(scale,default_scale)
        
            logging.info(f'> Cutout size: {scale:.1f} [pixels]')
            
            return fwhm, sources, scale
            
      
        
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            return np.nan,np.nan,np.nan

    
