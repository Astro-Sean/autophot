#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:50:18 2022

@author: seanbrennan
"""

class psf:
    """
    A class to handle Point Spread Function (PSF) operations on astronomical images.
    """

    def __init__(self, input_yaml, image):
        """
        Initializes the PSF class with input parameters and image data.
        
        Parameters:
        input_yaml (str): Path to the YAML configuration file.
        image (ndarray): The image data to process.
        """
        self.input_yaml = input_yaml
        self.image = image

    
# =============================================================================
#
# =============================================================================
    def create_initial_moffat_psf(self, fwhm_x, fwhm_y=None, size=25, beta=4.765):
        
        from astropy.modeling.models import Moffat2D
        import numpy as np
        """
        Create an initial Moffat PSF model for use in EPSFBuilder.
        
        Parameters:
        - fwhm_x: float
            Full Width at Half Maximum of the Moffat PSF in the x direction.
        - fwhm_y: float
            Full Width at Half Maximum of the Moffat PSF in the y direction.
        - size: int
            Size of the square grid (size x size) for the PSF.
        - beta: float, optional
            The beta parameter of the Moffat PSF. Default is 4.765.
            https://academic.oup.com/mnras/article/328/3/977/1247204
        
        Returns:
        - initial_psf: 2D array
            2D array representing the initial Moffat PSF.
        """
        
        if (fwhm_y is None):
            fwhm_y = fwhm_x
        # Convert FWHM_x and FWHM_y to the Moffat scale parameters (alpha_x and alpha_y)
        alpha_x = fwhm_x / (2 * np.sqrt(2**(1 / beta) - 1))
        alpha_y = fwhm_y / (2 * np.sqrt(2**(1 / beta) - 1))
        
        # Create a grid for the PSF
        y, x = np.mgrid[:size, :size]
        x_center, y_center = size // 2, size // 2
        
        # Create the Moffat PSF model with separate alpha_x and alpha_y for x and y directions
        moffat = Moffat2D(amplitude=1.0, x_0=x_center, y_0=y_center, alpha=alpha_x, gamma=beta)
        
        # Scale the y axis separately
        moffat.y_stddev = alpha_y / alpha_x
        
        # Evaluate the PSF on the grid
        initial_psf = moffat(x, y)
        
        return initial_psf


    def build(self, psfSources,
              prepareTemplates=False,

              usePSFlist=False,
              numSources=10):
        
        
        
        import os
        import sys
        import logging
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        from astropy.io import fits
        from astropy.nddata import NDData
        from astropy.stats import SigmaClip
        from astropy.table import Table
        from astropy.visualization import ZScaleInterval, ImageNormalize
        
        from photutils.detection import IRAFStarFinder
        from photutils.psf import extract_stars, EPSFBuilder, EPSFFitter, IterativePSFPhotometry,EPSFModel
        from photutils.background import LocalBackground, MMMBackground
        from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic
        
        from functions import border_msg, set_size
        from astropy.stats import sigma_clip


        logger = logging.getLogger(__name__)

        # Save PSF model for use later
        fpath = self.input_yaml['fpath']
        write_dir = self.input_yaml['write_dir']
        base = os.path.basename(fpath).split('.')[0]

        try:

            logger.info(border_msg(
                'Build ePSF model from sources in the field'))

            # Only fit to a small image with radius ~the fwhm around the center of the source

            # fittingRadius = int(np.ceil(
            #     self.input_yaml['fitting']['fitting_radius'] * self.input_yaml['fwhm']))+0.5
            # fittingGrid = np.meshgrid(np.arange(0,2*fittingRadius),np.arange(0,2*fittingRadius))
            aperture_radius = self.input_yaml['photometry']['ap_size'] * \
                self.input_yaml['fwhm']
            # Size of large cutout
            scale = self.input_yaml['scale']
            # imageGrid = np.meshgrid(np.arange(0,2*imageScale),np.arange(0,2*imageScale))

            # is the user has not given as list of stars to use as PSF gridx
            if not usePSFlist:
                
                border =  scale
                
                width = self.image.shape[1]
                height = self.image.shape[0]
                mask_x = (psfSources['x_pix'].values >= border) & (
                    psfSources['x_pix'].values < width - border)
                mask_y = (psfSources['y_pix'].values >= border) & (
                    psfSources['y_pix'].values < height-border)

                psfSources = psfSources[(mask_x) & (mask_y)]
                psfSources = psfSources[psfSources['s2n'] > 10]
       
  
                s2n_mask = sigma_clip(psfSources['fwhm'].values,
                                    sigma_lower = 3,
                                    sigma_upper = 5,
                                    masked = True,
                                    # maxiters=10,
                                    cenfunc = np.nanmean,
                                    stdfunc = 'mad_std'
                                    )
                

                # print(psfSources['s2n'].values)
                psfSources =psfSources[~s2n_mask.mask]
                # print(psfSources['s2n'].values)
                
                psfSources.sort_values(by='s2n', ascending=False, inplace=True)
                psfSources = psfSources.head(numSources)
 
            else:

                psfSources = pd.read_csv(usePSFlist)

            logger.info(f'> Using {len(psfSources)} sources to build the ePSF')

            image_copy = self.image.copy()

            nddata = NDData(data=image_copy)


            
            
            
            boxsize = max(int(np.ceil(self.input_yaml['fwhm'])),7 )
            boxsize = int(boxsize if boxsize % 2 != 0 else boxsize+1)
            
            fitsize = max(int(np.ceil(2*self.input_yaml['fwhm'])),11)
            fitsize = int(fitsize if fitsize % 2 != 0 else fitsize+1)
            

            
            scale = int(scale*2)
            
            star_shape = (scale if scale % 2 != 0 else scale+1)
            
            data =  self.create_initial_moffat_psf(fwhm_x = self.input_yaml['fwhm'],size = star_shape)
            
            xcenter = star_shape / 2.
            ycenter = star_shape / 2.
    
            epsf = EPSFModel(data=data, origin=(xcenter, ycenter),norm_radius=aperture_radius)
                
            from photutils.background import LocalBackground, MMMBackground, MedianBackground

            localbkg_estimator = LocalBackground(aperture_radius +  np.ceil(self.input_yaml['fwhm']),
                                                  aperture_radius + np.ceil(self.input_yaml['fwhm'])+5,
                                                   MedianBackground()
                                                  )
          
            
            from photutils.segmentation import detect_threshold
            
            
   
            
            max_check = 5
            counter = 0
        
            while True:
                counter+=1
                    
                psfSources.reset_index(inplace = True,drop = True)
                import matplotlib.pyplot as plt
                from astropy.visualization import simple_norm
                from photutils.utils import circular_footprint
                stars = Table()
                stars['x'] = psfSources['x_pix'].values
                stars['y'] = psfSources['y_pix'].values
                extracted_stars = extract_stars(nddata, stars, size=star_shape)
                
                footprint = circular_footprint(int(np.ceil(self.input_yaml['fwhm'])))
                # all_sources['x_pix'], all_sources['y_pix'] = centroid_sources(image, all_sources['x_pix'], all_sources['y_pix'],
                                                                              # footprint=footprint, centroid_func=centroid_2dg)
                
                oversample = 1
                epsf_builder = EPSFBuilder(oversampling=oversample,
                                           maxiters=1000,
                                           fitter=EPSFFitter(fit_boxsize=fitsize),
                                           recentering_boxsize=boxsize,
                                           norm_radius=aperture_radius,
                                           progress_bar=False,
                                            # smoothing_kernel= 'quartic',
                                            # center_accuracy=0.5,
                                           smoothing_kernel=None, 
                                           recentering_func=centroid_2dg,
                                           # footprint=footprint, centroid_func=centroid_2dg,
                                           sigma_clip=SigmaClip(sigma=3,
                                                                     cenfunc=np.nanmean,
                                                                     stdfunc='mad_std',
                                                                    ))
    
                                           
                try:
                    logger.info(f"\nBuilding ePSF with initial symmetric Moffat profile with FWHM: {self.input_yaml['fwhm']:.1f} px")
                    epsf, fitted_stars = epsf_builder.build_epsf(extracted_stars,init_epsf = epsf)
                except:
                    logger.info(f'\nBuilding ePSF failed - PSF likely assymetric - trying without initial Moffat profile\n')
                    epsf, fitted_stars = epsf_builder.build_epsf(extracted_stars,init_epsf = None)
                    

                remove_stars = []
                
                if len(fitted_stars) < 10 or 1: break
                logger.info(f'\n> Checking {len(fitted_stars)} stars for blended sources')
                from photutils.psf import SourceGrouper
                grouper = SourceGrouper(min_separation=self.input_yaml['fwhm'])
                        
                for i, star in enumerate(fitted_stars):
                  
             
                    try:
                    
                        star = star.data
              
                       
                        init_params = Table()
                        init_params['x'] = [star.shape[1]/2]
                        init_params['y'] = [star.shape[0]/2]
                        
                        threshold_value = np.nanmean(detect_threshold(star, nsigma = 3.5) )
                        finder = IRAFStarFinder(fwhm      = self.input_yaml['fwhm'],
                                                threshold = threshold_value,
                                                sharplo   =  0.1, sharphi = 1,
                                                roundlo=-0.3, roundhi=0.3, 
                                                exclude_border = True,
                                                
                                                
                                                )
                        
                        psfphot2 = IterativePSFPhotometry(epsf, star_shape, 
                                                          finder=finder,
                                                          localbkg_estimator=localbkg_estimator,
                                                          aperture_radius=aperture_radius,
                                                          maxiters=2,
                                                          grouper=grouper)
                        
                        
                        
                        
                        phot = psfphot2(star, init_params=init_params)
                        
                        if len(phot)==1: continue
                        remove_stars.append(i)
                    except:
                        remove_stars.append(i)
                        
                    
                        
                if len(remove_stars)==0 or (counter>max_check): break
                    
                logger.info(f'> Removing {len(remove_stars)} stars and rebuilding PSF')
                for i in remove_stars: psfSources.drop(index = i,inplace = True)


            logger.info(f'> {len(fitted_stars )} sources fitted to build ePSF')
                        
            # =============================================================================
            #             
            # =============================================================================
            
            from photutils.aperture import CircularAperture

            epsf_data = epsf.data
            psf_model =      epsf_data
            new_radius = aperture_radius + np.ceil(0.5*self.input_yaml['fwhm'])

            
            if 1:
                stars = extracted_stars
   
                # Number of rows and columns for the grid
                nrows = 3
                ncols = 2
                plt.ioff()
                # Create a figure with a GridSpec layout to allow for the right image to be larger and centered
                fig = plt.figure(figsize=set_size(540,1))
                grid = fig.add_gridspec(nrows=nrows, ncols=ncols + 2, 
                                        width_ratios=[1, 1, 0.5, 0.5], 
                                        height_ratios=[1, 1, 1])

                # Plot the 3x2 grid of images
                
                for i in range(nrows):
                    for j in range(ncols):
                        ax = fig.add_subplot(grid[i, j])
                        try:
                            # Normalization using ZScaleInterval
                            interval = ZScaleInterval()
                            vmin, vmax = interval.get_limits(stars[i * ncols + j])
                            norm = ImageNormalize(vmin=vmin, vmax=vmax)
                            # Display the star image in the grid
                            ax.imshow(stars[i * ncols + j], origin='lower', cmap='viridis', norm=norm,interpolation='nearest')
                            ax.set_title(f'Source {i * ncols + j + 1} / {len(stars)}')
                            # ax.axis('off')
                        except:
                            plt.delaxes(ax)

                # Plot the additional image to the right of the 3x2 grid, centered vertically and larger
                ax_right = fig.add_subplot(grid[0:3, 2:4])  # Span 3 rows and 2 columns to center it vertically
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(    psf_model[abs(psf_model)>1e-6] )
                norm = ImageNormalize(vmin=vmin, vmax=vmax)
                ax_right.imshow(    psf_model , origin='lower', cmap='viridis',norm = norm,interpolation='nearest')
                
                ax_right.set_title('ePSF model')
                
                center = epsf.origin
                # Create circles for aperture and annulus
                circle = plt.Circle(center, radius=new_radius, color='red', ls='-', fill=False, label='aperture', lw=1.5)
                ax_right.add_patch(circle)


                plt.tight_layout()
                
                
                fig.savefig(   os.path.join(write_dir, 'PSF_sources_'+base+'.pdf'), bbox_inches='tight')
                plt.close()
                
            # =============================================================================
            #                 
            # =============================================================================
            
            center = epsf.origin
            # Define the aperture
            aperture = CircularAperture(center, r= new_radius)

            # Determine which points are within the aperture
            aperture_mask = aperture.to_mask(method='subpixel', subpixels=5)

            # Apply the mask to the grid of coordinates
            aperture_mask_data = aperture_mask.to_image(epsf_data.shape).astype(bool)

            # Apply the mask to the ePSF data array
            epsf_data[~aperture_mask_data] = 0
            
            psf_model =      epsf_data
            
            
            epsf = EPSFModel(data=psf_model, origin=epsf.origin,norm_radius=aperture_radius)
            
            
            hdu = fits.PrimaryHDU(psf_model)

            hdul = fits.HDUList([hdu])
            psf_model_savepath = os.path.join(
                write_dir, 'PSF_model_'+base+'.fits')
            hdul.writeto(psf_model_savepath,
                         overwrite=True)

            logger.info(f'\n> PSF model created with shape: {psf_model.shape[0]} {psf_model.shape[1]} using {len(fitted_stars)} sources')
                      
            logger.info('> PSF model saved as: %s' % psf_model_savepath)

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno, e)
            return None, None
        
        
        # sys.exit()
        return epsf, psfSources


# =============================================================================
# 
# =============================================================================


    def fit(self, epsf_model, sources, plot=False, plotTarget=False,forcePhotometry = False,ignore_sources = None):

        import numpy as np
        from lmfit import Model
        from background import background
        import warnings
        import logging
        from functions import SNR, SNR_err
        from functions import border_msg
        from functions import pix_dist
        from photutils.aperture import CircularAperture

        import pandas as pd
        import numpy as np
        import logging
        import warnings
        from lmfit import Model
        from functions import border_msg, pix_dist, scale_roll, rebin,SNR_err

        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats

        from astropy.io import fits
        import os
        import sys
        from astropy.modeling import fitting
        from background import background

        from photutils.psf.epsf_stars import EPSFStar, EPSFStars
        from photutils.psf import EPSFBuilder, EPSFFitter

        logger = logging.getLogger(__name__)

        if not plotTarget:
            logger.info(border_msg(
                'Fitting PSF to %d sources in the field' % len(sources)))
        else:
            logger.info(border_msg( f'Fitting PSF to {self.input_yaml["target_name"]}' ))

        scale = self.input_yaml['scale']
        fwhm =     self.input_yaml['fwhm']
        
        
        if (epsf_model is None):
            
            logger.info('ePSF model not avilable - skipping')

            return sources
            
        try:

            from photutils.detection import DAOStarFinder,IRAFStarFinder
            from photutils.psf import PSFPhotometry,IterativePSFPhotometry
            
            
            if not (ignore_sources is None) and 0:

                
                # Create a boolean mask to filter out sources that fall within the masked region
                def is_within_tolerance(point, others, tolerance=10):
                    import numpy as np
                    distances = np.sqrt((others['x_pix'] - point[0])**2 + (others['y_pix'] - point[1])**2)
                    # distances = distances[distances!=0]
                    return np.any(distances <= tolerance)# Out of bounds coordinates are considered not in mask
                
                # Apply the filter to the DataFrame
                sources = sources[~sources.apply(lambda row: is_within_tolerance([row['x_pix'], row['y_pix']], ignore_sources,tolerance = 2 *fwhm), axis=1)]
                
                                


            
            aperture_radius = self.input_yaml['photometry']['ap_size'] * \
                self.input_yaml['fwhm']

            
            fit_shape = int(np.ceil(aperture_radius))
            fit_shape = int(fit_shape if fit_shape % 2 != 0 else fit_shape+1)
      
        
            epsf_model_copy = epsf_model.copy()

            logger.info(f'Fitting PSF within {fit_shape} pixels of inital position')

            from photutils.background import LocalBackground, MMMBackground, MedianBackground
            bkgstat = MMMBackground()
            localbkg_estimator = LocalBackground(aperture_radius +  np.ceil(self.input_yaml['fwhm']),
                                                  aperture_radius + np.ceil(self.input_yaml['fwhm'])+5,
                                                   MedianBackground()
                                                  )
            
            # localbkg_estimator = 
            from photutils.segmentation import detect_threshold
            image_copy = self.image.copy()
            threshold_value = np.nanmean(detect_threshold(image_copy, nsigma = 3) )
            finder = IRAFStarFinder(fwhm      = self.input_yaml['fwhm'],
                                    threshold = threshold_value,
                                    # sharplo   =  0.3, sharphi = 1,
                                    # roundlo=0, roundhi=0.2, 
                                    # exclude_border = True,
                                    xycoords = list(zip(sources['x_pix'],sources['y_pix']))
                                    )
                                    # min_sep_fwhm = 


            if forcePhotometry:
                logger.info('Perfroming forced photometry - fixing PSF coordinates')
                epsf_model_copy.x_0.fixed = True
                epsf_model_copy.y_0.fixed = True
                
                
            
            
            psfphot = PSFPhotometry(psf_model=epsf_model_copy,
                                    localbkg_estimator=localbkg_estimator,
                                    fitter=fitting.LevMarLSQFitter(),
                                    fit_shape =   fit_shape,
                                    finder = finder,
                                    aperture_radius=aperture_radius,
                                    )
            

            from astropy.table import Table
            
            init_params = Table()
            init_params['x'] = sources['x_pix'].values
            init_params['y'] = sources['y_pix'].values


            tbl = psfphot(image_copy,  init_params=init_params)
            names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
            output_table = tbl[names].to_pandas()
            
          


            with np.errstate(divide='ignore', invalid='ignore'):
                
                sources['x_pix'] = output_table['x_fit'].values
                sources['y_pix'] = output_table['y_fit'].values
                
                
                s2n = output_table['flux_fit'].values / \
                      output_table['flux_err'].values
                    
                s2n[s2n<0] = 1e-6
                
                
                # TDOO: Update code 
                # flux = output_table['flux_fit'].values
                
                # 
                # flux[flux<0] = 0
                # output_table['flux_fit'] = flux
    
   
    
                # print(output_table['flux_fit'])
                sources['flux_PSF'] = output_table['flux_fit'].values / \
                    self.input_yaml['exptime']
                    
                sources['inst_%s_PSF' % (self.input_yaml['imageFilter'])] = -2.5*np.log10(sources['flux_PSF'])
                
                sources['inst_%s_PSF_err' % (self.input_yaml['imageFilter'])] = SNR_err(s2n)
    
            if plot or plotTarget:
                from functions import set_size
                import matplotlib.pyplot as plt
                from astropy.nddata import Cutout2D
                from astropy import units as u
                
                from matplotlib.gridspec import GridSpec
                from astropy.visualization import (
                    ImageNormalize, LinearStretch, ZScaleInterval)

                dir_path = os.path.dirname(os.path.realpath(__file__))
                plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))

                fpath = self.input_yaml['fpath']
                base = os.path.basename(fpath)
                write_dir = os.path.dirname(fpath)
                base = os.path.splitext(base)[0]

                subtractedImage = psfphot.make_residual_image(image_copy,  (scale*2,scale*2))

                if plotTarget and len(sources) == 1:

                   
                    # Define the position and size for the cutouts
                    position = (sources['x_pix'].values[0], sources['y_pix'].values[0])
                    size = (scale*2,scale*2) * u.pixel
                    
                    # Create the cutout for the subtracted image
                    subtractedImage = Cutout2D(subtractedImage, position, size,mode = 'partial',fill_value = 1e-30).data
                    
                    # Create the cutout for the plot image
                    plotImage = Cutout2D(image_copy, position, size,mode = 'partial',fill_value = 1e-30).data
                    

 
                else:
                    plotImage = image_copy

                bestFit = plotImage - subtractedImage

                norm = ImageNormalize(plotImage,
                                      interval=ZScaleInterval(),
                                      stretch=LinearStretch())
                if plotTarget:
                    heights = [1, 1, 0.75]
                    widths = [1, 1, 0.75, 1, 1, 0.75]

                    ncols = 6
                    nrows = 3

                else:

                    widths = [1, 1]
                    heights = [1]

                    ncols = 2
                    nrows = 1

                plt.ioff()
                fig = plt.figure(figsize=set_size(540, 0.75))

                grid = GridSpec(nrows, ncols, wspace=0.5, hspace=0.5,
                                height_ratios=heights,
                                width_ratios=widths)

                if plotTarget:

                    ax1 = fig.add_subplot(grid[0:2, 0:2])

                    ax2 = fig.add_subplot(grid[0:2, 3:5])

                    ax1_B = fig.add_subplot(grid[2, 0:2])
                    ax1_R = fig.add_subplot(grid[0:2, 2])

                    ax2_B = fig.add_subplot(grid[2, 3:5])
                    ax2_R = fig.add_subplot(grid[0:2, 5])

                    ax1_R.yaxis.tick_right()
                    ax2_R.yaxis.tick_right()

                    ax1_R.xaxis.tick_top()
                    ax2_R.xaxis.tick_top()

                    bbox = ax1_R.get_position()
                    offset = -0.03
                    ax1_R.set_position(
                        [bbox.x0 + offset, bbox.y0, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox = ax2_R.get_position()
                    offset = -0.03
                    ax2_R.set_position(
                        [bbox.x0 + offset, bbox.y0, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox = ax1_B.get_position()
                    offset = 0.06
                    ax1_B.set_position(
                        [bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox = ax2_B.get_position()
                    offset = 0.06
                    ax2_B.set_position(
                        [bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                else:

                    ax1 = fig.add_subplot(grid[0])

                    ax2 = fig.add_subplot(grid[1])

                ax1.xaxis.tick_top()
                ax2.xaxis.tick_top()

                ax2.axes.yaxis.set_ticklabels([])

                ax1.imshow(plotImage,
                           origin='lower',
                           aspect='auto',
                           interpolation='none',
                           norm=norm)

                ax2.imshow(subtractedImage,
                           norm=norm,
                           origin='lower',
                           interpolation='none',
                           aspect='auto')

                if plotTarget:
                    t = np.arange(plotImage.shape[0])
                    f = np.arange(plotImage.shape[1])

                    hx, hy = plotImage.mean(0), plotImage.mean(1)

                    ax1_R.step(hy, t, color='blue')
                    ax1_B.step(f, hx, color='blue')

                    # ax1.scatter(sources['x_pix'] ,sources['y_pix'],marker = 'x',
                    #             color = 'red',zorder = 10,s = 25 )

                    # ax2.scatter(sources['x_pix'] ,sources['y_pix'],marker = 'x',
                    #             color = 'red',zorder = 10,s = 25 )

                    hx, hy = subtractedImage.mean(0), subtractedImage.mean(1)

                    ax2_R.step(hy, t, color='blue')
                    ax2_B.step(f, hx, color='blue')

                    # hx, hy = bkg_surface.mean(0), bkg_surface.mean(1)

                    # ax1_R.step(hy,t,color = 'green')
                    # ax1_B.step(f,hx,color = 'green')

                    # ax2_R.step(hy,t,color = 'green')
                    # ax2_B.step(f,hx,color = 'green')

                    hx, hy = bestFit.mean(0), bestFit.mean(1)

                    ax1_R.step(hy, t, color='red')
                    ax1_B.step(f, hx, color='red')

                    ax2_R.step(hy, t, color='red')
                    ax2_B.step(f, hx, color='red')

                    # ax1_B.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
                    ax1_B.set_xlim(ax1.get_xlim()[0], ax1.get_xlim()[1])

                    ax1_R.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])

                    ax2_B.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])

                    ax2_R.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])


                if not plotTarget:

                    save_loc = os.path.join(
                        write_dir, 'psfSubtractions_'+base+'.pdf')

                    fig.savefig(save_loc, bbox_inches='tight')

                else:

                    save_loc = os.path.join(
                        write_dir, 'targetPSF_'+base+'.pdf')

                    fig.savefig(save_loc, bbox_inches='tight')

                plt.close()

        except Exception as e:
            import os
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, e)
            return None

        return sources
