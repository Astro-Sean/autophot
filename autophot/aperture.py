#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:39:35 2022

@author: seanbrennan
"""

class aperture():
    
    def __init__(self,input_yaml,image):
        
        self.input_yaml = input_yaml
        self.image = image

        
    def measure(
        self,
        sources,
        ap_size=None,
        remove_background=True,
        return_background_surface=False,
        plot=False,
        saveTarget=False,
        ):
        """
        Perform aperture photometry on sources within an astronomical image.
    
        Parameters:
        - sources: pandas DataFrame containing 'x_pix' and 'y_pix' of source positions
        - ap_size: float, optional, size of the aperture for photometry; if None, defaults to a value based on FWHM
        - remove_background: bool, optional, whether to subtract background
        - return_background_surface: bool, optional, if True, return background surface array
        - plot: bool, optional, whether to generate plots for visual inspection
        - saveTarget: bool, optional, if True, save plots with a specified target name
        
        Returns:
        - sources: pandas DataFrame with photometry results added
        - background_surfaces: numpy array (optional), background surfaces if requested
        """
        
        # Import necessary libraries
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
        from astropy.nddata import Cutout2D
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import  GridSpec
        from functions import SNR, mag, SNR_err, set_size  # Custom functions
        from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
        import os,sys
        import pathlib
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Suppress chained assignment warnings in pandas
        pd.options.mode.chained_assignment = None
    
        try:
            # Extract parameters from the input YAML
            fwhm = self.input_yaml['fwhm']
            expTime = self.input_yaml['exptime']
        
            # Determine aperture size based on input or configuration
            if ap_size is None:
                ap_size = self.input_yaml['photometry']['ap_size'] * fwhm
            
            # Set annulus inner and outer radii for background estimation
            annulusIN = np.ceil(ap_size + fwhm)
            annulusOUT = np.ceil(annulusIN + 5)
            
            # Set aperture scale size (cutout image size)
            aperture_scale = int(annulusOUT + fwhm)
            if aperture_scale % 2 != 0:
                aperture_scale += 1
            
            cutout_center = (aperture_scale, aperture_scale)
    
            # Initialize list to store background surfaces if needed
            background_surfaces = []
    
            # Area of aperture used for flux calculations
            area = np.pi * ap_size ** 2
            
            # Create a copy of the image for processing
            image_copy = self.image.copy()
            positionSources = sources[['x_pix', 'y_pix']]
    
            # Iterate over each source position for photometry
            for index, row in positionSources.iterrows():
                xpos = float(np.floor(row['x_pix'])-0.5)
                ypos = float(np.floor(row['y_pix'])-0.5)
                
                position = (xpos, ypos)
                size = 2 * aperture_scale  # Diameter of the aperture
                
                # Create a cutout of the image around the source
                cutout = Cutout2D(image_copy, position, size, mode='partial', fill_value=0).data
    
                # Define an annulus for background estimation
                annulus_masks = CircularAnnulus(cutout_center, r_in=annulusIN, r_out=annulusOUT).to_mask(method='center')
                annulus_mask_array = annulus_masks.to_image(shape=cutout.shape).astype(bool)
                
                annulus = cutout.copy()
                annulus[~annulus_mask_array] = np.nan
    
                # Calculate background statistics using sigma-clipped stats
                mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus, cenfunc=np.nanmean, stdfunc=np.nanstd)
                
                # Define a smaller aperture mask to find the maximum pixel value in the source
                masks = CircularAperture(cutout_center, r=np.ceil(ap_size) * 0.33).to_mask(method='center')
                mask_array = masks.to_image(shape=cutout.shape).astype(bool)
                
                aperture_center = cutout.copy()
                aperture_center[~mask_array] = np.nan
                
                max_pixel_value = np.nanmax(aperture_center) - mean_sigclip
                
                # Perform aperture photometry
                apertures = CircularAperture(cutout_center, r=ap_size)
                phot = aperture_photometry(cutout, apertures).to_pandas()
                phot['aperture_bkg'] = median_sigclip * area
                phot['aperture_sum_bkgsub'] = phot['aperture_sum'] - phot['aperture_bkg']
    
                # Extract photometry results and compute SNR
                aperture_sum = float(phot['aperture_sum_bkgsub'].values[0])
                sources.loc[index, 'maxPixel'] = max_pixel_value / expTime
                sources.loc[index, 'flux_AP'] = aperture_sum / expTime
                sources.loc[index, 'fluxSky_AP'] = phot['aperture_bkg'].values[0] / expTime
                sources.loc[index, 'noiseSky'] = std_sigclip / expTime
                sources.loc[index, 'SNR'] = float(SNR(sources.loc[index, 'maxPixel'], sources.loc[index, 'noiseSky']))
                
                # Calculate magnitudes and errors
                sources.loc[index, 'inst_%s_AP' % self.input_yaml['imageFilter']] = mag(sources.at[index, 'flux_AP'])
                sources.loc[index, 'inst_%s_AP_err' % self.input_yaml['imageFilter']] = SNR_err(sources.at[index, 'SNR'])
                 
                # Plotting section, if enabled
                if plot:
                    plt.ioff()  # Disable interactive mode for plotting
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))
                    
                    fpath = self.input_yaml['fpath']
                    base = os.path.basename(fpath)
                    write_dir = os.path.dirname(fpath)
                    base = os.path.splitext(base)[0]
    
                    norm = ImageNormalize(cutout, interval=ZScaleInterval(), stretch=LinearStretch())
                    
                    heights = [1, 1, 0.75]
                    widths = [1, 1, 0.75]
                    ncols = 3
                    nrows = 3
                    
                    fig = plt.figure(figsize=set_size(330, 1))
                    grid = GridSpec(nrows, ncols, wspace=0.5, hspace=0.5, height_ratios=heights, width_ratios=widths)
                    
                    ax1 = fig.add_subplot(grid[0:2, 0:2])
                    ax1_B = fig.add_subplot(grid[2, 0:2])
                    ax1_R = fig.add_subplot(grid[0:2, 2])
                    
                    ax1_R.yaxis.tick_right()
                    ax1_R.xaxis.tick_top()
                    ax1.xaxis.tick_top()
                    
                    # Adjust subplot positions
                    bbox = ax1_R.get_position()
                    offset = -0.03
                    ax1_R.set_position([bbox.x0 + offset, bbox.y0, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    
                    bbox = ax1_B.get_position()
                    offset = 0.06
                    ax1_B.set_position([bbox.x0, bbox.y0 + offset, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0])
    
                    ax1.imshow(cutout, origin='lower', aspect='auto', norm=norm)
                    
                    t = np.arange(cutout.shape[0])
                    f = np.arange(cutout.shape[1])
                    hx, hy = cutout.mean(0), cutout.mean(1)
                    
                    ax1_R.step(hy, t, color='blue')
                    ax1_B.step(f, hx, color='blue')
                    
                    # Plot center lines and circles for aperture and annulus
                    center = (cutout.shape[1] / 2, cutout.shape[0] / 2)
                    ax1.axvline(center[0], ls='--', color='red')
                    ax1_B.axvline(center[0], ls='--', color='red')
                    ax1.axhline(center[0], ls='--', color='red')
                    ax1_R.axhline(center[0], ls='--', color='red')
                    
                    # Create circles for aperture and annulus
                    circle = plt.Circle(center, radius=ap_size, color='red', ls='-', fill=False, label='aperture', lw=1.5)
                    ax1.add_patch(circle)
                    
                    circle = plt.Circle(center, radius=annulusIN, color='red', ls='--', fill=False, label='annulus', lw=1.5)
                    ax1.add_patch(circle)
                    
                    circle = plt.Circle(center, radius=annulusOUT, color='red', ls='--', fill=False, lw=1.5)
                    ax1.add_patch(circle)
    
                    # Save the plot
                    if not saveTarget:
                        saveDir = os.path.join(write_dir, 'aperture')
                        pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True)
                        saveName = os.path.join(saveDir, 'aperture_%d.pdf' % index)
                        plt.savefig(saveName, bbox_inches='tight')
                    else:
                        save_loc = os.path.join(write_dir, 'aperture_' + base + '.pdf')
                        fig.savefig(save_loc)
                    
                    plt.close()
    
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno, e)
            plt.close('all')
    
        # Return results
        if return_background_surface:
            return sources, np.array(background_surfaces)
        else:
            return sources




# =============================================================================
#     
# =============================================================================
    def measure_optimum_radius(self, sources, nSamples=10, plot=False, norm_factor=0.90):
        """
        Determine the optimum aperture radius size for photometry by analyzing the 
        signal-to-noise ratio (SNR) across various aperture sizes.
    
        Parameters:
        - sources: pandas DataFrame containing source information, including SNR and positions.
        - nSamples: int, number of top sources to consider for analysis. Default is 25.
        - plot: bool, if True, generate a plot showing the SNR distribution and optimal radius.
        - norm_factor: float, the normalization factor used to determine the optimal radius.
    
        Returns:
        - optimum_radius: float, the optimal aperture radius in units of FWHM.
        """
        
        # Import necessary libraries
        import os,warnings
        import numpy as np
        from functions import SNR, SNR_err, norm, border_msg,set_size
        from astropy.stats import sigma_clip
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import logging
        import matplotlib as mpl
        import matplotlib.gridspec as gridspec
    
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.info(border_msg(f'Finding optimum radius size using {nSamples} sources'))
        
        # Suppress chained assignment warnings in pandas
        pd.options.mode.chained_assignment = None
        plt.ioff()  # Turn off interactive plotting
    
        try:
            step_size = 0.1
            search_size = np.arange(0.1, 6.1, step_size)
    
            # Sort sources by SNR and filter those with SNR > 10
            sources.sort_values(by=['SNR'], ascending=False, inplace=True)
            sources = sources[sources['SNR'] > 20]
            
            # Apply sigma clipping to filter sources based on SNR
            s2n_mask = sigma_clip(
                sources['SNR'].values,
                sigma_lower=3,
                sigma_upper=5,
                masked=True,
                cenfunc=np.nanmean,
                stdfunc='mad_std'
            )
            
            if sum(s2n_mask.mask)>0:
                # Keep only unmasked sources
                sources = sources[~s2n_mask.mask]
    
            # Select top nSamples sources for further analysis
            if len(sources) > nSamples:
                selected_sources = sources.head(nSamples)
            else:
                selected_sources = sources
    
            fwhm = self.input_yaml['fwhm']
            positions = list(zip(np.array(sources.x_pix), np.array(sources.y_pix)))
            output = []
    
            # Iterate over potential aperture sizes and measure SNR for each
            for s in search_size:
                sources_i = self.measure(sources=selected_sources, ap_size=s * fwhm)
    
                SNR_i = sources_i['SNR'].values
                starFlux = sources_i['flux_AP'].values
                noiseFlux = sources_i['noiseSky'].values
                SNR_i_err = SNR_err(SNR_i)
    
                output.append([[s] * len(positions), list(SNR_i), list(SNR_i_err), list(starFlux), list(noiseFlux)])
    
            # Calculate normalized SNR distribution to find the optimum aperture size
            sum_distribution = []
    
            for j in range(len(search_size)):
                steps = []
                for i in range(len(SNR_i)):
                    steps.append(norm(np.array([j[3][i] for j in output]), norm2one=True)[j])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    filtered_stepsize = sigma_clip(steps, sigma=3, masked=False, cenfunc=np.nanmean, stdfunc='mad_std')
                   
                    sum_distribution.append(np.nanmean(filtered_stepsize))
    
            sum_distribution_norm = norm(np.array(sum_distribution), norm2one=True)
            sum_distribution_max_idx = np.argmax((sum_distribution_norm > norm_factor))
    
            optimum_radius = search_size[sum_distribution_max_idx]
    
            logger.info(f'Optimal Aperture: {optimum_radius:.1f} x FWHM [pixels]')
    
            # Plotting the SNR distribution if requested
            if plot:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))
                save_loc = os.path.join(self.input_yaml['write_dir'], f'optimum_aperture_{self.input_yaml["base"]}.pdf')
    
                max_SNRs = [np.nanmax([j[1][i] for j in output]) for i in range(len(SNR_i))]
                cmap = plt.get_cmap('rainbow')
                bounds = np.linspace(np.nanmin(max_SNRs), np.nanmax(max_SNRs), len(max_SNRs) + 1, endpoint=True).astype(int)
                norm_cb = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
                fig = plt.figure(figsize=set_size(330, 1))
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                im = cm.ScalarMappable(norm=norm_cb, cmap=cmap)
    
                for i in range(len(SNR_i)):
                    max_SNR = np.nanmax([j[1][i] for j in output])
                    ax1.plot([j[0][i] for j in output], [j[1][i] for j in output] / max_SNR, ls='--', alpha=0.25, color=cmap(norm_cb(max_SNR)))
                    ax2.plot([j[0][i] for j in output], norm([j[3][i] for j in output], norm2one=True), ls='--', alpha=0.25, color=cmap(norm_cb(max_SNR)))
                    ax2.plot([j[0][i] for j in output], sum_distribution_norm, ls='-', alpha=0.25, color='black')
    
                for ax in [ax1, ax2]:
                    ax.axvline(optimum_radius, ls=':', color='b', label='Optimum radius')
    
                ax2.set_xlabel('Aperture Size [FWHMs]')
                ax1.set_ylabel('SNR (normalized)')
                ax2.set_ylabel('Flux (normalized)')
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax1.set_ylim(-0.01, 1.05)
                ax1.set_xlim(0, search_size.max() + 0.1)
    
                clb = fig.colorbar(im, ax=fig.axes)
                clb.ax.set_ylabel(r'$Max~S/N$', rotation=270, labelpad=5)
    
                # handles, labels = ax1.get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # ax1.legend(by_label.values(), by_label.keys(), frameon=False, loc='lower right')
    
                handles, labels = ax2.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax2.legend(by_label.values(), by_label.keys(), frameon=False, loc='lower right')
    
                fig.savefig(save_loc, format='pdf', bbox_inches='tight')
                plt.close(fig)
    
        except Exception as e:
            # Error handling and logging
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.info(exc_type, fname, exc_tb.tb_lineno, e)
            optimum_radius = 1.5  # Default fallback value in case of error
            plt.close('all')
    
        return round(optimum_radius, 1)

    
# =============================================================================
#     
# =============================================================================
    
    def measure_aperture_correction(self, sources, nSamples=25, fwhm=None, apSize=None, apInfinite=None, plot=False):
        """
        Measure the aperture correction by comparing flux measurements in different aperture sizes.
    
        Parameters:
        - sources: pandas DataFrame containing source information.
        - nSamples: int, number of sources to sample for correction calculation. Default is 25.
        - fwhm: float, full-width at half maximum for the PSF. If None, will use the value from input YAML.
        - apSize: float, aperture size in pixels. If None, will use the value from input YAML scaled by FWHM.
        - apInfinite: float, a large aperture size for flux measurement to represent 'infinite' aperture. 
                      If None, will default to 1.1 times apSize.
        - plot: bool, if True, generate a plot of the aperture correction.
    
        Returns:
        - correction: float, the aperture correction factor in magnitudes.
        - correction_err: float, the error of the aperture correction.
        """
        
        # Import necessary libraries
        import os
        import logging
        import numpy as np
        from astropy.stats import sigma_clip, mad_std
        from functions import mag, border_msg
    
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Finding aperture correction'))
    
        # Sort sources by flux in ascending order for selection
        sources.sort_values(by=['flux_AP'], ascending=False, inplace=True)
    
        # Use FWHM from input YAML if not provided
        if not fwhm:
            fwhm = self.input_yaml['fwhm']
    
        # Set default aperture sizes if not provided
        if not apSize:
            apSize = self.input_yaml['photometry']['ap_size'] * fwhm
        if not apInfinite:
            apInfinite = apSize * 1.1
    
        # Select top nSamples sources or all if fewer available
        selected_sources = sources.head(nSamples) if len(sources) > nSamples else sources
    
        # Measure flux in the normal aperture size
        sources_normal = self.measure(sources=selected_sources, ap_size=apSize)
        flux_apSize = sources_normal['flux_AP'].values.copy()
    
        # Measure flux in a large (near-infinite) aperture size
        sources_large = self.measure(sources=selected_sources, ap_size=apInfinite)
        flux_apInfinite = sources_large['flux_AP'].values.copy()
    
        # Calculate the aperture difference in magnitudes
        apertureDifference = mag(flux_apInfinite / flux_apSize)
    
        # Apply sigma clipping to remove outliers from the aperture difference
        mask = sigma_clip(apertureDifference, sigma=3, masked=True, cenfunc=np.nanmedian, stdfunc=mad_std)
        apertureDifference = apertureDifference[~mask.mask]
    
        # Calculate correction and its error
        correction = np.nanmedian(apertureDifference)
        correction_err = np.nanstd(apertureDifference)
    
        logger.info(f'Aperture correction: {correction:.3f} +/- {correction_err:.3f}')
    
        # Plot the aperture correction if requested
        if plot:
            import matplotlib.pyplot as plt
            from functions import set_size
    
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))
    
            save_loc = os.path.join(self.input_yaml['write_dir'], f'aperture_correction_{self.input_yaml["base"]}.pdf')
    
            plt.ioff()
            fig = plt.figure(figsize=set_size(330, 1))
            ax1 = fig.add_subplot(111)
    
            # Plot the histogram of aperture differences
            ax1.hist(apertureDifference, bins='auto', color='blue', histtype='step', label='Aperture Correction')
    
            ax1.set_xlabel(r'Correction [ mag ]')
            ax1.set_ylabel('N')
            ax1.axvline(correction, color='black')
            ax1.legend(loc='best', frameon=False)
    
            fig.tight_layout()
            fig.savefig(save_loc, format='pdf', bbox_inches='tight')
            plt.close()
    
        return correction, correction_err
