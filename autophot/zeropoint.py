 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:45:49 2022

@author: seanbrennan
"""

class zeropoint:
    """
    Zeropoint class for cleaning up sequence stars based on certain criteria.
    """

    def __init__(self, input_yaml):
        """
        Initialize the Zeropoint class.

        Parameters:
        - input_yaml: Dictionary or YAML object containing configuration data.
        """
        self.input_yaml = input_yaml

    def clean(self, sources, upperMaglimit=13, lowerMaglimit=21, limitSNR=5):
        """
        Clean the input source catalog based on magnitude and signal-to-noise ratio (SNR) criteria.

        Parameters:
        - sources: pandas DataFrame containing source information.
        - upperMaglimit: float, upper magnitude limit to filter out bright stars.
        - lowerMaglimit: float, lower magnitude limit to filter out faint stars.
        - limitSNR: float, minimum signal-to-noise ratio to keep sources.

        Returns:
        - cleanedSources: pandas DataFrame with sources that meet all criteria.
        """
        import numpy as np
        import logging
        from functions import border_msg

        # Set up logger
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Cleaning up sequence stars for zeropoint'))

        try:
            # Filter sources with no filter information
            filter_col = f"{self.input_yaml['imageFilter']}"
            no_filter = np.isnan(sources[filter_col])

            if no_filter.sum() > 0:
                logger.info(f'Removing {no_filter.sum()} sources with no filter information')
            
            cleaned_sources = sources[~no_filter]

            # Remove sources fainter than the lower magnitude limit
            too_faint = cleaned_sources[filter_col] > lowerMaglimit
            if too_faint.sum() > 0:
                logger.info(f'Removing {too_faint.sum()} sources fainter than {lowerMaglimit:.1f} mag')
                cleaned_sources = cleaned_sources[~too_faint]

            # Remove sources brighter than the upper magnitude limit
            too_bright = cleaned_sources[filter_col] < upperMaglimit
            if too_bright.sum() > 0:
                logger.info(f'Removing {too_bright.sum()} sources brighter than {upperMaglimit:.1f} mag')
                cleaned_sources = cleaned_sources[~too_bright]

            # Remove sources with SNR lower than the limit
            low_snr = cleaned_sources['SNR'].values < limitSNR
            if low_snr.sum() > 0:
                logger.info(f'Removing {low_snr.sum()} sources with S/N lower than {limitSNR:.1f}')
                cleaned_sources = cleaned_sources[~low_snr]

        except Exception as e:
            # Handle exceptions and print traceback information
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Exception: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
            return None

        return cleaned_sources

        
    
# =============================================================================
#     
# =============================================================================
    def get(self, sources, useMean=True, useMedian=False):
        """
        Measures the zeropoint offset for the given sources.
    
        Parameters:
        - sources (DataFrame): DataFrame containing source data.
        - useMean (bool): Whether to use the mean for calculating the zeropoint.
        - useMedian (bool): Whether to use the median for calculating the zeropoint.
        
        Returns:
        - updatedsources (DataFrame): DataFrame with zeropoint information updated.
        - outputZP (dict): Dictionary containing zeropoint offsets and their errors for different methods.
        """
        import numpy as np
        import logging
        from functions import border_msg, SNR_err
        from astropy.stats import mad_std, sigma_clip
    
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Measuring zeropoint offset'))
    
        try:
            # Determine methods to calculate zeropoint
            methods = ['AP']
            if 'flux_PSF' in sources.columns:
                methods.append('PSF')
    
            method_labels = {'AP': 'Aperture', 'PSF': 'PSF'}
            updated_sources = sources.copy()
            output_zp = {}
    
            for method in methods:
                output_zp[method] = []
    
                # Calculate zeropoint and error
                method_zp = sources[f'{self.input_yaml["imageFilter"]}'] - sources[f'inst_{self.input_yaml["imageFilter"]}_{method}']
                error_snr = SNR_err(sources['SNR'])
                method_zp_err = np.sqrt(error_snr**2 + sources[f'{self.input_yaml["imageFilter"]}_err'])
    
                updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'] = method_zp
                updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}_err'] = method_zp_err
    
                # Remove non-finite values (NaN or Inf)
                nan_mask = np.isfinite(updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'])
                if np.sum(~nan_mask) > 0:
                    logger.info(f'[{method_labels[method]} zeropoint] Removing {sum(~nan_mask)} non-finite (NaN or inf) sources')
                    updated_sources = updated_sources.loc[nan_mask]
    
                # Apply sigma clipping to filter outliers
                mask_zp = sigma_clip(
                    updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'].values,
                    sigma=3,
                    masked=True,
                    maxiters=10,
                    cenfunc=np.nanmedian,
                    stdfunc=mad_std
                )
    
                if np.sum(mask_zp.mask) > 0:
                    logger.info(f'[{method_labels[method]} zeropoint] Removing {np.sum(mask_zp.mask)} sources with sigma clipping')
                    updated_sources = updated_sources.loc[~mask_zp.mask]
    
                # Calculate the zeropoint offset and its error
                image_zp = np.nanmean(updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'].values)
                image_zp_err = np.nanstd(updated_sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'].values)
    
                output_zp[method.replace('_', '')] = [image_zp, image_zp_err]
                logger.info(f'[{method_labels[method]} zeropoint] Measured offset: {image_zp:.3f} +/- {image_zp_err:.3f}\n')
    
        except Exception as e:
            # Handle exceptions and log error information
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error(f"Exception: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
    
        return updated_sources, output_zp
    
# =============================================================================
#     
# =============================================================================
    
    
    def plot_histogram(self, sources, aperture_correction=None, measured_zeropoint=None):
        """
        Plots histograms of the zeropoint for different methods and optionally applies aperture corrections.
    
        Parameters:
        - sources (DataFrame): DataFrame containing source data with zeropoint calculations.
        - aperture_correction (tuple, optional): A tuple containing aperture correction and its error.
        - measured_zeropoint (dict, optional): A dictionary of measured zeropoints for different methods.
        """
        from functions import set_size, calculate_bins
        import matplotlib.pyplot as plt
        import os
        import numpy as np
    
        try:
            # Set up the plotting style
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))
    
            # Extract file paths and settings from input YAML
            fpath = self.input_yaml['fpath']
            base = os.path.splitext(os.path.basename(fpath))[0]
            write_dir = os.path.dirname(fpath)
    
            plt.ioff()
            fig = plt.figure(figsize=set_size(330, 1))
            ax1 = fig.add_subplot(111)
    
            methods = ['AP']
            if 'flux_PSF' in sources.columns:
                methods.append('PSF')
    
            method_labels = {'AP': 'Aperture', 'PSF': 'PSF'}
            method_colors = {'AP': 'blue', 'PSF': 'red'}
            
            all_zp = []  # To keep track of all zeropoint values for x-axis limits
    
            # Plot histograms for each method
            for method in methods:
                zp_method = sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'].values
                all_zp.extend(list(zp_method))
                ax1.hist(
                    zp_method,
                    bins=calculate_bins(zp_method),
                    label=method_labels[method],
                    histtype='step',
                    lw=0.5,
                    color=method_colors[method]
                )
    
            # If aperture correction is provided, plot corrected zeropoint
            if aperture_correction is not None:
                method = 'AP'
                zp_corrected = sources[f'zp_{self.input_yaml["imageFilter"]}_{method}'].values - aperture_correction[0]
                all_zp.extend(list(zp_corrected))
                ax1.hist(
                    zp_corrected,
                    bins=calculate_bins(zp_corrected),
                    label=f'{method_labels[method]}\nw/ correction',
                    histtype='step',
                    ls='--',
                    lw=0.25,
                    color=method_colors[method]
                )
    
            # Determine x-axis limits based on zeropoint range
            xlims = np.nanpercentile(all_zp, [0.5, 99.5])
            ax1.set_xlim(xlims[0] - 0.25, xlims[1] + 0.25)
    
            # Final plot formatting
            ax1.legend(loc='best', frameon=False)
            ax1.set_ylabel('N')
            ax1.set_xlabel('Zeropoint [mag]')
            
            # Save the plot
            save_loc = os.path.join(write_dir, f'zeropoint_{base}.pdf')
            fig.savefig(save_loc, bbox_inches='tight')
            plt.close()
    
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Exception: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
            plt.close('all')
    
        return

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    