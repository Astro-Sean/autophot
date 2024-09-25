#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:13:16 2022

@author: seanbrennan
"""

class plot:
    def __init__(self, input_yaml):
        """
        Initialize the Plot class with input YAML configuration.

        Parameters:
        input_yaml (dict): Configuration dictionary with file paths and other settings.
        """
        self.input_yaml = input_yaml

    def subtractionCheck(self, image, ref, diff,inset_size = 21, mark_location=[], mask=None,aligned_sources = None):
        """
        Perform and visualize a subtraction check of astronomical images.

        Parameters:
        image (ndarray): Main image data.
        ref (ndarray): Reference image data.
        diff (ndarray): Difference image data.
        mark_location (list): Location [x, y] for the inset (default is an empty list).
        inset_size (int): Size of the inset around the marked location (default is 31).
        mask (ndarray or None): Optional mask to overlay on the images (default is None).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from functions import set_size
        from astropy.visualization import ZScaleInterval
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        # Set matplotlib style
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))

        # Extract file path information
        fpath = self.input_yaml['fpath']
        base = os.path.basename(fpath)
        write_dir = os.path.dirname(fpath)
        base = os.path.splitext(base)[0]

        # Apply ZScale normalization to the images
        interval = ZScaleInterval()
        image_zscaled = interval(image)
        ref_zscaled = interval(ref)
        diff_zscaled = interval(diff)

        # Get photometry radius
        radius = self.input_yaml['photometry']['ap_size'] * self.input_yaml['fwhm']

        # Create a figure with 3 subplots
        images = [image_zscaled, ref_zscaled, diff_zscaled]
        titles = ['Image', 'Reference', 'Difference']
        fig = plt.figure(figsize=set_size(540, 0.5), constrained_layout=True)
        gs = GridSpec(1, 3, figure=fig, wspace=0.1)

        # Plot each image with a color scale and add inset if required
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = fig.add_subplot(gs[i])
            im = ax.imshow(img, origin='lower', aspect='auto', cmap='gray')
            ax.set_title(title)

            if mark_location:
                x, y = mark_location
                ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')
                
                ax_inset.set_zorder(1000)

                # Plot the inset image
                ax_inset.imshow(img, origin='lower', aspect='auto', cmap='gray')
                ax_inset.set_xlim(x - inset_size, x + inset_size)
                ax_inset.set_ylim(y - inset_size, y + inset_size)
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                
                # Style the inset border
                ax_inset.spines['top'].set_color('red')
                ax_inset.spines['bottom'].set_color('red')
                ax_inset.spines['left'].set_color('red')
                ax_inset.spines['right'].set_color('red')
                ax_inset.spines['top'].set_linewidth(0.5)
                ax_inset.spines['bottom'].set_linewidth(0.5)
                ax_inset.spines['left'].set_linewidth(0.5)
                ax_inset.spines['right'].set_linewidth(0.5)

                # Mark the inset with a border
                mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="red", lw=0.5)
            
            # Set axis labels
            ax.set_xlabel('X [Pixel]')
            ax.set_ylabel('Y [Pixel]')

        # Overlay the mask if provided
        if mask is not None:
            from matplotlib import colors
            red_transparent = colors.ListedColormap(['none', 'red'])
            for ax in fig.axes:
                ax.imshow(mask, cmap=red_transparent, alpha=0.5, origin='lower')
                

                

        # Save the figure to a PDF file
        save_loc = os.path.join(write_dir, 'subtraction_check_' + base + '.pdf')
        fig.savefig(save_loc, dpi=300)
        plt.close()

        return 1

    # =============================================================================
    #     
    # =============================================================================
        
    def SourceCheck(self, image,
                targetSources=None,
                psfSources=None, 
                catalogSources=None,
                fwhmSources=None,
                subtracted=False,
                mask=None):
        """
        Create a plot to check sources in astronomical images and overlay various source markers.
    
        Parameters:
        image (ndarray): Image data to be plotted.
        targetSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for target sources (optional).
        psfSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for PSF sources (optional).
        catalogSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for catalog sources (optional).
        fwhmSources (dict or None): Dictionary with 'x_pix' and 'y_pix' for FWHM sources (optional).
        subtracted (bool): Whether the image is subtracted (used for filename) (default is False).
        mask (ndarray or None): Optional mask to overlay on the image (default is None).
        """
        

        try:
            
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            from functions import set_size
            from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
            from matplotlib.patches import Circle, Rectangle
        
        
            # Set up matplotlib style
            dir_path = os.path.dirname(os.path.realpath(__file__))
            plt.style.use(os.path.join(dir_path, 'autophot.mplstyle'))
    
            # Extract file path and base name
            fpath = self.input_yaml['fpath']
            base = os.path.basename(fpath)
            write_dir = os.path.dirname(fpath)
            base = os.path.splitext(base)[0]
    
            # Get photometry radius and scale from input YAML
            radius = self.input_yaml['photometry']['ap_size'] * self.input_yaml['fwhm']
            scale = self.input_yaml['scale']
    
            # Create the figure
            plt.ioff()  # Turn interactive mode off
            fig = plt.figure(figsize=set_size(330, 1))
            ax1 = fig.add_subplot(111)
    
            # Normalize and plot the image
            norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=LinearStretch())
            ax1.imshow(image, origin='lower', aspect='auto', cmap='gray', interpolation=None, norm=norm)
    
            # Plot the target source as a circle
            circle = Circle(
                (self.input_yaml['target_x_pix'], self.input_yaml['target_y_pix']),
                radius, edgecolor='gold', facecolor='none', zorder=4, lw=0.33
            )
            ax1.add_patch(circle)
            ax1.text(
                self.input_yaml['target_x_pix'], self.input_yaml['target_y_pix'] + radius + 2,
                self.input_yaml['target_name'], color='gold', fontsize=3, ha='center', va='bottom'
            )
    
            # Plot PSF sources as squares
            if psfSources is not None:
                for x, y in zip(psfSources['x_pix'], psfSources['y_pix']):
                    lower_left_corner = (x - scale, y - scale)
                    square = Rectangle(
                        lower_left_corner, 2 * scale, 2 * scale, edgecolor='blue', facecolor='none',
                        label='PSF', zorder=0, lw=0.33
                    )
                    ax1.add_patch(square)
    
            # Plot catalog sources as circles (if not in HST mode)
            if catalogSources is not None and not self.input_yaml['HST_mode']:
                for x, y in zip(catalogSources['x_pix'], catalogSources['y_pix']):
                    circle = Circle(
                        (x, y), radius, edgecolor='red', facecolor='none', label='Catalog', 
                        zorder=1, lw=0.33
                    )
                    ax1.add_patch(circle)
    
            # Plot FWHM sources as crosses
            if fwhmSources is not None:
                for x, y in zip(fwhmSources['x_pix'], fwhmSources['y_pix']):
                    cross_size = 2 * radius  # The size of the cross is the diameter of the circle
    
                    # Horizontal line of the cross
                    cross = plt.Line2D(
                        [x - cross_size / 2, x + cross_size / 2], [y, y],
                        color='green', label='FWHM', linewidth=0.33
                    )
                    ax1.add_line(cross)
    
                    # Vertical line of the cross
                    cross_vertical = plt.Line2D(
                        [x, x], [y - cross_size / 2, y + cross_size / 2],
                        color='green', linewidth=0.33
                    )
                    ax1.add_line(cross_vertical)
    
            # Overlay the mask if provided
            if mask is not None:
                from matplotlib import colors
                red_transparent = colors.ListedColormap(['none', 'red'])
                ax1.imshow(mask, cmap=red_transparent, alpha=0.5, origin='lower')
    
            # Set axis labels and limits
            ax1.set_xlabel('X [Pixel]')
            ax1.set_ylabel('Y [Pixel]')
            ax1.set_xlim(0, image.shape[1])
            ax1.set_ylim(0, image.shape[0])
    
            # Create and add legend
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            leg = ax1.legend(
                by_label.values(), by_label.keys(), loc='lower center',
                bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                columnspacing=1.5, fontsize=3
            )
    
            # Finalize figure layout
            fig.tight_layout()
            ax1.set_aspect('equal', adjustable='box')
    
            # Save figure
            if not subtracted:
                save_loc = os.path.join(write_dir, 'sourcecheck_' + base + '.pdf')
            else:
                save_loc = os.path.join(write_dir, 'subtracted_sourcecheck_' + base + '.pdf')
    
            fig.savefig(save_loc, dpi=300, bbox_extra_artists=[leg])
            plt.close()
    
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, e)
    
        
        
    
    
# =============================================================================
#     
# =============================================================================
    def plot_lightcurve(output_file,snr_limit = 5,fwhm = 3,method = 'PSF',reference_epoch = 0,
                        redshift = 0,show_limits = False,show_details = True,default_size = (540,1)):
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from functions import set_size
        
        
        # Color array taken from Superbol by Matt Nicholl
        cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5', 'w': 'firebrick',
            'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'I': 'chocolate',
            'G': 'salmon', 'E': 'salmon', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
            'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
            'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan',
            'W': 'forestgreen', 'Q': 'peru'}
        
        # Maintains order from blue to red effective wavelength
        bandlist = 'FSDNAuUBgcVwrRoGEiIzyYJHKWQ'
        
        
        markers = ['x', 'o', '^', 's', 'D']
        marker_iterator = iter(markers)
        
        # bandlist = 'gr'
        data = pd.read_csv(output_file)
        
        
        
        
        
        if redshift !=0 :
            from functions import get_distance_modulus
            dm = get_distance_modulus(redshift)
        else:
            dm = 0
            
            
        fig = plt.figure(figsize = set_size(*default_size))
        ax1 = fig.add_subplot(111)
        
        ax1.invert_yaxis()
        
        
        
        
        num_detect = 0
        num_nondetect = 0
        for band in bandlist:
            band = band + '_' + method
            if band not in data.columns: continue
        
            data_band = data[~np.isnan(data[band])]
        
            detects_idx =data_band.snr >= snr_limit
            nondetects_idx = data_band.snr < snr_limit
                
            
            
            detects = data_band[detects_idx]
            
            nondetects = data_band[nondetects_idx ]
            
            
            num_detect+=len(detects)
            num_nondetect+=len(nondetects)
            
            
            marker = next(marker_iterator)
            
            
            ax1.errorbar(detects.mjd-reference_epoch,detects[band],yerr = detects[band+'_err'],
                         c = cols[band[0]],ls = '',
                         capsize = 2,
                         marker = marker,label = band[0])
            if show_limits:
                ax1.errorbar(nondetects.mjd-reference_epoch,nondetects[band],yerr = [0.5],
                             c = cols[band[0]],ls = '',marker = marker,lolims = True,alpha = 0.5,zorder = 0)
              
        
        
        ax1.set_ylim(ax1.get_ylim())
        if redshift != 0.0:
            
        
            ax11 = ax1.twinx()
            ax11.set_xlim(ax1.get_xlim())
            
            ax11.set_ylim(ax1.get_ylim()-dm )
            ax11.set_ylabel('Abs. Magnitude')
            
            
        ax1.set_ylabel('App. Magnitude')
        
        if reference_epoch != 0.0:
            ax1.set_xlabel(fr'Days since {reference_epoch}')
        else:
            ax1.set_xlabel('Modified Julian Date')
        
        
        
        if show_details:
            text = fr'# detect {num_detect}'+'\n'+fr'# nondetect {num_nondetect}'
            plt.text(0.02, 0.95, text, transform=ax1.transAxes,
                     verticalalignment='top', 
                     horizontalalignment='left', 
                     bbox=dict(facecolor='white', alpha=0.8))
        
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        ncols = len(by_label.values()) % 3
        if ncols==0: ncols = len(by_label.values())
        ax1.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=[0.5, 1], 
                   loc='lower center',
                   frameon = False,ncols = ncols)
        
        plt.show()
            
        





        
        
        
        
        return

