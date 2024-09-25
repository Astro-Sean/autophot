import os
import numpy as np
import pandas as pd
from functions import set_size
from functions import get_distance_modulus
import pandas as pd
import pathlib
import os, glob
import shutil

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:36:04 2023

@author: seanbrennan
"""

# Function to plot the lightcurve
def plot_lightcurve(output_file, snr_limit=3, fwhm=3, method='PSF', reference_epoch=0,
                    redshift=0, show_limits=False, limit_method='inject', show_details=False,
                    default_size=(540, 1), return_detections=True, format='png'):
    
    import matplotlib.pyplot as plt
    
    # Define color mapping for different photometric bands
    cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5', 'w': 'firebrick',
            'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'I': 'chocolate',
            'G': 'salmon', 'E': 'salmon', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
            'S': 'mediumorchid', 'D': 'purple', 'A': 'midnightblue',
            'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan',
            'W': 'forestgreen', 'Q': 'peru'}
    
    # Define the order of bands for plotting, from blue to red wavelengths
    bandlist = 'FSDNAuUBgcVwrRoGEiIzyYJHKWQ'
    
    # Marker styles for plotting data points
    markers = ['o', 's', 'D', '<', '>', 'p', '*', 'h', 'H', '^', 'd', 'x', '+', '|', '_']
    marker_iterator = iter(markers)
    
    # Load the output data from the CSV file
    data = pd.read_csv(output_file)
    
    # Get the directory path for saving figures
    save_path = os.path.dirname(output_file)
    
    # Calculate the distance modulus if the redshift is provided
    if redshift != 0:
        dm = get_distance_modulus(redshift)
    else:
        dm = 0
        
    # Create a figure and axis for plotting
    fig = plt.figure(method, figsize=set_size(*default_size))
    ax1 = fig.add_subplot(111)
    
    # Invert the y-axis (magnitude scale)
    ax1.invert_yaxis()
    
    # Initialize counters for detections and non-detections
    num_detect = 0
    num_nondetect = 0
    
    # Initialize a list to store detections if requested
    if return_detections:
        output_detections = []
        
    # Iterate through each band in the predefined bandlist
    for band in bandlist:
        band = band + '_' + method
        # Skip bands that are not present in the data
        if band not in data.columns:
            continue
        
        # Filter out rows with NaN error values for this band
        data_band = data[~np.isnan(data[band+'_err'])]
        
        data_band['lmag_'+limit_method] = data_band['lmag_'+limit_method] + data_band[fr'zp_{method}']
        
        data_band['lmag_'+limit_method][np.isnan(data_band['lmag_'+limit_method])] = 999
        # Identify detections based on the signal-to-noise ratio (SNR) limit
        # detects_idx = (data_band[band] > data_band['lmag_'+limit_method] )
        detects_idx = data_band[band] < data_band['lmag_'+limit_method]  #& (np.isnan(data_band['lmag_'+limit_method] ))
        
        detects = data_band[detects_idx]
        
        
        nondetects = data_band[~detects_idx]
        
        # Add the detected data to the output list if required
        if return_detections:
            output_detections.append(detects)
        
        # Update the count of detections and non-detections
        num_detect += len(detects)
        num_nondetect += len(nondetects)
        
        # Get the next marker style for plotting
        marker = next(marker_iterator)
        
        # Plot the detections with error bars
        markers, caps, bars = ax1.errorbar(detects.mjd - reference_epoch,
                                           detects[band], yerr=detects[band+'_err'],
                                           markerfacecolor=cols[band[0]],
                                           markeredgecolor='black', ls='',
                                           capsize=3,
                                           zorder=1,
                                           ecolor='black',
                                           marker=marker,
                                           markersize=5,
                                           label=band[0])
        # Set transparency for error bars and caps
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        
        # Plot non-detections if limits should be shown
        if show_limits:
            ax1.errorbar(nondetects.mjd - reference_epoch, nondetects['lmag_'+limit_method] ,
                         markeredgecolor=cols[band[0]],
                         markersize=5,
                         markerfacecolor='none',
                         ls='', marker='v', alpha=0.25, zorder=0)
    
    # Save detections to a CSV file if required
    if return_detections and len(output_detections) >= 1:
        detections_loc = os.path.join(save_path, 'detections.csv')
        output_detections = pd.concat(output_detections, ignore_index=True)
        output_detections.to_csv(detections_loc, index=False)
    else:
        detections_loc = None
        
    # Set the y-axis limits
    ax1.set_ylim(ax1.get_ylim())
    
    # Add a secondary y-axis for absolute magnitude if redshift is provided
    if redshift != 0.0:
        ax11 = ax1.twinx()
        ax11.set_xlim(ax1.get_xlim())
        ax11.set_ylim(ax1.get_ylim() - dm)
        ax11.set_ylabel('Abs. Magnitude')
        
    ax1.set_ylabel('App. Magnitude')
    
    # Set the x-axis label based on whether a reference epoch is provided
    if reference_epoch != 0.0:
        ax1.set_xlabel(fr'Days since {reference_epoch}')
    else:
        ax1.set_xlabel('Modified Julian Date')
    
    # Optionally display the number of detections and non-detections
    if show_details:
        text = fr'# detect {num_detect}'+'\n'+fr'# nondetect {num_nondetect}'
        plt.text(0.02, 0.95, text, transform=ax1.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # Create a legend for the plot
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    ncols = int(len(by_label.values()))
    
    # Adjust the number of columns in the legend if needed
    if ncols > 6:
        ncols /= 2
    if ncols == 0:
        ncols = len(by_label.values())
    
    # Add the legend to the plot
    ax1.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=[0.5, 1],
               loc='lower center',
               frameon=False, ncols=ncols)
    
    # Adjust the layout of the figure
    fig.tight_layout()
    
    # Display the plot
    plt.show(block=False)
    
    # Save the plot to a file
    plt.savefig(os.path.join(save_path, 'lightcurve_%s.%s' % (method, format)), dpi=300)
    
    # Return the location of detections file or None
    if return_detections:
        return detections_loc
    else:
        return



# =============================================================================
# 
# =============================================================================

# Function to check detection plots
def check_detection_plots(output_file, method='AP'):
    
    
    # Read the data from the output file
    data = pd.read_csv(output_file)
    save_path = os.path.dirname(output_file)
    
    # Create a directory to save the detection plots
    save_path = os.path.join(save_path, 'detections_%s' % method)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Loop through each row in the data
    for _, row in data.iterrows():
        
        loc = row['filename']
        loc = os.path.dirname(loc)
        files = glob.glob(os.path.join(loc, '*'))
        
        date = ''.join(row['date'].split('-'))
        f = row['filter']
        
        try:
            if method == 'AP':
                file = [i for i in files if os.path.basename(i).startswith('aperture_')][0]
            elif method == 'PSF':
                file = [i for i in files if os.path.basename(i).startswith('targetPSF_')][0]
        except:
            continue
        
        new_fname = os.path.basename(file)
        new_fname = f'{f}band_{date}_' + new_fname
        new_fpath = os.path.join(save_path, new_fname)
        
        # Copy the file to the save path
        shutil.copyfile(file, new_fpath)
        
    return save_path
