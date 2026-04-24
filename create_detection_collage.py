#!/usr/bin/env python3
"""
Create a collage of detections from autophot dataset.
Shows original image on left and subtraction cutout on right for each detection.
"""

import os
import csv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.cm as cm

# Configuration
DATA_DIR = "/home/sbrennan/Desktop/SN2026gzf/panstarrs/images_REDUCED_W"
DETECTIONS_CSV = os.path.join(DATA_DIR, "detections_lightcurve_output_PSF.csv")
OUTPUT_FILE = "/home/sbrennan/Desktop/SN2026gzf/Detection_Collage.pdf"

# Cutout parameters
FWHM_MULTIPLIER = 5  # Cutout size = FWHM * FWHM_MULTIPLIER
MIN_CUTOUT_SIZE = 30  # Minimum cutout size in pixels
ORIGINAL_SIZE = 150  # pixels (half-width for original image display)

def parse_detections(csv_file):
    """Parse the detections CSV file."""
    detections = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['filename']:  # Skip empty rows
                detections.append({
                    'filename': row['filename'],
                    'diff_path': row['filename_path'],
                    'date': row['date'],
                    'mjd': float(row['mjd']),
                    'xpix': float(row['xpix']),
                    'ypix': float(row['ypix']),
                    'snr': float(row['snr_psf']),
                    'mag': float(row['apparent_mag']) if row['apparent_mag'] else None,
                    'fwhm': float(row['target_fwhm'])
                })
    return detections

def get_original_path(diff_path):
    """Get the path to the original image from the diff path."""
    # diff path: /path/to/diff_<filename>_APT.fits
    # original path: /path/to/<filename>_APT.fits
    dirname = os.path.dirname(diff_path)
    filename = os.path.basename(diff_path)
    original_filename = filename.replace('diff_', '', 1)  # Remove 'diff_' prefix
    return os.path.join(dirname, original_filename)

def make_cutout(data, x, y, size):
    """Make a cutout around the target position."""
    # Convert to integer indices
    x_int, y_int = int(x), int(y)
    
    # Define cutout boundaries
    x_min = max(0, x_int - size)
    x_max = min(data.shape[1], x_int + size)
    y_min = max(0, y_int - size)
    y_max = min(data.shape[0], y_int + size)
    
    return data[y_min:y_max, x_min:x_max]

def normalize_image(data):
    """Normalize image for display using z-scales."""
    # Remove NaN and Inf values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use percentile scaling
    p1, p99 = np.percentile(data[data != 0], (1, 99))
    if p99 - p1 > 0:
        data_norm = (data - p1) / (p99 - p1)
    else:
        data_norm = data - np.min(data)
    
    return np.clip(data_norm, 0, 1)

def normalize_subtraction(data):
    """Normalize subtraction image to better show positive/negative flux."""
    # Remove NaN and Inf values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use symmetric scaling centered on 0 for subtraction images
    # Get the absolute values for scaling
    abs_data = np.abs(data)
    p95 = np.percentile(abs_data[abs_data != 0], 95)
    
    if p95 > 0:
        # Scale so that p95 maps to 0.8
        scale = p95 / 0.8
        data_norm = data / scale
    else:
        data_norm = data
    
    return data_norm

def create_detection_collage(detections):
    """Create the detection collage."""
    n_detections = len(detections)
    
    # Calculate figure size
    fig_width = 8  # inches
    fig_height = n_detections * 2  # 2 inches per detection
    
    fig, axes = plt.subplots(n_detections, 2, figsize=(fig_width, fig_height))
    
    # Handle single detection case
    if n_detections == 1:
        axes = axes.reshape(1, -1)
    
    for i, det in enumerate(detections):
        print(f"Processing detection {i+1}/{n_detections}: {det['filename']}")
        
        # Get paths
        diff_path = det['diff_path']
        orig_path = get_original_path(diff_path)
        
        # Check if files exist
        if not os.path.exists(orig_path):
            print(f"  Warning: Original image not found: {orig_path}")
            continue
        if not os.path.exists(diff_path):
            print(f"  Warning: Diff image not found: {diff_path}")
            continue
        
        # Read FITS images
        try:
            with fits.open(orig_path) as hdul:
                orig_data = hdul[0].data
                from wcs import get_wcs
                orig_wcs = get_wcs(hdul[0].header)
            
            with fits.open(diff_path) as hdul:
                diff_data = hdul[0].data
        except Exception as e:
            print(f"  Error reading FITS files: {e}")
            continue
        
        # Calculate cutout size based on FWHM
        cutout_size = max(MIN_CUTOUT_SIZE, int(det['fwhm'] * FWHM_MULTIPLIER))
        
        # Convert MJD to time string
        time_obj = Time(det['mjd'], format='mjd')
        iso_str = time_obj.iso
        if 'T' in iso_str:
            time_str = iso_str.split('T')[1][:5]  # Get HH:MM part
        else:
            time_str = iso_str.split(' ')[1][:5]  # Fallback for space separator
        
        # Make cutouts
        orig_cutout = make_cutout(orig_data, det['xpix'], det['ypix'], ORIGINAL_SIZE)
        diff_cutout = make_cutout(diff_data, det['xpix'], det['ypix'], cutout_size)
        
        # Normalize for display
        orig_norm = normalize_image(orig_cutout)
        diff_norm = normalize_subtraction(diff_cutout)
        
        # Plot original image
        ax_orig = axes[i, 0]
        im_orig = ax_orig.imshow(orig_norm, cmap='gray', origin='lower')
        ax_orig.set_title(f"{det['date']} {time_str}", fontsize=10, loc='left', x=0.02, y=0.95)
        ax_orig.axis('off')
        
        # Add info text
        info_text = f"SNR: {det['snr']:.1f}"
        if det['mag']:
            info_text += f"\nMag: {det['mag']:.2f}"
        ax_orig.text(0.02, 0.05, info_text, transform=ax_orig.transAxes, 
                    color='white', fontsize=8, verticalalignment='bottom')
        
        # Add aperture circle on original image
        # Target is at the center of the original cutout
        orig_center_x = orig_cutout.shape[1] // 2
        orig_center_y = orig_cutout.shape[0] // 2
        aperture_radius = det['fwhm'] * 2  # Aperture = 2 * FWHM
        aperture_circle = Circle((orig_center_x, orig_center_y), aperture_radius, 
                                fill=False, edgecolor='yellow', linewidth=1.5, alpha=0.8)
        ax_orig.add_patch(aperture_circle)
        
        # Plot subtraction cutout
        ax_diff = axes[i, 1]
        im_diff = ax_diff.imshow(diff_norm, cmap='gray', origin='lower', vmin=-1, vmax=1)
        ax_diff.set_title('Subtraction', fontsize=10)
        ax_diff.axis('off')
        
        # Add aperture circle on subtraction image
        diff_center_x = diff_cutout.shape[1] // 2
        diff_center_y = diff_cutout.shape[0] // 2
        aperture_circle_diff = Circle((diff_center_x, diff_center_y), aperture_radius, 
                                     fill=False, edgecolor='yellow', linewidth=1.5, alpha=0.8)
        ax_diff.add_patch(aperture_circle_diff)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nCollage saved to: {OUTPUT_FILE}")
    plt.close()

def main():
    print("Parsing detections CSV...")
    detections = parse_detections(DETECTIONS_CSV)
    print(f"Found {len(detections)} detections")
    
    # Sort by date
    detections.sort(key=lambda x: x['date'])
    
    print("\nCreating collage...")
    create_detection_collage(detections)
    
    print("Done!")

if __name__ == "__main__":
    main()
