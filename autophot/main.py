#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:55:18 2022

@author: seanbrennan
"""

import argparse

def run_photometry():
    """
    Perform photometry operations.

    Args:
        -f (str): Filepath of fits file
        -c (str): Path to the input YAML file.
        -temp (bool): Flag to prepare a template. Default is False.

    Returns:
        None
    """
    


    import os  # For interacting with the operating system
    import sys  # For system-specific parameters and functions
    import pathlib  # For object-oriented filesystem paths
    import shutil  # For high-level file operations
    import logging  # For logging messages
    import datetime  # For manipulating dates and times
    import time  # For time-related functions
    import warnings  # For issuing warning messages
    import glob
    import numpy as np  # For numerical operations

    from collections import OrderedDict  # For ordered dictionaries

    # Importing modules from the Astropy library for astronomical calculations
    from astropy.time import Time  # For handling time and dates
    from astropy.io import fits  # For reading and writing FITS files
    import astropy.wcs as wcs  # For World Coordinate System transformations
    from astropy import units as u  # For handling physical units
    from astropy.coordinates import SkyCoord  # For celestial coordinate transformations
    from astropy.stats import SigmaClip  # For sigma clipping
    from astropy.stats import sigma_clipped_stats  # For calculating statistics with sigma clipping
    from astropy.nddata.utils import Cutout2D  # For creating 2D cutouts of images

    from astropy.utils.exceptions import AstropyWarning  # For handling Astropy-specific warnings

    # Importing modules from the Photutils library for photometry
    from photutils.centroids import centroid_com, centroid_2dg, centroid_sources  # For centroid calculations
    from photutils.utils import circular_footprint  # For creating circular footprints
    from photutils.background import Background2D, MedianBackground  # For background estimation
    from photutils.segmentation import detect_sources, detect_threshold  # For source detection and segmentation
    import yaml  # For parsing YAML configuration files
    import pandas as pd  # For data manipulation and analysis

    # Importing various utility functions from the functions module
    from functions import (
        get_image,  # Function to get an image
        get_header,  # Function to get the header of an image
        border_msg,  # Function to create a border message
        autophot_yaml,  # Function to handle YAML configuration for autophot
        arcmins2pixel,  # Function to convert arcminutes to pixels
        pix_dist,  # Function to calculate pixel distance
        convert_to_mjd_astropy,  # Function to convert to Modified Julian Date using Astropy
        suppress_stdout,  # Function to suppress standard output
        # is_undersampled  # Function to check if data is undersampled
    )

    from templates import templates  # Importing templates module

    # Importing functions related to image masking
    from mask import (
        run_maximask_with_file,  # Function to run maximask with a file
        create_image_mask  # Function to create an image mask
    )

    from fwhm import find_fwhm  # Function to find the Full Width at Half Maximum (FWHM)

    from catalog import catalog  # Function to handle catalog operations

    from aperture import aperture  # Function for aperture photometry

    from zeropoint import zeropoint  # Function to calculate the zero point

    from psf import psf  # Function for Point Spread Function (PSF) analysis

    from limits import limits  # Function to set limits for various parameters

    from plot import plot  # Function to plot data


    from templates import templates

    # Importing additional utility functions from the functions module
    from functions import (
        dict_to_string_with_hashtag,  # Function to convert a dictionary to a string with hashtags
        quadratureAdd,  # Function to add values in quadrature
        beta_value,  # Function to calculate the beta value
        # distance_to_uniform_row_col,  # Function to calculate distance to uniform row/column
        # convolve_and_remove_nans_zeros  # Function to convolve data and remove NaNs and zeros
    )

    from wcs import check_wcs  # Function to check World Coordinate System (WCS) information


    # Initialize argument parser with a description
    parser = argparse.ArgumentParser(description='Perform photometry operations.')

    # Add arguments to the parser
    parser.add_argument('-f', dest='filepath', type=str, help='Filepath of fits file')
    parser.add_argument('-c', dest='input_yaml', type=str, help='Path to the input YAML file.')
    parser.add_argument('-temp', dest='prepareTemplate', action='store_true', help='Flag to prepare a template.', default=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access parsed arguments
    scienceFile = args.filepath  # Filepath of the FITS file
    input_yaml_loc = args.input_yaml  # Path to the input YAML file
    prepareTemplate = args.prepareTemplate  # Flag to prepare a template

    # Filter out Astropy warnings
    warnings.filterwarnings("ignore", category=AstropyWarning, append=True)

    # Load the input YAML file
    with open(input_yaml_loc, 'r') as file:
        input_yaml = yaml.safe_load(file)

    # Extract coordinates from the YAML file
    coords = {}
    coords['ra'] = input_yaml['target_ra']  # Right Ascension
    coords['dec'] = input_yaml['target_dec']  # Declination
    try:

        # Get the working directory and output directory name from the YAML configuration
        wdir = input_yaml['fits_dir']
        newDir = '_' + input_yaml['outdir_name']
        
        # Create the new output directory path
        baseDir = os.path.basename(wdir)
        workLoc = baseDir + newDir
        new_output_dir = os.path.join(os.path.dirname(wdir), workLoc)
        
        # Create the new output directory if it doesn't exist
        pathlib.Path(new_output_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(new_output_dir)  # Change the current working directory to the new output directory
        
        # Take in the original data and copy it to the new location
        # The new file will have _APT appended to the filename
        # Create names of directories with name=filename and account for correct subdirectory locations
        base_wext = os.path.basename(scienceFile)
        base = os.path.splitext(base_wext)[0]
        fname_ext = os.path.splitext(base_wext)[1]
        
        # Replace spaces and periods in the base filename and remove specific substrings
        base = base.replace(' ', '_')
        base = base.replace('.', '_')
        base = base.replace('_APT', '')
        base = base.replace('_ERROR', '')
        
        # Store the base filename without any extension in the YAML configuration
        input_yaml['base'] = base
        
        if prepareTemplate:
            # If preparing a template, set the current directory to the directory of the science file
            cur_dir = os.path.dirname(scienceFile)
        else:
            # Create a subdirectory system based on the input directory structure
            root = os.path.dirname(scienceFile)
            sub_dirs = root.replace(wdir, '').split('/')
            sub_dirs = [i.replace('_APT', '').replace(' ', '_') for i in sub_dirs]
            cur_dir = new_output_dir
        
            for i in range(len(sub_dirs)):
                if i:  # If the directory is not blank
                    newSubdir = os.path.join(cur_dir, sub_dirs[i] + '_APT')
                    pathlib.Path(newSubdir).mkdir(parents=True, exist_ok=True)
                    cur_dir = newSubdir
        
            # Finally, create a folder with the filename as its name
            cur_dir = os.path.join(cur_dir, base)
            pathlib.Path(cur_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a logger to capture all print statements and write them to a file
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        
        # Set up logging to file
        logging.basicConfig(
            level=logging.INFO,
            format='%(filename)s-%(levelname)s: %(message)s',
            datefmt='%m-%d %H:%M',
            filename=os.path.join(cur_dir, 'LOG_' + str(base) + '.log'),
            filemode='w'
        )
        
        # Create a console handler for logging
        console = logging.StreamHandler()
        
        # Define a handler which writes INFO messages or higher to sys.stderr
        console.setLevel(logging.INFO)
        
        # Set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        
        # Add the handler to the root logger
        logging.getLogger('').addHandler(console)


        # =============================================================================
        # 
        # =============================================================================
        
        # Check if preparing a template
        if prepareTemplate:
            fpath = scienceFile
            if os.path.exists(fpath+'.original'):
            # Replace template file with original
                logging.info('\n\nPre-reduced file found - Replacing template file with original')
                os.remove(fpath)
                shutil.copyfile(fpath+'.original', fpath)
            elif not os.path.exists(fpath.replace('.original', '')):
                # Only pre-reduced file found - copy and save
                logging.info('Only Pre-reduced file found - copying and saving')
                shutil.copyfile(fpath, fpath.replace('.original', ''))
                fpath = fpath.replace('.original', '')
            elif '.original' in fpath:
            # Copy template for recovery
                shutil.copyfile(fpath, fpath.replace('.original', ''))
                fpath = fpath.replace('.original', '')
            else:
                logging.info('Copying template for recovery')
                shutil.copyfile(fpath, fpath+'.original')
        else:
            # Copy new file to new directory
            fpath = os.path.join(cur_dir, base +'_APT'+fname_ext).replace(' ','_')
            shutil.copyfile(scienceFile, fpath)

    


        # =============================================================================
        # 
        # =============================================================================
        # Set location of new (copied) file
        input_yaml['fpath'] = fpath

        # base of new (copied) file [without extension]
        base = os.path.basename(fpath)

        write_dir = (cur_dir + '/').replace(' ','_')
        input_yaml['write_dir'] = write_dir

        # Get image and header file
        image    = get_image(fpath)
        header = get_header(fpath)

        logging.info(border_msg('File: '+str(base)))

        logging.info('\n\n\nFilename: %s' %  str(fpath))
        logging.info('Start Time: %s' %  str(datetime.datetime.now()))

        # Get insturment and telescope data
        instKey = 'INSTRUME'
        teleKey = 'TELESCOP'

        instrument = header[instKey]
        telescope = header[teleKey]

        tele_default_yml = 'telescope.yml'
        teleData = autophot_yaml(os.path.join(input_yaml['wdir'],tele_default_yml)).load()
        
        try:
            date_mjd = header[teleData[telescope][instKey][instrument]['mjd']]

        
        except Exception as e:
            try:
                
                date_iso = header[teleData[telescope][instKey][instrument]['date']]
                date_mjd =  convert_to_mjd_astropy(date_iso)
            except Exception as e:
                date_mjd = 999
                    
        # =============================================================================
        # 
        # =============================================================================
        
        output = OrderedDict({})


        output.update({'fname':fpath})
        output.update({'telescope':teleData[telescope][instKey][instrument]['Name']})
        output.update({'TELESCOP':telescope})
        output.update({'INSTRUME':instKey})
        output.update({'instrument':instrument})
        output.update({'mjd':date_mjd})


        input_yaml['tele'] = telescope
        input_yaml['inst'] = instrument
        input_yaml['instKey'] = instKey
        
        
        # try to get gain
        if teleData[telescope][instKey][instrument]['gain'] in header.keys():
            gain = header[teleData[telescope][instKey][instrument]['gain']]
        else:
            gain = 1
            
        input_yaml['gain'] = gain

        
        # Try to get saturation level, if not
        if teleData[telescope][instKey][instrument]['saturate'] in header.keys():

            saturate = 0.95 * header[teleData[telescope][instKey][instrument]['saturate']] #/ input_yaml['gain'] 
            
        else:
            
            saturate = 0.95 * np.nanmax(image[np.isfinite(image)])
            
        input_yaml['saturate'] = saturate 
        
        # =============================================================================
        #             
        # =============================================================================


        # try to get read noise
        if teleData[telescope][instKey][instrument]['readnoise'] in header.keys():
            readnoise = header[teleData[telescope][instKey][instrument]['readnoise']]
        else:
            readnoise = 0


        if teleData[telescope][instKey][instrument]['exptime'] in header.keys():
            expTime = header[teleData[telescope][instKey][instrument]['exptime']]
        else:
            expTime = 1
        
        header['EXPTIME'] = expTime
            

        # =============================================================================
        # 
        # =============================================================================
        

        # These are the keys we want to avoid
        avoid_keys = ['']

        open_filter = False
        found_correct_key = False
        filter_keys = [i for i in list(teleData[telescope][instKey][instrument]) if i.startswith('filter_key_')]

        for filter_header_key in filter_keys:

            if teleData[telescope][instKey][instrument][filter_header_key] not in list(header.keys()):
                continue

            if header[teleData[telescope][instKey][instrument][filter_header_key]].lower() in avoid_keys:
                open_filter = True
                continue

            if header[teleData[telescope][instKey][instrument][filter_header_key]] in teleData[telescope][instKey][instrument]:
                found_correct_key = True
                break

        if not found_correct_key and open_filter:
            raise Exception('Cannot find correct filter keyword')

   
        # Get image filter
        input_yaml['filter_key'] = teleData[telescope][instKey][instrument][filter_header_key]

        imageFilter =  teleData[telescope][instKey][instrument][str(header[input_yaml['filter_key']])]


        if input_yaml['HST_mode']:
            imageFilter = header['FILTER']
            
        # =============================================================================
        #                 
        # =============================================================================
        

        # Try to get pixel scale of image
        pixel_scale = 0.25
        if  input_yaml['wcs']['guess_pixel_scale']:

            try:
                with suppress_stdout():
                    imageWCS = wcs.WCS(header, fix = True)# WCS values

                    xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(imageWCS)
   

                    pixel_scale = xy_pixel_scales[0] * 3600

            except Exception as e:
                print(e)

        # =============================================================================
        #         
        # =============================================================================
        
        # Pixel scale
        if 'pixel_scale' in teleData[telescope][instKey][instrument] and not input_yaml['wcs']['guess_pixel_scale']:
            pixel_scale= teleData[telescope][instKey][instrument]['pixel_scale']
        
        if pixel_scale>5:
            pixel_scale =  0.2
            input_yaml['wcs']['guess_pixel_scale'] = False
            
        # Special case for MPI+GROND in the IR
        if telescope == 'MPI-2.2' and imageFilter in ['J','H','K']:
            IR_gain_key = '%s_GAIN' % imageFilter
            logging.info('Detected GROND IR - setting GAIN key to %s_GAIN' % IR_gain_key)
            teleData[telescope][instKey][instrument]['gain'] = IR_gain_key
            gain = IR_gain_key

        if telescope == 'MPI-2.2':
            if imageFilter in ['J','H','K']:
                logging.info('Detected GROND IR - setting pixel scale to 0.6')
                pixel_scale = 0.6
            else:
                pixel_scale = 0.158

        input_yaml['wcs']['pixel_scale'] = pixel_scale
        
        if input_yaml['target_name'] != None:
            target_ra  =  input_yaml['target_ra']
            target_dec =  input_yaml['target_dec']

            # target_prefix = coords['name_prefix']
            target_coords = SkyCoord(target_ra , target_dec ,unit = (u.deg,u.deg))
            input_yaml['target_ra'] = target_coords.ra.degree
            input_yaml['target_dec']= target_coords.dec.degree
        elif input_yaml['target_ra'] != None and input_yaml['target_dec'] != None:
            target_coords = SkyCoord(input_yaml['target_ra'] , input_yaml['target_dec'] ,unit = (u.deg,u.deg))
            input_yaml['target_ra'] = target_coords.ra.degree
            input_yaml['target_dec']= target_coords.dec.degree
        else:
            try:
                if input_yaml['use_header_radec']:
                    # use RA and Dec coordinates from header file
                    # TODO: Address this in a future version
                    raise  Exception('Not available in current version')
                    target_coords = SkyCoord(header['CAT-RA'] , header['CAT-DEC'] ,unit = (u.hourangle,u.deg))
                    input_yaml['target_ra'] =  target_coords.ra.degree
                    input_yaml['target_dec']=  target_coords.dec.degree
            except:
                logging.warning('NO RA:DEC keywords found')

        # Name of target with prefix - if available
        if not (input_yaml['target_name'] is None):
            input_yaml['target_name'] = input_yaml['target_name'].replace('SN','').replace('AT','')
        elif not (input_yaml['target_ra'] is None) and not (input_yaml['target_dec'] is None):
            input_yaml['target_name'] = 'Targeted Source'
        else:
            input_yaml['target_name'] = 'Center of Field'

        # =============================================================================
        # 
        # =============================================================================
       
        input_yaml['imageFilter'] = imageFilter
        input_yaml['pixel_scale']   =  pixel_scale
        input_yaml['exptime'] = float(expTime)

        input_yaml['saturate'] =    saturate
        input_yaml['gain'] = gain

        # =============================================================================
        # 
        # =============================================================================
                
        start = time.time()
        logging.info('Telescope: %s' % telescope)
        logging.info('Instrument: %s' % instrument)
        logging.info('Filter: %s'% imageFilter)
        logging.info('MJD: %.3f' % date_mjd)
        logging.info('Gain: %s [e/ADU]'% gain)
        logging.info('Readnoise: %.3f [e/pixel]' % readnoise)
        logging.info('Saturation level: %.3f [ADU]' % saturate )
        logging.info('Exposure time: %.3f [s]' % float(expTime) )
        logging.info('Pixel scale: %.3f [arcsec/pixel]' % pixel_scale )
   
        date = Time([date_mjd], format='mjd', scale='utc')
        date = date.iso[0].split(' ')[0]
        logging.info('Date of Observation : %s' % date)
        
        
        header['gain'] = gain
        header['saturate'] = saturate
        header['readnoise']= readnoise

        
        # =============================================================================
        # 
        # =============================================================================

        logging.info(border_msg('Removing image background'))

        # Define background estimation parameters
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        
        # Define box size and filter size for background estimation
        box_size = int(64)
        filter_size = int(8)
        
        # Make sure filter size and box size are odd numbers
        if filter_size % 2 == 0:
            filter_size += 1
        if box_size % 2 == 0:
            box_size += 1
        
        # Perform sigma clipping and background estimation
        _,_, std_image = sigma_clipped_stats(image, sigma=3.0, cenfunc=np.nanmedian, stdfunc='mad_std')
        # threshold = detect_threshold(image, nsigma=5) 
        # segment_map = detect_sources(image, threshold, npixels=7)
        
        # try:
            
        #     # Create a mask for the background
        #     background_mask = segment_map.make_source_mask()
        # except:
        #     background_mask = None
        # Estimate the background using the background mask
        bkg = Background2D(image, box_size, 
                   filter_size=filter_size,
                   sigma_clip=sigma_clip, 
                   bkg_estimator=bkg_estimator,
                   # mask=background_mask
                   )
        
        
        background_surface = bkg.background
        # Subtract the estimated background from the image
        # image = image - 
        
        # Get the median background value
        image_background_median = bkg.background_median
        
        # Log the background median value
        logging.info(f'> Background median: {image_background_median:.1f} counts')
        
        # Adjust the saturation value based on the background median
        if image_background_median < 0.95 * saturate:
            saturate -= image_background_median +  10*std_image
            logging.info(f'> Updated saturation value: {saturate:.1f} counts')
        else:
            logging.warning(f'> Background median [{image_background_median:.1f}] is greater than Saturation value [{saturate:.1f}]\n Setting Saturation to a very large number!')
            saturate = 1e12
        

                
        # =============================================================================
        #             
        # =============================================================================

        if input_yaml['cosmic_rays']['remove_cmrays']:
            
            if telescope != 'PS1': 
                from cosmic import remove_cosmic_rays
                
                image = remove_cosmic_rays(input_yaml = input_yaml,
                                           fpath = fpath,
                                           image=image,
                                           header=header,
                                           
                                           use_lacosmic = False).remove(bkg =  background_surface,
                                                                        satlevel = saturate,
                                                                        gain = gain)
            
            
# =============================================================================
#             
# =============================================================================



        # Update the saturation value in the input YAML and header
        input_yaml['saturate'] = saturate
        header['saturate'] = saturate
        
        image = image - background_surface
        # Write the modified image and header back to the file
        fits.writeto(fpath,
                  image,
                  header,
                  overwrite=True,
                  output_verify='silentfix+ignore') 
        
        # =============================================================================
        #                 
        # =============================================================================

        # Try to get WCS (World Coordinate System) information from the header
        existingWCS = True
        try:
            # Suppress standard output while getting WCS values from the header
            with suppress_stdout():
                WCSvalues_old = wcs.WCS(header, relax=True, fix=False)
        except Exception as e:
            # Log if no pre-existing WCS is found
            logging.info('No pre-existing WCS found: %s' % e)
            # existingWCS = False
        
        # Suppress standard output while checking WCS
        with suppress_stdout():
            imageWCS_obj = check_wcs(fpath=fpath, image=image, header=header, default_input=input_yaml)
        
        # Check if WCS needs to be removed or if there is no existing WCS or if preparing a template
        if input_yaml['wcs']['remove_wcs'] or not existingWCS or prepareTemplate:
            # Remove WCS from the header
            header = imageWCS_obj.remove()
            
            try:
                # Get WCS keywords from the header
                wcs_keywords = wcs.WCS(header).to_header()
                # Remove WCS keywords from the header
                for key in wcs_keywords:
                    if key in header:
                        del header[key]
            except:
                pass
        
            # Replace non-finite values in the image with a small number
            image[~np.isfinite(image)] = 1e-30
            
            # Write the updated image and header to the FITS file
            fits.writeto(fpath, image, header, overwrite=True, output_verify='silentfix+ignore')
            
            # Reload the image and header from the FITS file
            image = get_image(fpath)
            header = get_header(fpath)
            
            # Set existingWCS to False as WCS has been removed
            existingWCS = False
        
        # Check if WCS needs to be redone but not removed
        elif input_yaml['wcs']['redo_wcs'] and not input_yaml['wcs']['remove_wcs']:
            logging.info('Passing pre-existing WCS values to Astrometry.net')
        
        # Loop to redo WCS if required
        while input_yaml['wcs']['redo_wcs']:
            # Suppress standard output while checking WCS
            with suppress_stdout():
                imageWCS_obj = check_wcs(fpath=fpath, image=image, header=header, default_input=input_yaml)
                
                # Perform plate solving to update the header with WCS information
                updated_header = imageWCS_obj.plate_solve(solvefieldExe=input_yaml['wcs']['solve_field_exe_loc'])
            
            # Check if the updated header is a float and if there was an existing WCS
            if isinstance(updated_header, float) and existingWCS:
                # Update the header with old WCS values
                header.update(WCSvalues_old.to_header(), relax=True, fix=False)
            else:
                # Use the updated header
                header = updated_header
            
            # Write the updated image and header to the FITS file
            fits.writeto(fpath, image, header, overwrite=True, output_verify='silentfix+ignore')
            
            break  # Exit the loop after updating the WCS

        # =============================================================================
        #             
        # =============================================================================

        # Trim the image if specified in the input YAML
        if not (input_yaml['preprocessing']['trim_image'] is None):
            logging.info(f'Trimming image to {input_yaml["preprocessing"]["trim_image"]} arcmin box centered on target')

            # Get the WCS values from the header
            imageWCS = wcs.WCS(header, fix=True)

            # Create a cutout of the image centered on the target coordinates
            cutout = Cutout2D(image.astype(float),
                                target_coords,
                                (input_yaml['preprocessing']['trim_image'] * u.arcmin * 2),
                                wcs=imageWCS,
                                mode='trim',
                                fill_value=1e-30)

            # Update the image and header with the trimmed values
            image = cutout.data
            header.update(cutout.wcs.to_header())

            # Update the WCS values after trimming
            imageWCS = wcs.WCS(header, fix=True)

            # Write the modified image and header back to the FITS file
            fits.writeto(fpath,
                            image,
                            header,
                            overwrite=True,
                            output_verify='silentfix+ignore')

        # Update the NAXIS1 and NAXIS2 values in the input YAML
        input_yaml['NAXIS1'] = image.shape[1]
        input_yaml['NAXIS2'] = image.shape[0]


            
        image[~np.isfinite(image)] = 1e-30      
        fits.writeto(fpath,
                      image,
                      header ,
                      overwrite = True,
                      output_verify = 'silentfix+ignore')

        image    = get_image(fpath)
        header = get_header(fpath)
        
        # =============================================================================
        # 
        # =============================================================================
    
        # Get target pixel location
        imageWCS = wcs.WCS(header, fix = True)# WCS values

        # Calculate the pixel scale in arcseconds
        xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(imageWCS)
        pixel_scale = xy_pixel_scales[0] * 3600

        # =============================================================================
        # 
        # =============================================================================
        

        if input_yaml['target_name'] == None:
            # If not target coords are given, will use center of field
            if input_yaml['target_ra'] == None and input_yaml['target_dec'] == None:
                logging.info('No target information given, using image center')
                # if no object is given i.e name,ra,dec then take the middle
                # translate pixel values to ra,dec at center of image
                center = imageWCS.all_pix2world([image.shape[1]/2],[image.shape[0]/2],1)
                # get ra,dec in deg/SkyCoord format
                target_coords = SkyCoord(center[0][0] , center[1][0] ,unit = (u.deg,u.deg))
                # update input_yaml file
                input_yaml['target_ra'] = target_coords.ra.degree
                input_yaml['target_dec']= target_coords.dec.degree
                # Target coords are now set to center of image
                input_yaml['target_x_pix'] = image.shape[1]/2
                input_yaml['target_y_pix'] = image.shape[0]/2
                target_x_pix = image.shape[1]/2
                target_y_pix = image.shape[0]/2
            else:
                # if no name is given but ra and dec are, use those instead:
                logging.info('Using user given RA and Dec')
                target_ra  = input_yaml['target_ra']
                target_dec = input_yaml['target_dec']
                target_coords = SkyCoord(target_ra , target_dec ,unit = (u.deg,u.deg))
                target_x_pix, target_y_pix = imageWCS.all_world2pix(target_coords.ra.degree,
                                            target_coords.dec.degree,
                                            0)
                input_yaml['target_x_pix'] = target_x_pix
                input_yaml['target_y_pix'] = target_y_pix
                input_yaml['target_ra'] = target_coords.ra.degree
                input_yaml['target_dec']= target_coords.dec.degree
        elif input_yaml['target_name'] != None:
            try:
                # Get target info from [pre-saved] TNS_response
                logging.info('Using location of %s from TNS' % input_yaml['target_name'])
                target_ra  = coords['ra']
                target_dec = coords['dec']
                target_coords = SkyCoord(target_ra , target_dec ,unit = (u.deg,u.deg))
                target_x_pix, target_y_pix = imageWCS.all_world2pix(target_coords.ra.degree,target_coords.dec.degree, 0)
                input_yaml['target_ra'] = target_coords.ra.degree
                input_yaml['target_dec']= target_coords.dec.degree
                input_yaml['target_x_pix'] = target_x_pix
                input_yaml['target_y_pix'] = target_y_pix
            except Exception as e:
                raise Exception(str(e)+'\n\nFailed to converge on target position!nAre you sure %s is in this image?' % input_yaml['target_name'])
        else:
            try:
                target_coords = SkyCoord(header['RA'] , header['DEC'] ,unit = (u.deg,u.deg))
                target_x_pix, target_y_pix = imageWCS.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 0)
                input_yaml['target_ra'] =  target_coords.ra.degree
                input_yaml['target_dec']=  target_coords.dec.degree
                input_yaml['target_x_pix'] = target_x_pix
                input_yaml['target_y_pix'] = target_y_pix
            except Exception as e:
                logging.exception(e+'\n> NO RA:DEC keywords found <')
               
        # =============================================================================
        #                    
        # =============================================================================
                        
        if prepareTemplate:
            # Log message indicating template image cleanup
            logging.info(border_msg('Tidying up template image '))
        
            # Find the center and boundaries of the non-uniform region in the image
            center_y, center_x, top_row, bottom_row, left_col, right_col = templates(input_yaml = input_yaml).find_non_uniform_center(image)
            
            # Calculate the height and width of the cropped image
            height = bottom_row - top_row - 1
            width = right_col - left_col - 1
            
            position = (center_x, center_y)  # Note: Cutout2D expects (x, y) position
            size = (height, width-10)
            
            # Create a cutout of the non-uniform region in the image
            imageCutout = Cutout2D(
            image,
            position,
            size,
            wcs=imageWCS,
            mode='partial',
            fill_value=1e-30
            )
            
            # Update the WCS and header with the cutout values
            imageWCS = wcs.WCS(imageCutout.wcs.to_header(), relax=True)
            header.update(imageWCS.to_header(), relax=True)
            
            # Write the modified image and header back to the FITS file
            fits.writeto(fpath,
                  imageCutout.data,
                  header,
                  overwrite=True,
                  output_verify='silentfix+ignore')
        
        
        
        image    = get_image(fpath)
        header = get_header(fpath)
    
        
        imageWCS = wcs.WCS(header,relax = True,fix = False)# WCS values
    
        target_x_pix, target_y_pix = imageWCS.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 0)
        input_yaml['target_x_pix'] = target_x_pix
        input_yaml['target_y_pix'] = target_y_pix
        
        # =============================================================================
        # 
        # =============================================================================
        
        # Check if target name is specified
        if input_yaml['target_name'] != None:
            logging.info('{} is located at ({:.1f},{:.1f})'.format((input_yaml['name_prefix']+input_yaml['objname']),
                                                                   input_yaml['target_x_pix'],
                                                                   input_yaml['target_y_pix']))

        # Check if target pixel coordinates are outside of the image
        if target_x_pix < 0 or target_x_pix > image.shape[1] or target_y_pix < 0 or target_y_pix > image.shape[0]:
            raise Exception('*** EXITING - Target pixel coordinates outside of image [{}, {}] ***'.format(int(target_x_pix), int(target_y_pix)))
        
        # =============================================================================
        #                 
        # =============================================================================

        xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(imageWCS)

        # Check if local stars should be used for photometry
        if input_yaml['photometry']['use_local_stars'] or \
            input_yaml['photometry']['use_local_stars_for_PSF'] or \
            input_yaml['photometry']['use_local_stars_for_FWHM']:

            # Transform arcmin to pixel values
            local_radius =  arcmins2pixel(input_yaml['photometry']['use_source_arcmin'],xy_pixel_scales[0])

            input_yaml['photometry']['local_radius'] = local_radius
            logging.info('Using stars within %d arcmin [%d px]' % (input_yaml['photometry']['use_source_arcmin'],local_radius))


        # =============================================================================
        # 
        # =============================================================================
        # Get the WCS information from the header
        imageWCS = wcs.WCS(header, relax=True, fix=False)
    
        # =============================================================================
        #             
        # =============================================================================
        # Create an instance of the template class
        template_functions = templates(input_yaml)
        
        source_mask = None
        bright_sources = None

        # # Find the bright sources in the image
        # bright_sources = template_functions.find_bright_sources(header=header)

        # # Define the target location as a list of coordinates
        # target_location = [(input_yaml['target_x_pix'], input_yaml['target_y_pix'])]

        # # Create a source mask to exclude bright sources from the analysis
        # source_mask = template_functions.create_image_mask(image, sat_lvl=input_yaml['saturate'], 
        #                                                     fwhm=4, npixels=11, padding=1.5, snr_limit=3000, 
        #                                                     # create_source_mask=False,
        #                                                     ignore_position=target_location,
        #                                                     # remove_large_sources=False,
        #                                                     bright_sources=bright_sources
        #                                                     )

        # # Convert the source mask to a boolean array
        # source_mask = source_mask.astype(bool)
        # =============================================================================
        #             
        # =============================================================================

        # Measure the image FWHM, isolated sources, and scale
        imageFwhm, isolatedSources, scale = find_fwhm(input_yaml=input_yaml).measure_image(image=image, mask=source_mask)
        seeing = pixel_scale * imageFwhm
        logging.info('> Seeing %.3f [arcsec]' % seeing)
        
        
        if imageFwhm<2:
            logging.info(f'\nFWHM is less that 2 pixels [{imageFwhm:.2f} pixels]\nLow values can lead to undersampling, setting to 2 pixels!')
            imageFwhm = 2    
            
        # =============================================================================
        # When performing template subtraction it is good practive to have the template sharper that then image
        # =============================================================================

        if input_yaml['template_subtraction']['do_subtraction'] and not prepareTemplate:
            try:
                # Get the correct template and put it in the right placeunclea
                templateFpath = templates(input_yaml=input_yaml).getTemplate()

                if templateFpath is None:
                    # If template file is not found, raise an exception
                    raise Exception("Template file not found")

                if templateFpath is not None:
                    try:
                        templateDir = os.path.dirname(templateFpath)

                        # Copy the template PSF
                        template_psf = glob.glob(os.path.join(templateDir, 'PSF_model_*'))[0]
                        templatePSF_loc = os.path.join(os.path.dirname(fpath), 'template_PSF_model_.fits')
                        shutil.copyfile(template_psf, templatePSF_loc)
                    except:
                        # If copying the template PSF fails, set 'use_zogy' to False
                        input_yaml['template_subtraction']['use_zogy'] = False
                        raise Exception("Failed to copy template PSF")

            except:
                pass
            
                    
        header['FWHM'] = imageFwhm
        
        input_yaml['fwhm']  = imageFwhm
        input_yaml['scale'] = scale
         
        # =============================================================================
        #             
        # =============================================================================

        imageWCS = wcs.WCS(header,relax = True,fix = False)
       
        target_x_pix, target_y_pix = imageWCS.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)
        input_yaml['target_x_pix'] = target_x_pix
        input_yaml['target_y_pix'] = target_y_pix
        
        # =============================================================================
        # 
        # =============================================================================
        # Get the pixel scale in arcseconds
        xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(imageWCS)
        pixel_scale = xy_pixel_scales[0] * 3600
        
        # Update the pixel scale in the input YAML
        input_yaml['pixel_scale'] = pixel_scale
        
        # Write the modified image and header back to the FITS file
        fits.writeto(fpath,
                  image,
                  header,
                  overwrite=True,
                  output_verify='silentfix+ignore')
        
        # =============================================================================
        #             
        # =============================================================================

        # Set range for which PSF model can move around
        input_yaml['dx'] = np.ceil(imageFwhm) 
        input_yaml['dy'] = np.ceil(imageFwhm) 

        doAp = False

        if input_yaml['photometry']['do_ap_phot']:
            doAp = True

        # =============================================================================
        #             Perform photometry on the isolated sources
        # =============================================================================
        
        # Measure initial aperture photometry for isolated sources
        initialAperture = aperture(input_yaml=input_yaml, image=image)
        isolatedSources = initialAperture.measure(sources=isolatedSources, plot=False)

        # Check if finding optimum aperture radius is enabled
        if input_yaml['photometry']['find_optimum_radius']:
            # Measure optimum aperture radius
            optiumumAperture = initialAperture.measure_optimum_radius(sources=isolatedSources, plot=True)
            
            # Check if optimum aperture radius is less than 5 x FWHM
            if optiumumAperture < 5:
                # Update the aperture size in the input YAML
                input_yaml['photometry']['ap_size'] = optiumumAperture
                
                # Calculate the updated scale based on the new aperture size
                updated_scale = np.ceil((input_yaml['photometry']['ap_size'] * input_yaml['fwhm']) + 2 * input_yaml['fwhm'])
                
                # Adjust the scale to the nearest even integer
                scale = int(updated_scale if updated_scale % 2 == 0 else updated_scale + 1) + 0.5
                
                # Log the updated cutout size
                logging.info(f'> Updated cutout size: {2 * scale:.0f} px')
                
                # Update the scale in the input YAML
                input_yaml['scale'] = scale

        # Measure aperture correction for isolated sources
        apertureCorrection, apertureCorrection_err = initialAperture.measure_aperture_correction(isolatedSources,
                                                                                                  nSamples=25,
                                                                                                  fwhm=input_yaml['fwhm'],
                                                                                                  plot=False)
        # =============================================================================
        # 
        # =============================================================================

        # Get the WCS information from the header
        imageWCS = wcs.WCS(header, relax=True, fix=False)

        # Create an instance of the catalog class
        sequenceData = catalog(input_yaml=input_yaml)
        
        
        if input_yaml['catalog']['build_catalog']:
            
            
            uncleanCatalog  = sequenceData.build_complete_catalog(target_coords=target_coords,
                                                                catalog_list = ['refcat','sdss','pan_starrs','apass','2mass'],
                                                                max_seperation = 5)
        else:
            
        
            # Download the catalog data
            uncleanCatalog = sequenceData.download(target_coords=target_coords,
                                   target_name=input_yaml['target_name'],
                                   catalogName=input_yaml['catalog']['use_catalog'],
                                   catalog_custom_fpath=input_yaml['catalog']['catalog_custom_fpath']
                                   )

        # Clean the catalog by removing sources outside the image borders
        cleanCatalog = sequenceData.clean(selectedCatalog=uncleanCatalog,
                                          image_wcs=imageWCS,
                                          catalogName=input_yaml['catalog']['use_catalog'],
                                          get_local_sources=False,
                                          border=2*scale)
        
        border = 2 * scale
        width = image.shape[1]
        height = image.shape[0]
        
        mask_x = (cleanCatalog['x_pix'].values >= border) & (cleanCatalog['x_pix'].values < width - border)
        mask_y = (cleanCatalog['y_pix'].values >= border) & (cleanCatalog['y_pix'].values < height - border)
        
        

        cleanCatalog= cleanCatalog[(mask_x) & (mask_y)]
        
        # Measure the properties of the sources in the cleaned catalog
        cleanCatalog = sequenceData.measure(selectedCatalog=cleanCatalog,
                                            image=image)

        # =============================================================================
        # 
        # =============================================================================
        
        # Check if aperture photometry should be performed or if template preparation is required
        if not doAp or prepareTemplate:
            # Build PSF model and perform PSF photometry
            epsf_model, psfSources = psf(image=image, input_yaml=input_yaml).build(psfSources=isolatedSources)
            
            # print(epsf_model)
            if  not epsf_model:
                doAp = True
                psfSources = None

            else:

                cleanCatalog = psf(image=image, input_yaml=input_yaml).fit(epsf_model=epsf_model,
                                                                           sources=cleanCatalog,
                                                                           plotTarget=False,
                                                                           ignore_sources=bright_sources)
        else:
            psfSources = None
            
        # =============================================================================
        #                 
        # =============================================================================
                    
        # Convert pixel coordinates of isolated sources to world coordinates
        ra_isolatedSources, dec_isolatedSources = imageWCS.wcs_pix2world(isolatedSources.x_pix.values,
                                                                         isolatedSources.y_pix.values,
                                                                         0)

        # Calculate the distance of each isolated source from the target
        dist = pix_dist(isolatedSources.x_pix.values, target_x_pix,
                isolatedSources.y_pix.values, target_y_pix)

        # Add the RA, DEC, and distance columns to the isolatedSources DataFrame
        isolatedSources['RA'] = ra_isolatedSources
        isolatedSources['DEC'] = dec_isolatedSources
        isolatedSources['dist'] = dist

        # =============================================================================
        #                     
        # =============================================================================
        
        # Calculate the zeropoint and clean the catalog
        ZP = zeropoint(input_yaml=input_yaml)
        cleanCatalog = ZP.clean(sources=cleanCatalog)

        # Get the zeropoint and plot the histogram
        cleanCatalog, outputZP = ZP.get(sources=cleanCatalog)
        ZP.plot_histogram(cleanCatalog, aperture_correction=(apertureCorrection, apertureCorrection_err), measured_zeropoint=outputZP)

        # Update the header with the zeropoint values
        for m in outputZP.keys():
            try:
                header[f'zpoint_{m}'] = outputZP[m][0]
                header[f'zpoint_{m}_err'] = outputZP[m][1]
            except:
                header[f'zpoint_{m}'] = 999
                header[f'zpoint_{m}_err'] = 999

        # Write the modified image and header back to the FITS file
        fits.writeto(fpath, image, header, overwrite=True, output_verify='silentfix+ignore')
        
        # =============================================================================
        #    
        # =============================================================================
        # List of DataFrames, ignoring any None values
        
        if epsf_model:
            isolatedSources = psf(image=image, input_yaml=input_yaml).fit(epsf_model=epsf_model,
                                                sources=isolatedSources,
                                                plotTarget=False)

        dataframes = [df for df in [cleanCatalog, isolatedSources] if df is not None]

        # Combine the DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df[['x_pix', 'y_pix']]

        combined_df.sort_values(by='x_pix', inplace=True)

        # Define the tolerance for similarity
        tolerance = np.ceil( 2* input_yaml['fwhm']) 

        # List to keep track of which points are to be removed
        to_remove = []

        # Iterate over DataFrame rows
        for idx, row in combined_df.iterrows():
            if idx in to_remove:
                continue

            other_points = combined_df.drop(index=idx)
            distances = np.sqrt((row['x_pix'] - other_points['x_pix'].values) ** 2 +
                                (row['y_pix'] - other_points['y_pix'].values) ** 2)

            if np.nanmin(distances) < tolerance:
                remove_idx = combined_df.iloc[np.nanargmin(distances)]
                to_remove.append(remove_idx.name)

        to_remove = list(set(to_remove))
        all_sources = combined_df.drop(index=to_remove)

        # Reset index for better readability (optional)
        all_sources.reset_index(drop=True, inplace=True)

        border = 2 * scale
        width = image.shape[1]
        height = image.shape[0]

        mask_x = (all_sources['x_pix'].values >= border) & (all_sources['x_pix'].values < width - border)
        mask_y = (all_sources['y_pix'].values >= border) & (all_sources['y_pix'].values < height - border)

        all_sources = all_sources[(mask_x) & (mask_y)]
        
        
        footprint = circular_footprint(int(np.ceil(input_yaml['fwhm'])))
        all_sources['x_pix'], all_sources['y_pix'] = centroid_sources(image, all_sources['x_pix'], all_sources['y_pix'],
                                                                      footprint=footprint, centroid_func=centroid_2dg)


        ra_isolatedSources, dec_isolatedSources = imageWCS.wcs_pix2world(all_sources.x_pix.values,
                                         all_sources.y_pix.values,
                                         0)

        all_sources['RA'] = ra_isolatedSources
        all_sources['DEC'] = dec_isolatedSources

        # =============================================================================
        # 
        # =============================================================================

        makePlots = plot(input_yaml=input_yaml )

        makePlots.SourceCheck(image = image,
                              psfSources=psfSources,
                              catalogSources=cleanCatalog,
                              fwhmSources=isolatedSources,
                               mask = source_mask
                              )
        
        image_sources = None
        if not (input_yaml['photometry']['perform_global_photometry_sigma']  is None):
            
            try:
                image_sources = find_fwhm(input_yaml=input_yaml).measure_image(image=image,fwhm = imageFwhm,
                                                                                     sigma = input_yaml['photometry']['perform_global_photometry_sigma'])
            
                # Measure initial aperture photometry for isolated sources
                image_sources = aperture(input_yaml=input_yaml, image=image).measure(sources=image_sources, plot=False)
                
                
                image_sources = psf(image=image, input_yaml=input_yaml).fit(epsf_model=epsf_model,
                                                                           sources=image_sources,
                                                                           plotTarget=False)
                
                ra_isolatedSources, dec_isolatedSources = imageWCS.wcs_pix2world(image_sources .x_pix.values,
                                                                                    image_sources .y_pix.values,
                                                                                     0)
    
                image_sources ['RA'] = ra_isolatedSources
                image_sources ['DEC'] = dec_isolatedSources
            except: pass
        
        # =============================================================================
        # 
        # =============================================================================
        if prepareTemplate:
            # Check if the image filter is in ['u','g','r','i','z']
            if imageFilter in ['u','g','r','i','z']:
                imageFilter+='p'
            
            # Create a new basename for the template file
            newBasename = imageFilter+'_template.fits'

            # Log the renaming of the template filename
            logging.info('\nRenaming template filename:\n%s -> %s' % (fpath,os.path.join(cur_dir,newBasename)))

            # Save the cleaned catalog, PSF sources, and all sources to CSV files
            cleanCatalog.to_csv(os.path.join(cur_dir,'imageCalib_template.csv'))
            isolatedSources.to_csv(os.path.join(cur_dir,'psfSources_template.csv'))
            all_sources.to_csv(os.path.join(cur_dir,'all_sources.csv'),index = False)

            # Write the modified image and header to the new FITS file
            fits.writeto(os.path.join(cur_dir,newBasename),
                  image,
                  header,
                  overwrite = True,
                  output_verify = 'silentfix+ignore')
                          
        
            logging.info(f'\nEnd of  {imageFilter} template calibration')
            
            return 1
        
        # =============================================================================
        #             
        # =============================================================================

        subtraction_perform = False

        
        scienceFpath_OG = fpath
        
        if  input_yaml['template_subtraction']['do_subtraction']:

            # from templates import templates
            try:
                #  Get the correct template and put it in the right place
                templateFpath =  templates(input_yaml = input_yaml).getTemplate()
                
                if not templateFpath:
                    raise Exception('No template images found - skipping')

                # align_catalog
                templateFpath,template_lst, science_lst =  templates(input_yaml = input_yaml).align(scienceFpath=fpath,
                                            templateFpath = templateFpath,
                                            imageCatalog = all_sources,
                                            useWCS = not input_yaml['template_subtraction']['use_astroalign'])
                
                image = get_image(fpath)
                header = get_header(fpath)
                imageWCS = wcs.WCS(header,relax = True,fix = False)

                target_x_pix, target_y_pix = imageWCS.all_world2pix(input_yaml['target_ra'],
                                                                    input_yaml['target_dec'],
                                                                    0)
            
                input_yaml['target_x_pix'] = target_x_pix
                input_yaml['target_y_pix'] = target_y_pix
                
                if 1:
                    # Crop the science and template images
                    scienceFpath,templateFpath =  templates(input_yaml = input_yaml).crop(scienceFpath=fpath,
                                                           templateFpath = templateFpath)
                else:
                    scienceFpath = fpath
                    
                header = get_header(scienceFpath)
                imageWCS = wcs.WCS(header,relax = True,fix = False)

                target_x_pix, target_y_pix = imageWCS.all_world2pix(input_yaml['target_ra'],
                                                                    input_yaml['target_dec'],
                                                                    0)
                input_yaml['target_x_pix'] = target_x_pix
                input_yaml['target_y_pix'] = target_y_pix
                    
                cutout_scienceFpath = scienceFpath
                
                try:
                    # Perform template subtraction
                    scienceFpath, subtraction_mask =  templates(input_yaml = input_yaml).subtract(scienceFpath=scienceFpath,
                                    templateFpath=templateFpath,
                                    method =  input_yaml['template_subtraction']['method'])
            
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    logging.info(exc_type, fname, exc_tb.tb_lineno,e)
                    raise Exception ('FIX TEMPLATE SUBTRACTION')
                    
                subtraction_perform  = True

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info(exc_type, fname, exc_tb.tb_lineno,e)
                
                logging.info('\n\nTEMPLATE SUBTRACTION FAILED - Attempting on original image\n\n')
                scienceFpath = scienceFpath_OG
                subtraction_perform = False
        else:
            scienceFpath = fpath

        # =============================================================================
        #             
        # =============================================================================
        
        # Get the image data and check if it is a zero image
        image = get_image(scienceFpath)
        
        if np.sum(image) == 0 and not subtraction_perform:
            logging.info('\n\nTEMPLATE SUBTRACTION RETURN ZERO IMAGE - Attempting on original image\n\n')
            scienceFpath = scienceFpath_OG
            subtraction_perform = False
            image = get_image(scienceFpath)
        
        # Get the header of the image
        header = get_header(scienceFpath)
        # Get the WCS information from the header
        imageWCS = wcs.WCS(header)
     
        # Convert the target coordinates to pixel coordinates
        target_x_pix, target_y_pix = imageWCS.all_world2pix(input_yaml['target_ra'],
                                    input_yaml['target_dec'],
                                    0
                            )

        # Log the location of the transient
        logging.info('\nTransient is located at:\nx =  %.3f\ny =  %.3f' % (target_x_pix, target_y_pix))
        
        # Update the input YAML with the target pixel coordinates
        input_yaml['target_ra'] = target_coords.ra.degree
        input_yaml['target_dec'] = target_coords.dec.degree
        input_yaml['target_x_pix'] = target_x_pix
        input_yaml['target_y_pix'] = target_y_pix
        
        # Check if template subtraction was performed
        if subtraction_perform:
            # Create an instance of the plot class
            makePlots = plot(input_yaml=input_yaml)
            # Plot the template subtraction check
            makePlots.subtractionCheck(image=get_image(cutout_scienceFpath),
                           ref=get_image(templateFpath),
                           diff=image,
                           mark_location=[target_x_pix, target_y_pix],
                           inset_size=scale,
                           mask=subtraction_mask)
            
            
        
        # =============================================================================
        #  Do photometry on target
        # =============================================================================
        
        # Perform initial aperture photometry on the target position
        targetPosition = pd.DataFrame([])
        targetPosition['x_pix'] = np.array([target_x_pix])
        targetPosition['y_pix'] = np.array([target_y_pix])

        # Measure the target position using initial aperture photometry
        initialAperture = aperture(input_yaml=input_yaml, image=image)
        targetPosition = initialAperture.measure(sources=targetPosition, plot=True, saveTarget=True)
        
        idx = 0
        
        prelim_SNR = targetPosition['SNR'].iloc[0]
        
        perform_forcePhotometry = False
        if prelim_SNR < 3 and not doAp:
            logging.info(f'Target not well detected [SNR < {prelim_SNR:.1f}]  performing forced photometry')
            perform_forcePhotometry = True

        # Perform PSF fitting on the target position if aperture photometry is not required
        if not doAp:
            targetPosition = psf(image=image, input_yaml=input_yaml).fit(epsf_model=epsf_model,
                                          sources=targetPosition,
                                          plotTarget=True,
                                          forcePhotometry=perform_forcePhotometry)

        logging.info(border_msg('Targeted photometry on %s' % input_yaml['target_name']))
        
        # =============================================================================
        #             
        # =============================================================================
        
        
        # Convert pixel coordinates to world coordinates
        extracted_position = imageWCS.pixel_to_world(targetPosition['x_pix'], targetPosition['y_pix'])

        # Create a SkyCoord object for the extracted position
        coords_science_i = SkyCoord(extracted_position.ra.degree, extracted_position.dec.degree, unit=(u.deg, u.deg))

        # Calculate the separation between the extracted position and the target coordinates
        sep = coords_science_i.separation(target_coords)

        # Calculate the detection limit
        detection_limit = 3
        target_beta = beta_value(n=detection_limit, sigma=float(targetPosition['noiseSky'].iloc[0]), f_ul=float(targetPosition['maxPixel'].iloc[0]))

        # Log the measured SNR and target detectability
        logging.info('\nTarget location measured with SNR: %.1f' % targetPosition['SNR'].iloc[0])
        logging.info('Target detectability: %.1f %%' % (target_beta * 100))

        # Log the fitted location
        logging.info(f'Fitted location {sep.arcsecond[0]:.1f} arcseconds from expected position\n')
        
        # =============================================================================
        #             
        # =============================================================================

        for method in ['AP','PSF']:
            # Check if the method is available in the outputZP dictionary
            if method not in outputZP:
                logging.info('%s zeropoint not available  - skipping' % method)
                continue

            try:
                # Calculate the calibrated magnitude and error terms
                targetPosition.at[idx,'%s_%s' % (input_yaml['imageFilter'],method)] = targetPosition.at[idx,'inst_%s_%s' % (input_yaml['imageFilter'],method)] + outputZP[method][0]
                
                
                errorTerms = [targetPosition.at[idx,'inst_%s_%s_err' % (input_yaml['imageFilter'],method)], outputZP[method][1]]
                targetPosition.at[idx,'%s_%s_err' % (input_yaml['imageFilter'],method)] = quadratureAdd(errorTerms)

                # Log the calibrated magnitude and error
                logging.info('Calibrated %s %s-band magnitude: %.3f +/- %.3f [mag]'  % (method,input_yaml['imageFilter'], 
                                                    targetPosition.at[idx,'%s_%s' % (input_yaml['imageFilter'],method)],
                                                    targetPosition.at[idx,'%s_%s_err' % (input_yaml['imageFilter'],method)]))
            except Exception as e:
                # Log any issues encountered during calibration
                exc_type, _, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info(exc_type, fname, exc_tb.tb_lineno,e)
                logging.info('Issue measuring target using %s  - skipping\n->> %s' % (method,e))

                # Set the calibrated magnitude and error to NaN
                targetPosition.at[idx,'inst_%s_%s' % (input_yaml['imageFilter'],method)] =np.nan
                targetPosition.at[idx,'inst_%s_%s_err' % (input_yaml['imageFilter'],method)] =np.nan
                targetPosition.at[idx,'%s_%s' % (input_yaml['imageFilter'],method)] = np.nan
                targetPosition.at[idx,'%s_%s_err' % (input_yaml['imageFilter'],method)] = np.nan

        
        # =============================================================================
        #         
        # =============================================================================

        # Calculate the probable and injected detection limits if the target has low SNR
        ProbableLimit = np.nan
        InjectedLimit = np.nan

        if targetPosition.at[idx,'SNR']<5:
            # Log that the target is detected with low SNR
            logging.info(border_msg('Target detected with low SNR [%.1f] - getting detection limits' % targetPosition.at[idx,'SNR']))
            
            # Create an instance of the limits class
            getLimits = limits(input_yaml=input_yaml)

            try:
                # Get the expanded cutout of the image
                expandedCutout = getLimits.getCutout(image = image)

                # Calculate the probable detection limit
                ProbableLimit = getLimits.getProbableLimit(expandedCutout,
                                                            bkg_level = 3,
                                                            detection_limit=3,
                                                            useBeta = True,
                                                            beta =0.75,plot = True)

                # Calculate the injected detection limit
                
                if epsf_model:
                    InjectedLimit = getLimits.getInjectedLimit(expandedCutout,
                                                                initialGuess = ProbableLimit  ,
                                                                detection_limit=3,
                                                                useBeta = True,
                                                                epsf_model=epsf_model,
                                                                beta =0.75,plot = True)
                
                # Log the probable detection limit for each method
                for method in ['AP','PSF']:
                    if method not in outputZP: continue
                    logging.info('\tProbable detection limit (%s): %.3f ' % (method,ProbableLimit + outputZP[method][0] ))
                
                # Log the injected detection limit for each method
                for method in ['AP','PSF']:
                    if method not in outputZP: continue
                    logging.info('\tInjected detection limit (%s): %.3f ' % (method,InjectedLimit + outputZP[method][0] ))
            
            except Exception as e:
                # Log any issues encountered during the calculation of detection limits
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info(exc_type, fname, exc_tb.tb_lineno,e)
                logging.info('Issue measuring limiting magnitude  - skipping\n->> %s' % (e))
                ProbableLimit = np.nan
                InjectedLimit = np.nan
  


        # =============================================================================
        #                 Save output
        # =============================================================================
        # Create an empty dictionary to store the output values
        output = {}

        # Add the relevant information to the output dictionary
        output['filename'] = fpath
        output['date'] = date
        output['mjd'] = date_mjd
        output['telescope'] = telescope
        output['instrument'] = instrument
        output['image_fwhm'] = imageFwhm
        output['exptime'] = input_yaml['exptime']
        output['filter'] = input_yaml['imageFilter']
        output['xpix'] = targetPosition.at[idx, 'x_pix']
        output['ypix'] = targetPosition.at[idx, 'y_pix']
        output['snr'] = targetPosition.at[idx, 'SNR']

        # Convert the pixel coordinates to world coordinates
        extracted_position = imageWCS.pixel_to_world(output['xpix'], output['ypix'])
        output['ra'] = extracted_position.ra.degree
        output['dec'] = extracted_position.dec.degree

        # Add the zeropoint values to the output dictionary
        available_methods = outputZP.keys()
        for method in available_methods:
            if method not in outputZP:
                continue
            output['zp_%s' % method] = outputZP[method][0]
            output['zp_%s_err' % method] = outputZP[method][1]

        # Add the aperture correction values to the output dictionary
        output['aperture_corr'] = [apertureCorrection]
        output['aperture_corr_err'] = [apertureCorrection_err]

        # Add the flux and magnitude values to the output dictionary
        for method in available_methods:
            # if 'inst_%s_%s' % (input_yaml['imageFilter'], method) not in targetPosition.columns:
            #     continue
            output['flux_%s' % (method)] = targetPosition.at[idx, 'flux_%s' % (method)]
            output['inst_%s_%s' % (input_yaml['imageFilter'], method)] = targetPosition.at[idx, 'inst_%s_%s' % (input_yaml['imageFilter'], method)]
            output['inst_%s_%s_err' % (input_yaml['imageFilter'], method)] = targetPosition.at[idx, 'inst_%s_%s_err' % (input_yaml['imageFilter'], method)]
            output['%s_%s' % (input_yaml['imageFilter'], method)] = targetPosition.at[idx, '%s_%s' % (input_yaml['imageFilter'], method)]
            output['%s_%s_err' % (input_yaml['imageFilter'], method)] = targetPosition.at[idx, '%s_%s_err' % (input_yaml['imageFilter'], method)]

        # Add the beta, probable limit, and injected limit to the output dictionary
        output['beta'] = target_beta
        output['lmag_prob'] = ProbableLimit
        output['lmag_inject'] = InjectedLimit

        # Check if template subtraction was performed and add it to the output dictionary
        if subtraction_perform:
            output['subtraction_perform'] = True

    

        # Calculate the elapsed time
        end = time.time() - start
        output['etime'] = end

        # Save the output as a CSV file
        output_df = pd.DataFrame(output, index=[0])
        output_df.to_csv(os.path.join(write_dir, 'output.csv'), index=False, float_format='%.6f')
        

        # Save both the output and the clean catalog to the same CSV file
        # Convert the output dictionary to a string
        output_str = dict_to_string_with_hashtag(output)
        
        # Open the file in write mode to add the output string first
        with open(os.path.join(write_dir, 'CALIB_' + base + '.csv'), 'w') as file:
            # Write the output dictionary string with a header
            file.write("# Output dictionary\n")
            file.write(output_str + "\n")
            
        # Open the file again in append mode to add the clean catalog
        with open(os.path.join(write_dir, 'CALIB_' + base + '.csv'), 'a') as file:
            # Add a header for the clean catalog section
            # file.write("\n# Clean Catalog\n")
            
            # Save the clean catalog as a CSV format and append it
            cleanCatalog.to_csv(file, index=False, float_format='%.6f')



        if not (image_sources is None):
            
            fname = f"sources_{input_yaml['photometry']['perform_global_photometry_sigma']:.0f}sigma_" + base + '.csv'
                    
            # Open the file in write mode to add the output string first
            with open(os.path.join(write_dir, fname), 'w') as file:
                # Write the output dictionary string with a header
                # file.write("# Output dictionary\n")
                file.write(output_str + "\n")
                
            # Open the file again in append mode to add the clean catalog
            with open(os.path.join(write_dir, fname), 'a') as file:
                # Add a header for the clean catalog section
                # file.write("\n# Clean Catalog\n")
                
                # Save the clean catalog as a CSV format and append it
                image_sources.to_csv(file, index=False, float_format='%.6f')
                
            
            
        # Save the all sources as a CSV file
        all_sources.to_csv(os.path.join(cur_dir, 'all_sources.csv'), index=False, float_format='%.6f')
        
        
                      
        print(border_msg(f'Photometric measurements done [{end:.1f}s]',body = '*',corner = '!'))
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.info(exc_type, fname, exc_tb.tb_lineno,e)

    

    return None

if __name__ == "__main__":

    run_photometry()
