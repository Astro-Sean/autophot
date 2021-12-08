#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def main(object_info,autophot_input,fpath):
    '''
    Main function in AutoPHOT to perform photometric reduction and calibration
    
    :param object_info: Dictionary containing transient coordinates
    :type object_info: Dictionary
    :param autophot_input: Main AutoPHOT command dictionary
    :type autophot_input: TYPE
    :param fpath: File path for *FITS* image
    :type fpath: str
    :return: Executions photometric calibration and reduction steps, as well as necessary WCS correction and limiting magnitude tests.
    :rtype: Output files 

    '''


    # ensure to use copy of original inputed synatc instruction files
    autophot_input = autophot_input.copy()
    # Basic Packages
    import sys
    import shutil
    import os
    import numpy as np
    import warnings
    import pandas as pd
    import pathlib
    import collections
    import lmfit
    import time
    import logging
    import datetime
    import astroalign as aa
    import gc
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # from matplotlib.pyplot import Circle
    from os.path import dirname
    # Astrpy and photutils
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import astropy.wcs as wcs
    from astropy.stats import sigma_clipped_stats
    from astropy.time import Time
    from functools import reduce
    from astropy.coordinates import Angle
    from matplotlib.ticker import MultipleLocator
    from astropy.table import Table
    from scipy.stats import binned_statistic
    from astroquery.skyview import SkyView
    from reproject import reproject_interp
    from astropy.wcs import WCS

    from astropy.visualization import  ZScaleInterval
    from autophot.packages.aperture import plot_aperture

    # Proprietary modules developed for AUTOPHOT
    from autophot.packages.functions import  getheader,getimage,calc_mag,calc_mag_error,set_size,pix_dist
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma,gauss_sigma2fwhm
    from autophot.packages.functions import moffat_2d,moffat_fwhm,border_msg
    from autophot.packages.check_wcs import updatewcs,removewcs
    from autophot.packages.call_astrometry_net import AstrometryNetLOCAL
    from autophot.packages.template_subtraction import subtract
    from autophot.packages.call_yaml import yaml_autophot_input as cs
    from autophot.packages.functions import SNR, SNR_err,border_msg
    
    
    from autophot.packages.limit import limiting_magnitude_prob,inject_sources
    from autophot.packages.call_crayremoval import remove_cosmic_rays
    from autophot.packages.functions import arcmins2pixel,pixel2arcsec
    from autophot.packages.find import get_fwhm
    from autophot.packages.aperture import measure_aperture_photometry,find_aperture_correction,do_aperture_photometry,find_optimum_aperture_size
    import autophot.packages.psf as psf
    import autophot.packages.call_catalog as call_catalog
    from autophot.packages.psf import compute_multilocation_err

    from autophot.packages.functions import beta_value
    from autophot.packages.airmass_extinction import find_airmass_extinction
    from autophot.packages.psf import PSF_MODEL
    from autophot.packages.zeropoint import get_zeropoint
    from autophot.packages.template_subtraction import prepare_templates
    from autophot.packages.template_subtraction import get_pstars


    from astropy.nddata.utils import Cutout2D

    from autophot.packages.functions import trim_zeros_slices
    from matplotlib.gridspec import  GridSpec
    
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Start time of this image
    start_time = time.time()
    try:
        # Preparing output dictionary
        output = collections.OrderedDict({})
        # paths can't end with backslash - remove if needed
        if autophot_input['fits_dir'] != None:
            if autophot_input['fits_dir'].endswith('/'):
                autophot_input['fits_dir'] = autophot_input['fits_dir'][:-1]
        if autophot_input['wdir'].endswith('/'):
            autophot_input['wdir'] = autophot_input['wdir'][:-1]
        # =============================================================================
        # Prepare new file - we will work on a copied file and not the original file
        # =============================================================================
        # Change to new working directory set by 'outdir_name' in yaml file
        if autophot_input['fname']:
            wdir  = autophot_input['fname']
            work_loc = str(pathlib.Path(dirname(wdir)))
        else:
            wdir = autophot_input['fits_dir']
            new_dir = '_' + autophot_input['outdir_name']
            base_dir = os.path.basename(wdir)
            work_loc = base_dir + new_dir
            new_output_dir = os.path.join(dirname(wdir),work_loc)

            if not autophot_input['template_subtraction']['prepare_templates']:

                # create new output directory is it doesn't exist
                pathlib.Path(new_output_dir).mkdir(parents = True, exist_ok=True)
                os.chdir(new_output_dir)
        # Take in original data, copy it to new location
        # New file will have _APT append to fname
        # Create names of directories with name=filename and account for correct sub directories locations
        base_wext = os.path.basename(fpath)
        base = os.path.splitext(base_wext)[0]
        fname_ext = os.path.splitext(base_wext)[1]

        # Replace Whitespaces with underscore
        base = base.replace(' ','_')
        # Replace full stops (excluding the extension) with underscore
        base = base.replace('.','_')

        # If file already has AUTOPHOT in it - for debugging
        base = base.replace('_APT','') 

        base = base.replace('_ERROR','')
        # Need special criteria for working on single file and/or template files
        if not autophot_input['fname'] and not autophot_input['template_subtraction']['prepare_templates']:
            root = dirname(fpath)
            sub_dirs = root.replace(wdir,'').split('/')
            sub_dirs = [i.replace('_APT','').replace(' ','_') for i in sub_dirs]
            cur_dir = new_output_dir
        else:

            sub_dirs = ['']
            cur_dir = dirname(wdir)

        # This is the name of the file without any extension
        autophot_input['base'] = base
        # Move through list of subdirs,
        # where the final copied file will be moved
        # creating new sub directories with folder name
        # extension until it reaches folder
        # it will then create a new folder with name of file to which
        # every output file is moved to

        # Remove duplicates
        sub_dirs = list(set([i.replace('_APT','').replace(base,'') for i in sub_dirs]))
        # print(sub_dirs) 
        if not autophot_input['template_subtraction']['prepare_templates']:
            for i in range(len(sub_dirs)):
                if i: # If directory is not blank
                     pathlib.Path(cur_dir +'/' + sub_dirs[i]+'_APT').mkdir(parents = True,exist_ok=True)
                     cur_dir = cur_dir +'/' + sub_dirs[i]+'_APT'
        # create filepath of write directory
        if not autophot_input['template_subtraction']['prepare_templates']:

            cur_dir = cur_dir + '/' + base
            pathlib.Path(cur_dir).mkdir(parents = True,exist_ok=True)

            # copy new file to new directory
            shutil.copyfile(fpath, (cur_dir+'/'+base + '_APT'+fname_ext).replace(' ','_'))
        else:

            cur_dir = os.path.dirname(fpath)

        # new fpath for working fits file
        if not autophot_input['template_subtraction']['prepare_templates']:
            fpath = os.path.join(cur_dir, base + '_APT'+fname_ext)
        # Set location of new (copied) file
        autophot_input['fpath'] = fpath
        # base is name [without extension]
        base = os.path.basename(fpath)
        # write dir is where all files will be saved, pre-iteration
        write_dir = (cur_dir + '/').replace(' ','_')
        autophot_input['write_dir'] = write_dir
        # Get image and header from function library
        image    = getimage(fpath)
        headinfo = getheader(fpath)
        if object_info == None:
            sys.exit('No Target Info')
        if autophot_input == None:
            sys.exit("No autophot_input input file")
        if autophot_input['template_subtraction']['prepare_templates']:
            if 'PSF_model' in fpath:
                print('Preparing Template files - ignoring PSF models')
                return

        # =============================================================================
        # Set up logging file
        # =============================================================================
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        # set up logging to file
        logging.basicConfig(level=logging.INFO,
                            format='%(filename)s-%(levelname)s: %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(cur_dir,str(base)+'.log'),
                            filemode='w')
        console = logging.StreamHandler()
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.info('File: '+str(base) + ' - PID: '+str(os.getpid()))
        logging.info('Start Time: %s' % str(datetime.datetime.now()) )
        #==============================================================================
        # Main YAML input and autophot_input files
        #==============================================================================
        # catalog autophot_input contains header information of selected
        # header given by 'catalog' keyword in yaml file
        filepath ='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
        catalog_autophot_input_yml = 'catalog.yml'
        catalog_autophot_input = cs(os.path.join(filepath+'/databases',catalog_autophot_input_yml),autophot_input['catalog']['use_catalog']).load_vars()
        
        
        base_filepath ='/'.join(os.path.os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])
        filters_yml = 'filters.yml'
        filters_input = cs(os.path.join(base_filepath+'/databases',filters_yml )).load_vars()
                                   
        # available_filters = list(filters_input['W_eff'].keys())
        
        
        
        # =============================================================================
        # Get time of observation
        # =============================================================================
        # Geting date of observation from header info
        # if 'MJD-OBS' in headinfo, use that
        # if not:
        #     look for 'DATE-AVG' or 'DATE-OBS' and convert to mjd
        # if not:
        #     return np.nan and continue onwards
        try:
            if 'MJD-OBS' in headinfo:
                mjd_date = headinfo['MJD-OBS']
            else:
                if 'DATE-OBS' in headinfo:
                    time_obs = headinfo['DATE-OBS']
                if 'DATE-AVG' in headinfo:
                    time_obs = headinfo['DATE-AVG']
                time_obs_iso = Time(time_obs,  format='isot')
                mjd_date = time_obs_iso.mjd
                headinfo['MJD-OBS'] = (mjd_date,'Modified Julian Date by AutoPhOT')
        except:
            logging.warning(' Modied Julian Date NOT FOUND' )
            mjd_date = np.nan
            pass
        # Find exact time of observations for airmass calculations

        try:
            try:
                time_obs = headinfo['DATE-AVG']
            except:
                time_obs = headinfo['DATE-OBS']
            time_obs_iso = Time(time_obs,  format='isot')
            autophot_input['obs_time'] = time_obs_iso.mjd
        except:
            autophot_input['obs_time'] = mjd_date

        # =============================================================================
        # Telescope and instrument parameters
        # =============================================================================
        if autophot_input['target_name'] != None:
            # Get target info from [pre-saved] TNS_response

            target_ra  = object_info['ra']
            target_dec = object_info['dec']

            target_prefix = object_info['name_prefix']
            target_coords = SkyCoord(target_ra , target_dec ,unit = (u.hourangle,u.deg))
            autophot_input['target_ra'] = target_coords.ra.degree
            autophot_input['target_dec']= target_coords.dec.degree
        elif autophot_input['target_ra'] != None and autophot_input['target_dec'] != None:
            target_coords = SkyCoord(autophot_input['target_ra'] , autophot_input['target_dec'] ,unit = (u.deg,u.deg))
            autophot_input['target_ra'] = target_coords.ra.degree
            autophot_input['target_dec']= target_coords.dec.degree
        else:
            try:
                if autophot_input['use_header_radec']:
                    # use RA and Dec coordinates from header file
                    # TODO: Address this in a future version
                    raise  Exception('Not available in current version')
                    target_coords = SkyCoord(headinfo['CAT-RA'] , headinfo['CAT-DEC'] ,unit = (u.hourangle,u.deg))
                    autophot_input['target_ra'] =  target_coords.ra.degree
                    autophot_input['target_dec']=  target_coords.dec.degree
            except:
                logging.warning('NO RA:DEC keywords found')

        # Name of target with prefix - if available
        if not (autophot_input['target_name'] is None):

             tname = target_prefix + ' ' + autophot_input['target_name']

        elif not (autophot_input['target_ra'] is None) and not (autophot_input['target_dec'] is None):

             tname = 'RA:  %.6f\nDEC: %.6f' % (autophot_input['target_ra'],autophot_input['target_dec'])
             # target_prefix = ''

        else:

             tname = 'Center of Field'
             # target_prefix = ''
        # =============================================================================
        # Telescope and instrument parameters
        # =============================================================================
        # Get f.o.v from tele_autophot_input if needed, for wcs and upate autophot_input dictionary
        # if f.o.v parametrs are not present, set guess_scale in astometry.net
        try:
            telescope = headinfo['TELESCOP']
        except:
            telescope  = 'UNKNOWN'
            headinfo['TELESCOP'] = 'UNKNOWN'
        if telescope  == '':
            telescope  = 'UNKNOWN'
            headinfo['TELESCOP'] = 'UNKNOWN'
        inst_key = 'INSTRUME'
        inst = headinfo[inst_key]
        tele_autophot_input_yml = 'telescope.yml'
        teledata = cs(os.path.join(autophot_input['wdir'],tele_autophot_input_yml))
        tele_autophot_input = teledata.load_vars()
        
        
        autophot_input['mjd'] = mjd_date
        # Update outputs with filename, telescope and observation time in mjd
        output.update({'fname':fpath})
        output.update({'telescope':tele_autophot_input[telescope][inst_key][inst]['Name']})
        output.update({'TELESCOP':telescope})
        output.update({'INSTRUME':inst_key})
        output.update({'instrument':inst})
        output.update({'mjd':mjd_date})
        autophot_input['tele'] = telescope
        autophot_input['inst'] = inst

        if 'NAXIS1' in headinfo and 'NAXIS' in headinfo:
            autophot_input['NAXIS1'] = headinfo['NAXIS1']
            autophot_input['NAXIS2'] = headinfo['NAXIS2']
        else:
            autophot_input['NAXIS1'] = image.shape[0]
            autophot_input['NAXIS2'] = image.shape[1]

        # =============================================================================
        # Trim image if desired   
        # =============================================================================

        #TODO: replace this with cutout2d

        if autophot_input['preprocessing']['trim_edges'] and 'TRIMMED' not in headinfo:
            logging.info('\nTrimming edges of image by +/- %d pixels' % autophot_input['trim_edges_pixels'])

            image_trim = image[autophot_input['preprocessing']['trim_edges_pixels']:image.shape[0]-autophot_input['preprocessing']['trim_edges_pixels'],
                               autophot_input['preprocessing']['trim_edges_pixels']:image.shape[1]-autophot_input['preprocessing']['trim_edges_pixels']]

            logging.info('Image shape (%d,%d) -> (%d,%d)' % (image.shape[0],image.shape[1],image_trim.shape[0],image_trim.shape[1]))

            headinfo['TRIMMED'] = True

            fits.writeto(fpath,image_trim,
                         headinfo,
                         overwrite = True,
                         output_verify = 'silentfix+ignore')

            image = image_trim

        # =============================================================================
        #  Saturation  level
        # =============================================================================

        try:
            autophot_input['sat_lvl'] = headinfo['SATURATE']
        except:
            autophot_input['sat_lvl'] = 2**16
        # =============================================================================
        # Find filter
        # =============================================================================
        # Get correct filter keyword with the default being 'FILTER'
        # In collaboration with 'write_yaml function' will search for 'FILTER' using filter_key_0 key
        # if found sets 'filter_key' in autophot_input file to correct filter header
        # if not:
        #     file search for filter_key_1 key in telescope_autophot_input and check
        #     if result value in headinfo.
        #           Will continue until no more filter_key_[] are in telescope_autophot_input or
        #           right keyword is found
        # if fails:
        #     returns filter_key = 'no_filter'
        # Was implemented to allow for,  same telescope/instrument with different header name for the filter keyoward

        # These are the keys we want to avoid
        avoid_keys = ['clear','open']

        open_filter = False
        found_correct_key = False
        filter_keys = [i for i in list(tele_autophot_input[telescope][inst_key][inst]) if i.startswith('filter_key_')]
        for filter_header_key in filter_keys:
            if tele_autophot_input[telescope][inst_key][inst][filter_header_key] not in list(headinfo.keys()):
                continue
            if headinfo[tele_autophot_input[telescope][inst_key][inst][filter_header_key]].lower() in avoid_keys:
                open_filter = True
                continue
            if headinfo[tele_autophot_input[telescope][inst_key][inst][filter_header_key]] in tele_autophot_input[telescope][inst_key][inst]:
                found_correct_key = True
                break
        if not found_correct_key and open_filter:
            logging.info('Cannot find filter key or filter key is set to OPEN')
            autophot_input['filter_key'] = 'no_filter'
            use_filter = 'no_filter'
        else:
            autophot_input['filter_key'] = tele_autophot_input[telescope][inst_key][inst][filter_header_key]
            use_filter =  tele_autophot_input[telescope][inst_key][inst][str(headinfo[autophot_input['filter_key']])]

        # Set filter keyword in yaml file
        autophot_input['image_filter'] = use_filter
        try:
            headinfo[autophot_input['filter_key']]
        except:
            autophot_input['filter_key'] = 'no_filter'
            logging.warning('Filter keywoard == no_filter')
        logging.info('Filter keyoward used: %s' % autophot_input['filter_key'] )
        # If no filter is detected
        if use_filter.lower() == 'no_filter':

            if autophot_input['force_filter'] != 'None':

                use_filter = autophot_input['force_filter']

            if autophot_input['force_filter'].lower() != 'clear':

                use_filter = autophot_input['force_filter']
            else:
                # List of filters, will check spread and return most appropiate filter
                use_filter = autophot_input['filter_key']

        # =============================================================================
        # Prepare templates
        # =============================================================================
        if telescope == 'MPI-2.2' and use_filter in ['J','H','K']:
            IR_gain_key = '%s_GAIN' % use_filter
            logging.info('Detected GROND IR - setting GAIN key to %s_GAIN' % IR_gain_key)
            tele_autophot_input[telescope][inst_key][inst]['gain'] = IR_gain_key
            
        if autophot_input['template_subtraction']['prepare_templates']:

            # Will just work on /template/ folder
           
            # If telescope if GROND check pixelscale
            if telescope == 'MPI-2.2':
                if use_filter in ['J','H','K']:
                    logging.info('Detected GROND IR - setting pixel scale to 0.3')
                    autophot_input['pixel_scale'] = 0.3
                else:
                    logging.info('Detected GROND Optical - setting pixel scale to 0.16')
                    autophot_input['pixel_scale'] = 0.16
            else:

                autophot_input['pixel_scale']   = tele_autophot_input[telescope][inst_key][inst]['pixel_scale']

            prepare_templates(fpath,
                              wdir,
                              write_dir,
                              tele_autophot_input=tele_autophot_input ,
                                get_fwhm = True,
                                build_psf = True,
                                clean_cosmic = True,
                                use_astroscrappy = True,
                                solve_field_exe_loc = autophot_input['wcs']['solve_field_exe_loc'],
                                use_lacosmic = False,
                                use_filter = use_filter,
                                redo_wcs = True,
                                target_ra = autophot_input['target_ra'],
                                target_dec =  autophot_input['target_dec'],
                                search_radius = 0.5,
                                cpu_limit= 180,
                                downsample = 2,
                                threshold_value = 25,
                                ap_size =  autophot_input['photometry']['ap_size'],
                                inf_ap_size = autophot_input['photometry']['inf_ap_size'],
                                r_in_size = autophot_input['photometry']['r_in_size'],
                                r_out_size = autophot_input['photometry']['r_out_size'],
                                use_moffat = autophot_input['fitting']['use_moffat'],
                                psf_source_no = autophot_input['psf']['psf_source_no'],
                                fitting_method = autophot_input['fitting']['fitting_method'],
                                regriding_size  = autophot_input['psf']['regriding_size'],
                                fitting_radius =autophot_input['fitting']['fitting_radius'])

            return
        # ============================================================================
        # Begin photometric reductions
        # =============================================================================
        start = time. time()
        logging.info('Telescope: %s' % telescope)
        logging.info('Filter: %s'% use_filter)
        logging.info('MJD: %.3f' % mjd_date)
        date = Time([mjd_date], format='mjd', scale='utc')
        logging.info('Date of Observation : %s' % date.iso[0].split(' ')[0])

        # =============================================================================
        # File deduction - beginning of photometry reductions
        # =============================================================================
        try:
            try:
                fpath = reduce(fpath,use_filter,autophot_input)
                image    = fits.getdata(fpath)
                
     
                    
                    
                headinfo = getheader(fpath)
            except:
                pass
            
            if fpath == None:
                raise Exception

            # =============================================================================
            #        Get useful paramters from telescope.yml 
            # =============================================================================
            if 'AIRMASS' in tele_autophot_input[telescope][inst_key][inst]:
                autophot_input['AIRMASS'] = tele_autophot_input[telescope][inst_key][inst]['AIRMASS']

                AIRMASS_key = autophot_input['AIRMASS']

                if AIRMASS_key is None:
                    AIRMASS = np.nan
                elif AIRMASS_key in headinfo:
                    AIRMASS = headinfo[AIRMASS_key]
                else:
                    AIRMASS = np.nan

            if 'RDNOISE' in tele_autophot_input[telescope][inst_key][inst]:

                rdnoise_key = tele_autophot_input[telescope][inst_key][inst]['RDNOISE']

                if rdnoise_key is None:
                    rdnoise = 0 
                elif rdnoise_key in headinfo:
                    rdnoise = headinfo[rdnoise_key]
                else:
                    rdnoise = 0 
                 
            else:
                logging.info('Read noise key not found')
                rdnoise = 0
                
            logging.info('Read Noise: %.1f [e^- /pixel]' % rdnoise)
            autophot_input['rdnoise'] = rdnoise

            if 'GAIN' in tele_autophot_input[telescope][inst_key][inst]:

                GAIN_key = tele_autophot_input[telescope][inst_key][inst]['GAIN']

                if GAIN_key is None:
                    GAIN = 1
                elif GAIN_key in headinfo:
                    GAIN = headinfo[GAIN_key]
                else:
                    GAIN=1

                autophot_input['gain'] = GAIN
                

          
            logging.info('GAIN: %.1f [e^- /count]' % GAIN)
            # ============================================v=================================
            # Expsoure time
            # =============================================================================

            try:
                # TODO: put this into telescope.yml
                for i in ['EXPTIME','EXP_TIME','TIME-INT']:
                    if i in headinfo:
                        exp_time = headinfo[i]
            except:
                logging.warning('Cannot find exposure time: setting to 1')
                exp_time = 1
                
            # Known issue -> header values written as string
            # Below just splits them apart
            if isinstance(exp_time, str):
               exp_time_split = exp_time.split('/')
               if len(exp_time_split)>1:
                   exp_time = float(exp_time_split[0])
                   
                   
            autophot_input['exp_time'] = exp_time

            logging.info('Exposure time: %.f [s] ' % exp_time)

            # =============================================================================
            # Get pixel scale of image
            # =============================================================================

            if 'pixel_scale' in tele_autophot_input[telescope][inst_key][inst]:

                autophot_input['pixel_scale']   = tele_autophot_input[telescope][inst_key][inst]['pixel_scale']
            else:
                logging.warning('Pixel scale not found in telescope.yml')

                try:
                    w1 = wcs.WCS(headinfo)# WCS values

                    xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(w1)
                    autophot_input['pixel_scale'] = xy_pixel_scales[0]
                except:

                    logging.warning('Could not guess pixel scale - setting to 0.25')
                    autophot_input['pixel_scale'] = 0.25

             # If telescope if GROND check pixel scale
            if telescope == 'MPI-2.2':
                if use_filter in ['J','H','K']:
                    logging.info('Detected GROND IR - setting pixel scale to 0.6')
                    autophot_input['pixel_scale'] = 0.6
                else:
                    autophot_input['pixel_scale'] = 0.158

            # =============================================================================
            # Cosmic ray removal usign astroscrappy
            # =============================================================================

            if autophot_input['cosmic_rays']['remove_cmrays']:
                try:
                # if cosmic rays have no already been removed
                        if 'CRAY_RM'  not in headinfo:
                            headinfo = getheader(fpath)
                            # image with cosmic rays
                            image_old = fits.PrimaryHDU(image)
                            image = remove_cosmic_rays(image_old,
                                                     gain = GAIN,
                                                   
                                                     use_lacosmic = autophot_input['cosmic_rays']['use_lacosmic'])

                            # Update header and write to new file
                            headinfo['CRAY_RM'] = ('T', 'Comsic rays w/astroscrappy ')
                            fits.writeto(fpath,
                                         image,
                                         headinfo,
                                         overwrite = True,
                                         output_verify = 'silentfix+ignore')
                            logging.info('Cosmic rays removed - image updated')

                        else:

                            logging.info('Cosmic sources pre-cleaned - skipping!')
                except Exception as e:
                    logging.exception(e)

            # =============================================================================
            # WCS Check and target
            # =============================================================================
            # -- Various instances of when/if to query using astrometry.net --
            # if any instance of wcs_keywords are not found in the header infomation it
            # if succesful, it will add the UPWCS = T header/value to the header
            # hdu of the newly created file

            try:

                w1_old = wcs.WCS(headinfo)
                existing_WCS = True

            except:

                logging.info('No WCS found')
                existing_WCS = False
            if autophot_input['wcs']['remove_wcs']  or not existing_WCS and 'UPWCS' not in headinfo:
                logging.info('\nPerforming Astrometry.net')
                new_header = removewcs(headinfo,delete_keys = True)

                fits.writeto(fpath,image,
                             new_header,
                             overwrite = True,
                             output_verify = 'silentfix+ignore')

                headinfo = getheader(fpath)
            # search keywords for wcs validation
            wcs_keywords = ['CD1_1','CD1_2','CD2_1','CD2_2',
                            'CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                            'CDELT1','CDELT2','CTYPE1','CTYPE2']
            
            # if no wcs values are found in headinfo, ignore file and exit loop (raise exception)
            if autophot_input['wcs']['ignore_no_wcs']:
                if any(i not in headinfo for i in wcs_keywords):
                    logging.info('No wcs found - ignoring_wcs setting == True')
                    raise Exception('ignore files w/o WCS')
                    
            if 'UPWCS'  in headinfo:
                # if UPWCS already excecuetde and found in header continue
                logging.info('Astrometry.net already excuted')
            elif all(i not in headinfo for i in wcs_keywords):

                # Try to solve for WCS
                # Call local instance of astrometry.net alogorithm
                # astrometry.net documentation:
                # https://buildmedia.readthedocs.org/media/pdf/astrometrynet/latest/astrometrynet.pdf

                logging.info('No WCS values found - attempting to solve field')
                if autophot_input['wcs']['use_xylist']:
                    _,df,_,_ = get_fwhm(image,
                                                           write_dir,
                                                           base,
                                                           threshold_value = autophot_input['source_detection']['threshold_value'],
                                                           fwhm_guess = autophot_input['source_detection']['fwhm_guess'],
                                                           bkg_level = autophot_input['fitting']['bkg_level'],
                                                           max_source_lim = autophot_input['source_detection']['max_source_lim'],
                                                           min_source_lim = autophot_input['source_detection']['min_source_lim'],
                                                           int_scale = autophot_input['source_detection']['int_scale'],
                                                           fudge_factor = autophot_input['source_detection']['fudge_factor'],
                                                           fine_fudge_factor = autophot_input['source_detection']['fine_fudge_factor'],
                                                           source_max_iter = autophot_input['source_detection']['source_max_iter'],
                                                           sat_lvl = autophot_input['sat_lvl'],
                                                           lim_theshold_value = autophot_input['source_detection']['lim_theshold_value'],
                                                           scale_multipler = autophot_input['source_detection']['scale_multipler'],
                                                           sigmaclip_FWHM = autophot_input['source_detection']['sigmaclip_FWHM'],
                                                           sigmaclip_FWHM_sigma = autophot_input['source_detection']['sigmaclip_FWHM_sigma'],
                                                           sigmaclip_median = autophot_input['source_detection']['sigmaclip_median'],
                                                           isolate_sources = autophot_input['source_detection']['isolate_sources'],
                                                           isolate_sources_fwhm_sep = autophot_input['source_detection']['isolate_sources_fwhm_sep'],
                                                           init_iso_scale = autophot_input['source_detection']['init_iso_scale'],
                                                           remove_boundary_sources = autophot_input['source_detection']['remove_boundary_sources'],
                                                           pix_bound = autophot_input['source_detection']['pix_bound'],
                                                           sigmaclip_median_sigma = autophot_input['source_detection']['sigmaclip_median_sigma'],
                                                           save_FWHM_plot = autophot_input['source_detection']['save_FWHM_plot'],
                                                           plot_image_analysis = autophot_input['source_detection']['plot_image_analysis'],
                                                           save_image_analysis = autophot_input['source_detection']['save_image_analysis'],
                                                           use_local_stars_for_FWHM = autophot_input['photometry']['use_local_stars_for_FWHM'],
                                                           prepare_templates = autophot_input['template_subtraction']['prepare_templates'],
                                                           image_filter = autophot_input['image_filter'],
                                                           target_name = autophot_input['target_name'],
                                                           target_x_pix = None,
                                                           target_y_pix = None,
                                                           local_radius = autophot_input['photometry']['local_radius'],
                                                           mask_sources = autophot_input['preprocessing']['mask_sources'],
                                                           # mask_sources_XY_R = None,
                                                           remove_sat = autophot_input['source_detection']['remove_sat'],
                                                           use_moffat = autophot_input['fitting']['use_moffat'],
                                                           default_moff_beta = autophot_input['fitting']['default_moff_beta'],
                                                           vary_moff_beta = autophot_input['fitting']['vary_moff_beta'],
                                                           max_fit_fwhm = autophot_input['source_detection']['max_fit_fwhm'],
                                                           fitting_method = autophot_input['fitting']['fitting_method'],
                                                           use_catalog = autophot_input['source_detection']['use_catalog'] )

                    # df = df[(df['include_fwhm']) & (df['include_median'])]
                    n = np.vstack([df['x_pix'],df['y_pix']]).T
                    tab = Table(n,names = ['x','y'])
                    bintab = fits.BinTableHDU(tab)
                    fpath_BINTABLE = os.path.join(write_dir,'sources_'+base)
                    bintab.writeto(fpath_BINTABLE,overwrite=True)
                    fpath_astrometry  = fpath_BINTABLE
                else:
                    fpath_astrometry  = fpath
                    
                    
                # Run local instance of Astrometry.net - returns filepath of wcs file
                astro_check = AstrometryNetLOCAL(fpath_astrometry,
                                               NAXIS1 = autophot_input['NAXIS1'],
                                               NAXIS2 = autophot_input['NAXIS2'],
                                               solve_field_exe_loc = autophot_input['wcs']['solve_field_exe_loc'],
                                               pixel_scale = autophot_input['pixel_scale'],
                                               ignore_pointing = autophot_input['wcs']['ignore_pointing'],
                                               target_ra = autophot_input['target_ra'],
                                               target_dec = autophot_input['target_dec'],
                                               search_radius = autophot_input['wcs']['search_radius'],
                                               downsample = autophot_input['wcs']['downsample'],
                                               cpulimit = autophot_input['wcs']['cpulimit'])
                                               

                old_headinfo = getheader(fpath)
                try:
                    # Open wcs fits file with wcs values
                    new_wcs  = fits.open(astro_check,ignore_missing_end = True)
                    new_wcs_header = new_wcs[0].header
                    # script used to update per-existing header file with new wcs values
                    headinfo_updated = updatewcs(old_headinfo,new_wcs_header )

                    # close the wcs file
                    new_wcs.close()

                    # update header to show wcs has been checked
                    headinfo_updated['UPWCS'] = ('T', 'WCS by APT')
                    updated_wcs = True
                except:
                    if not existing_WCS or autophot_input['wcs']['force_wcs_redo']:
                        raise Exception('No WCS found and could not solve with Astrometry.net: skipping file')
                    logging.info('Astrometry Failed - trying with original WCS')
                    new_wcs = w1_old
                    old_headinfo.update(w1_old.to_header())
                    headinfo_updated = old_headinfo
                    # update header to show wcs has been checked
                    headinfo_updated['UPWCS'] = ('F', 'NOT WCS by APT')
                    updated_wcs = False
                # Write new header
                fits.writeto(fpath,image,
                             headinfo_updated,
                             overwrite = True,
                             output_verify = 'silentfix+ignore')
                headinfo = getheader(fpath)
                logging.info('WCS saved to new file')
                
                if autophot_input['wcs']['update_wcs_scale'] and updated_wcs == True:
                    'Update image scale params from '
                    logging.info('Update scale units from astrometry.net')
                    new_wcs  = fits.open(astro_check,ignore_missing_end = True)
                    new_wcs_header = new_wcs[0].header
                    find_scale =[c for c in list(new_wcs_header['comment']) if "scale" in c if 'arcsec/pix' in c]
                    pixel_scale =[s for s in find_scale if s.split(':')[0] =='scale']
                    pixel_scale_value = round(float(pixel_scale[0].split(':')[1].split(' ')[1]),3)
                    teledata.update_var(telescope,inst_key,inst,'pixel_scale',pixel_scale_value)
                    new_wcs.close()

            else:

                logging.info('WCS found')

            #==============================================================================
            # Load target information from TNS query if needed
            #==============================================================================
            w1 = wcs.WCS(headinfo)# WCS values
            
            if autophot_input['target_name'] == None:

                # If not target coords are given, will use center of field
                if autophot_input['target_ra'] == None and autophot_input['target_dec'] == None:
                   # if no object is given i.e name,ra,dec then take the middle
                   # of the screen as the target and get region around it

                   # translate pixel values to ra,dec at center of image
                   center = w1.all_pix2world([image.shape[1]/2],[image.shape[0]/2],1)
                   
                   # get ra,dec in deg/SkyCoord format
                   target_coords = SkyCoord(center[0][0] , center[1][0] ,unit = (u.deg,u.deg))
                   
                   # update autophot_input file
                   autophot_input['target_ra'] = target_coords.ra.degree
                   autophot_input['target_dec']= target_coords.dec.degree
                   
                   # Target coords are now set to cwnter of image
                   autophot_input['target_x_pix'] = image.shape[1]/2
                   autophot_input['target_y_pix'] = image.shape[0]/2
                   
                   target_x_pix = image.shape[1]/2
                   target_y_pix = image.shape[0]/2
                   
                else:
                    # if no name is given but ra and dec are, use those instead:

                   target_ra  = autophot_input['target_ra']
                   target_dec = autophot_input['target_dec']
                   
                   target_coords = SkyCoord(target_ra , target_dec ,unit = (u.deg,u.deg))
                   target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)
                   
                   autophot_input['target_ra'] = target_coords.ra.degree
                   autophot_input['target_dec']= target_coords.dec.degree
                   
                   autophot_input['target_x_pix'] = target_x_pix
                   autophot_input['target_y_pix'] = target_y_pix
                   
            elif autophot_input['target_name'] != None:
                try:

                    # Get target info from [pre-saved] TNS_response

                    target_ra  = object_info['ra']
                    target_dec = object_info['dec']
                    target_coords = SkyCoord(target_ra , target_dec ,unit = (u.hourangle,u.deg))
                    target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)
                    autophot_input['target_ra'] = target_coords.ra.degree
                    autophot_input['target_dec']= target_coords.dec.degree
                    autophot_input['target_x_pix'] = target_x_pix
                    autophot_input['target_y_pix'] = target_y_pix

                except:

                    raise Exception('Failed to converg on target position - Are you sure %s is in this image?' % autophot_input['target_name'])
            else:

               try:

                   target_coords = SkyCoord(headinfo['RA'] , headinfo['DEC'] ,unit = (u.hourangle,u.deg))
                   target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)
                   autophot_input['target_ra'] =  target_coords.ra.degree
                   autophot_input['target_dec']=  target_coords.dec.degree
                   autophot_input['target_x_pix'] = target_x_pix
                   autophot_input['target_y_pix'] = target_y_pix
               except Exception as e:

                   logging.exception(e+'> NO RA:DEC keywords found - attempting astrometry without <')

            # =============================================================================
            # Check that the transient is actually in the image     
            # =============================================================================

            if target_x_pix < 0 or target_x_pix> image.shape[1] or target_y_pix < 0 or target_y_pix> image.shape[0] :
                    raise Exception ( ' *** EXITING - Target pixel coordinates outside of image [%s , %s] *** ' % (int(target_x_pix), int(target_y_pix)))

            # =============================================================================
            # Use sources near the target location       
            # =============================================================================

            xy_pixel_scales = wcs.utils.proj_plane_pixel_scales(w1)

            if autophot_input['photometry']['use_local_stars'] or \
                autophot_input['photometry']['use_local_stars_for_PSF'] or \
                    autophot_input['photometry']['use_local_stars_for_FWHM']:

                # transform armin to pixel values
                local_radius =  arcmins2pixel(autophot_input['photometry']['use_source_arcmin'],xy_pixel_scales[0])

                autophot_input['photometry']['local_radius'] = local_radius
                logging.info('Using stars within %d arcmin [%d px]' % (autophot_input['photometry']['use_source_arcmin'],local_radius))
           
            # =============================================================================
            # Mask sources and/or galaxies
            # =============================================================================

            autophot_input['preprocessing']['mask_sources_XY_R']=[]

            if autophot_input['preprocessing']['mask_sources']:
                for RA_mask,DEC_mask,R_mask in autophot_input['preprocessing']['mask_sources_RADEC_R']:
                    mask_radius = Angle(float(R_mask), u.arcmin)
                    local_points = w1.all_world2pix([RA_mask, RA_mask+mask_radius.degree],
                                                    [DEC_mask,DEC_mask+mask_radius.degree],1)
                    mask_radius = pix_dist(local_points[0][0],local_points[0][1],  local_points[1][0],local_points[1][1])
                    autophot_input['preprocessing']['mask_sources_XY_R'].append((local_points[0][0],local_points[1][0],float(mask_radius)))

            #==============================================================================
            # FWHM - Using total image source detection
            #==============================================================================
            # get approx fwhm, dataframe of sources used and updated autophot_input
            # returns fwhm from gaussian fit - and dataframe of sources used
            image_fwhm,df,scale,image_params = get_fwhm(image,
                                                   write_dir,
                                                   base,
                                                   threshold_value = autophot_input['source_detection']['threshold_value'],
                                                   fwhm_guess = autophot_input['source_detection']['fwhm_guess'],
                                                   bkg_level = autophot_input['fitting']['bkg_level'],
                                                   max_source_lim = autophot_input['source_detection']['max_source_lim'],
                                                   min_source_lim = autophot_input['source_detection']['min_source_lim'],
                                                   int_scale = autophot_input['source_detection']['int_scale'],
                                                   fudge_factor = autophot_input['source_detection']['fudge_factor'],
                                                   fine_fudge_factor = autophot_input['source_detection']['fine_fudge_factor'],
                                                   source_max_iter = autophot_input['source_detection']['source_max_iter'],
                                                   sat_lvl = autophot_input['sat_lvl'],
                                                   lim_threshold_value = autophot_input['source_detection']['lim_threshold_value'],
                                                   scale_multipler = autophot_input['source_detection']['scale_multipler'],
                                                   sigmaclip_FWHM_sigma = autophot_input['source_detection']['sigmaclip_FWHM_sigma'],
                                                   isolate_sources_fwhm_sep = autophot_input['source_detection']['isolate_sources_fwhm_sep'],
                                                   init_iso_scale = autophot_input['source_detection']['init_iso_scale'],
                                                   pix_bound = autophot_input['source_detection']['pix_bound'],
                                                   sigmaclip_median_sigma = autophot_input['source_detection']['sigmaclip_median_sigma'],
                                                   save_FWHM_plot = autophot_input['source_detection']['save_FWHM_plot'],
                                                   
                                                   image_analysis = autophot_input['source_detection']['image_analysis'],
                                                   use_local_stars_for_FWHM = autophot_input['photometry']['use_local_stars_for_FWHM'],
                                                   prepare_templates = autophot_input['template_subtraction']['prepare_templates'],
                                        
                                                   target_x_pix = None,
                                                   target_y_pix = None,
                                                   local_radius = autophot_input['photometry']['local_radius'],
                                                  
                                                   # mask_sources_XY_R = None,
                                                   remove_sat = autophot_input['source_detection']['remove_sat'],
                                                   use_moffat = autophot_input['fitting']['use_moffat'],
                                                   default_moff_beta = autophot_input['fitting']['default_moff_beta'],
                                                   vary_moff_beta = autophot_input['fitting']['vary_moff_beta'],
                                                   max_fit_fwhm = autophot_input['source_detection']['max_fit_fwhm'],
                                                   fitting_method = autophot_input['fitting']['fitting_method'])
            image_fwhm_err = np.nanstd(df['FWHM'])
            
       
            logging.info('\nFWHM: %.3f +/- %.3f [ pixels ]' % (image_fwhm,image_fwhm_err))

            seeing = pixel2arcsec(image_fwhm,xy_pixel_scales[0])
            logging.info('\nSeeing: %.3f [ arcsec ]' % (seeing))
            
            
            # add fwhm to autophot_input file
            autophot_input['fwhm'] = image_fwhm
            autophot_input['scale'] = scale
            autophot_input['image_params'] = image_params
            
            
            # Set range for which PSF model can move around
            autophot_input['dx'] = image_fwhm
            autophot_input['dy'] = image_fwhm
            
            
            if image_fwhm < autophot_input['photometry']['nyquist_limit'] and autophot_input['photometry']['check_nyquist']:
                logging.warning('\nFWHM [%.1f]<%.1f - Sampling errors - using aperture Photometry' % (image_fwhm,autophot_input['nyquist_limit']))
                autophot_input['photometry']['do_ap_phot']  = True
                do_ap = True
                
                
            if image_fwhm >= 25:
                logging.info('FWHM Error %.3f - check image quality' % round(image_fwhm,3) )

            #  Remove any sources that are too close to target location
            dist_to_target = pix_dist(target_x_pix,df['x_pix'],target_y_pix,df['y_pix'])
            too_close_to_target = dist_to_target > 3*image_fwhm
            df = df[too_close_to_target]
            # =============================================================================
            # Find optimum radius for best SNR
            # =============================================================================

            if autophot_input['photometry']['find_optimum_radius']:

                optimum_ap_size = find_optimum_aperture_size(dataframe = df,
                                                               image = image,
                                                               exp_time = autophot_input['exp_time'],
                                                               fwhm= autophot_input['fwhm'],
                                                               write_dir = autophot_input['write_dir'],
                                                               base = autophot_input['base'],
                                                               ap_size = autophot_input['photometry']['ap_size'],
                                                               inf_ap_size = autophot_input['photometry']['inf_ap_size'],
                                                               r_in_size = autophot_input['photometry']['r_in_size'],
                                                               r_out_size = autophot_input['photometry']['r_out_size'],
                                                               GAIN =  autophot_input['gain'],
                                                               rdnoise =  autophot_input['rdnoise'],
                                                               plot_optimum_radius =  autophot_input['photometry']['plot_optimum_radius'])

                autophot_input['photometry']['ap_size'] = optimum_ap_size
                autophot_input['photometry']['r_in_size'] = optimum_ap_size + 1
                autophot_input['photometry']['r_out_size'] = optimum_ap_size + 2
                autophot_input['photometry']['inf_ap_size']= optimum_ap_size + 1.5
                
            logging.info('Aperture size: %.1f pixels' % (autophot_input['photometry']['ap_size'] * autophot_input['fwhm']))

            # =============================================================================
            # Do Aperture photometry to find the bright sources
            # =============================================================================

            df = do_aperture_photometry(image = image,
                                        dataframe = df,
                                        fwhm= autophot_input['fwhm'],
                                        ap_size = autophot_input['photometry']['ap_size'],
                                        inf_ap_size = autophot_input['photometry']['inf_ap_size'],
                                        r_in_size = autophot_input['photometry']['r_in_size'],
                                        r_out_size = autophot_input['photometry']['r_out_size'])
            
            
            ap_corr_base, ap_corr_base_err = find_aperture_correction(dataframe = df,
                                                                   
                                                                     write_dir = autophot_input['write_dir'],
                                                                     base = autophot_input['base'],
                                                                     ap_corr_plot = autophot_input['photometry']['ap_corr_plot'])

            aperture_area = np.pi * (autophot_input['photometry']['ap_size']*autophot_input['fwhm'])**2

            #==============================================================================
            # Catalog source detecion
            #==============================================================================

            # Search for sources in images that have corrospondong magnityide entry in given catalog
            specified_catalog = call_catalog.search(headinfo,
                                                    target_coords,
                                                    catalog_autophot_input,
                                                    image_filter = autophot_input['image_filter'],
                                                    wdir = autophot_input['wdir'],
                                                    catalog  = autophot_input['catalog']['use_catalog'],
                                                    include_IR_sequence_data = autophot_input['catalog']['include_IR_sequence_data'],
                                                    catalog_custom_fpath = autophot_input['catalog']['catalog_custom_fpath'],
                                                    radius = autophot_input['catalog']['catalog_radius'],
                                                    target_name =  autophot_input['target_name'],
                                                  )
            # TODO: check this function
            if autophot_input['catalog'] == 'custom' and False:

                specified_catalog = call_catalog.rematch_seqeunce_stars(specified_catalog,autophot_input)

            # TODO: fix limt -> limit in input parameters
            WCS_checked = False

            # While loop to see if WCS needs to be redone
            while True:
                # Re-aligns catalog sources with source detection and centroid
                c = call_catalog.match(image, headinfo, target_coords, 
                                                        catalog_keywords = catalog_autophot_input,
                                                        image_filter = autophot_input['image_filter'],
                                                        chosen_catalog = specified_catalog,
                                                        fwhm = image_fwhm,
                                                        local_radius = autophot_input['photometry']['local_radius'],
                                                        target_x_pix = autophot_input['target_x_pix'],
                                                        target_y_pix = autophot_input['target_y_pix'],
                                                        default_dmag = filters_input['default_dmag'],
                                                        
                                                        mask_sources_XY_R =autophot_input['preprocessing']['mask_sources_XY_R'],
                                                        use_moffat = autophot_input['fitting']['use_moffat'],
                                                        default_moff_beta = autophot_input['fitting']['default_moff_beta'],
                                                        vary_moff_beta = autophot_input['fitting']['vary_moff_beta'],
                                                        bkg_level = autophot_input['fitting']['bkg_level'],
                                                        scale = autophot_input['scale'],
                                                        max_catalog_sources = autophot_input['catalog']['max_catalog_sources'],
                                                        sat_lvl = autophot_input['sat_lvl'],
                                                        max_fit_fwhm = autophot_input['source_detection']['max_fit_fwhm'],
                                                        fitting_method = autophot_input['fitting']['fitting_method'],
                                    
                                                        matching_source_FWHM_limit = autophot_input['catalog']['matching_source_FWHM_limt'],
                                                        catalog_matching_limit = autophot_input['catalog']['catalog_matching_limit'],
                                                        include_IR_sequence_data = autophot_input['catalog']['include_IR_sequence_data'],
                                                       
                                                        pix_bound = autophot_input['source_detection']['pix_bound'],
                                                        plot_catalog_nondetections =  autophot_input['catalog']['plot_catalog_nondetections'])

                if len(c) ==0:
                    raise Exception('Could NOT find any catalog sources in field')
                # sigma clip distances - avoid mismatches default sigmoa = 3
                sigma_dist =  sigma_clipped_stats(c['cp_dist'].values.astype(float))

                median_offset_arcsecs = pixel2arcsec(np.nanmedian(list(sigma_dist)),xy_pixel_scales[0])

                logging.info('\nMedian offset: %.1f [ pixels ] / %.1f [ arcsec ]'% (np.nanmedian(list(sigma_dist)),median_offset_arcsecs))

                # TODO: update this to arcseconds not fwhm
                if np.nanmedian(list(sigma_dist)) <= autophot_input['wcs']['offset_param']:
                    # average offset distance within limit - conintue
                    break
                if not autophot_input['wcs']['allow_recheck']:
                    # allow WCS to be rechecked - continue
                    break
                if 'UPWCS' in headinfo or WCS_checked:
                    # if the WCS has already not been prefromed
                    logging.info('Position offset detected  [%d pixels] but UPWCS found - skipping astrometry' % np.nanmedian(list(sigma_dist)))
                    break
                
                logging.info('Inconsistent Matching - Removing and rechecking WCS')
                # remove wcs values and update -  will delete keys from header file
                fit_open = fits.open(fpath,ignore_missing_end = True)
                fit_header_wcs_clean = removewcs(fit_open,delete_keys = True)
                fit_open.close()

                fit_header_wcs_clean.writeto(fpath,
                                             overwrite = True,
                                             output_verify = 'silentfix+ignore')

                # Run astromety - return filepath of wcs file
                astro_check = AstrometryNetLOCAL(fpath,autophot_input = autophot_input)
                new_wcs  = fits.open(astro_check,ignore_missing_end = True)
                fit_open = updatewcs(fit_open,new_wcs)
                new_wcs.close()
                fit_open.writeto(fpath , overwrite = True,output_verify = 'silentfix+ignore')
                fit_open = fits.open(fpath,ignore_missing_end = True)
                headinfo = getheader(fpath)
                headinfo['UPWCS'] = ('T', 'CROSS CHECKED WITH ASTROMETRY.NET')
                fit_open.writeto(fpath , overwrite = True,output_verify = 'silentfix+ignore')
                fit_open.close()

                WCS_checked = True
            #==============================================================================
            # Isolate catalog sources
            #=============================================================================
            c_temp_dict = {}
            dist_list = []
            # Go through each matched catalog source and get its distance to every other source
            # TODO: remove this and replace with index
            for i in c.index:
                try:
                    dist = np.sqrt((c.x_pix[i]-np.array(c.x_pix))**2 +( c.y_pix[i]-np.array(c.y_pix))**2)
                    dist = dist[np.where(dist!=0)]
                    # add minimum - used to find isolated sources
                    dist_list.append(np.nanmin(dist))
                except:
                    dist_list.append(np.nan)
            # =============================================================================
            # Perform Photometry
            # =============================================================================

            if autophot_input['photometry']['do_ap_phot']:

                do_ap = True
                
            else:
                
                do_ap = False

            # First try to build PSF model in case it is needed later
            try:

                # user defined list of coordinates in ra/dec ( in degrees to use for PSF)
                if autophot_input['psf']['use_PSF_starlist']:
                    try:
                        if not os.path.isfile(autophot_input['psf']['PSF_starlist']):
                            raise Exception(' Starlist selected but cannot find PSF starlist')
                        df_PSF = pd.read_csv(autophot_input['psf']['PSF_starlist'],names=['ra', 'dec'],sep = ' ')
                        x_pix,y_pix = w1.all_world2pix(df_PSF.ra.values,df_PSF.dec.values,1)
                        df_PSF['x_pix']= x_pix
                        df_PSF['y_pix'] = y_pix

                    except:

                        autophot_input['psf']['use_PSF_starlist'] = False
                        df_PSF = df
                else:

                    df_PSF = df
                if autophot_input['photometry']['use_local_stars_for_PSF'] and not autophot_input['photometry']['use_PSF_starlist']:
                    dist = pix_dist(target_x_pix,df_PSF['x_pix'],target_y_pix,df_PSF['y_pix'])
                    df_PSF['dist'] = dist
                    
                r_table, fwhm_fit, psf_MODEL_sources = psf.build_r_table(base_image = image,
                                                                            selected_sources = df_PSF,
                                                                            fwhm = autophot_input['fwhm'],
                                                                            exp_time = autophot_input['exp_time'],
                                                                            image_params = autophot_input['image_params'],
                                                                            fpath = autophot_input['fpath'],
                                                                            GAIN = autophot_input['gain'],
                                                                            rdnoise = autophot_input['rdnoise'],
                                                                            use_moffat = autophot_input['fitting']['use_moffat'],
                                                                           
                                                                            fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                                            regrid_size = autophot_input['psf']['regriding_size'],
                                                                            use_PSF_starlist = autophot_input['psf']['use_PSF_starlist'],
                                                                            use_local_stars_for_PSF = autophot_input['photometry']['use_local_stars_for_PSF'],
                                                                            prepare_templates = autophot_input['template_subtraction']['prepare_templates'],
                                                                            scale = autophot_input['scale'],
                                                                            ap_size = autophot_input['photometry']['ap_size'],
                                                                            r_in_size = autophot_input['photometry']['r_in_size'],
                                                                            r_out_size = autophot_input['photometry']['r_out_size'],
                                                                            local_radius = autophot_input['photometry']['local_radius'],
                                                                            bkg_level = autophot_input['fitting']['bkg_level'],
                                                                            psf_source_no = autophot_input['psf']['psf_source_no'],
                                                                            min_psf_source_no = autophot_input['psf']['min_psf_source_no'],
                                                                            construction_SNR = autophot_input['psf']['construction_SNR'],
                                                                            remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                                            remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                                            remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                                            remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                                                            
                                                                            
                                                                            fitting_method = autophot_input['fitting']['fitting_method'],
                                                                            save_PSF_stars = autophot_input['psf']['save_PSF_stars'],
                                                                            # plot_PSF_model_residuals = autophot_input['psf']['plot_PSF_model_residuals'],
                                                                            save_PSF_models_fits = autophot_input['psf']['save_PSF_models_fits'])


                # Need to check if PSF model if build, inital assume it is not
                PSF_available  = False

                # if it fails, select aperture photometry and exit this attempt
                if fwhm_fit == None:
                    do_ap=True
                    PSF_available = False
                    logging.error('\nCould not build Residual table - using aperture photometry')
                elif np.any(r_table ==  None):
                    do_ap=True
                    logging.error('PSF fitting unavialable')
                else:
                    PSF_available = True
                # Get data of sources used for psf                
                if PSF_available:

                    psf_stats, unity_PSF_counts = psf.do(df = psf_MODEL_sources,
                                                        residual_image = r_table,
                                                        ap_size = autophot_input['photometry']['ap_size'],
                                                        fwhm = autophot_input['fwhm'],
                                                        use_moffat = autophot_input['fitting']['use_moffat'],
                                                        image_params = autophot_input['image_params'])
                    autophot_input['unity_PSF_counts'] = unity_PSF_counts
 
                if not do_ap and PSF_available:

                    logging.info('Using PSF Photometry on Sequence Stars' )
                    '''
                    Get approximate magnitude of psf to check if target is better than psf
                    very dodgey if target is not specific and target is set to center of image

                    '''
                    psf_flux = psf_stats['psf_counts']/autophot_input['exp_time']
                    approx_psf_mag = float(np.nanmin(calc_mag(psf_flux,autophot_input['gain'],0)))

                    logging.info('Approx PSF mag %.3f mag' % approx_psf_mag)

                    c_psf = psf.fit(image = image,
                                    sources = c,
                                    residual_table = r_table,
                                    fwhm = autophot_input['fwhm'],
                                    fpath = autophot_input['fpath'],
                                    fitting_radius = autophot_input['fitting']['fitting_radius'],
                                    regriding_size = autophot_input['psf']['regriding_size'],
                                    scale = autophot_input['scale'],
                                    bkg_level = autophot_input['fitting']['bkg_level'],
                                    remove_sat = autophot_input['source_detection']['remove_sat'],
                                    sat_lvl = autophot_input['sat_lvl'],
                                    use_moffat = autophot_input['fitting']['use_moffat'],
                                    image_params = autophot_input['image_params'],
                                    fitting_method = autophot_input['fitting']['fitting_method'],
                                    # return_psf_model = autophot_input['psf']['return_psf_model'],
                                    # save_plot = autophot_input['psf']['save_plot'],
                                    # show_plot = autophot_input['show_plot'],
                                    # remove_background_val = autophot_input['remove_background_val'],
                                    # hold_pos = autophot_input['hold_pos'],
                                    return_fwhm = True,
                                    return_subtraction_image = autophot_input['psf']['return_subtraction_image'],
                                    no_print = False,
                                    # return_closeup = autophot_input['return_closeup'],
                                    remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                    remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                    remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                    remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                    plot_PSF_residuals = autophot_input['psf']['plot_PSF_residuals'])
                   


                    c_psf = psf.do(df = c_psf,
                                residual_image = r_table,
                                ap_size = autophot_input['photometry']['ap_size'],
                                fwhm = autophot_input['fwhm'],
                                unity_PSF_counts = autophot_input['unity_PSF_counts'],

                                use_moffat = autophot_input['fitting']['use_moffat'],
                                image_params = autophot_input['image_params'])
                    

                    # background flux
                    bkg_flux = c_psf['bkg']/exp_time

                    # source flux
                    source_flux = c_psf['psf_counts']/exp_time
                    
                    # fitting error
                    source_flux_err = c_psf['psf_counts_err']/exp_time
                    
                  
                    SNR_val = SNR(flux_star = source_flux,
                                  flux_sky = bkg_flux,
                                  exp_t = autophot_input['exp_time'],
                                  radius = autophot_input['photometry']['ap_size']*autophot_input['fwhm'],
                                  G  = autophot_input['gain'],
                                  RN =  autophot_input['rdnoise'],
                                  DC = 0 )
                    chi2 = c_psf['chi2']
                    redchi2 = c_psf['redchi2']

            except Exception as e:
                logging.exception(e)
                # if something goes wrong- do aperture  photometry instead
                do_ap = True
                pass
            # Perform aperture photoometry if pre-selected or psf fitting wasn't viable
            if autophot_input['photometry']['do_ap_phot'] or do_ap == True:
                
                logging.info('Using Aperture Photometry on sequence Stars ' )
                # list of tuples of pix coordinates of sources
                positions  = list(zip(np.array(c.x_pix),np.array(c.y_pix)))
   
                ap , ap_error, max_pixels , bkg, bkg_std = measure_aperture_photometry(positions,
                                                                            image,
                                                                            ap_size =  autophot_input['photometry']['ap_size']    * image_fwhm,
                                                                            r_in =    autophot_input['photometry']['r_in_size']  * image_fwhm,
                                                                            r_out =   autophot_input['photometry']['r_out_size'] * image_fwhm)
                # Background flux from annulus
                bkg_flux = bkg/exp_time
                # Source flux
                source_flux = ap/exp_time
                
                source_flux_err = ap_error/exp_time

                SNR_val = SNR(flux_star = source_flux ,
                              flux_sky = bkg_flux,
                               exp_t = autophot_input['exp_time'],
                               radius = autophot_input['photometry']['ap_size']*autophot_input['fwhm'] ,
                               G  = autophot_input['gain'],
                               RN =  autophot_input['rdnoise'],
                               DC = 0 )
                
                
                
            # Adding everything to temporary dataframe - no idea why though
            c_temp_dict['dist'] = dist_list
            c_temp_dict['SNR'] = SNR_val
            c_temp_dict['flux_bkg'] = bkg_flux
            c_temp_dict['flux_star'] = source_flux
            c_temp_dict['flux_star_err']= source_flux_err
            
           
            try:
                c_temp_dict['chi2'] = chi2
                c_temp_dict['redchi2'] = redchi2
            except:
                 logging.info('No PSF sources fitted')
                 autophot_input['catalog']['remove_catalog_poorfits'] = False
                 
            # add to exisitng dataframe [c]
            c_add = pd.DataFrame.from_dict(c_temp_dict)
            c_add = c_add.set_index(c.index)
            c = pd.concat([c,c_add],axis = 1,sort = False)
            
            # drop any poorly fit sources:
            if autophot_input['catalog']['remove_catalog_poorfits']:
                redchi2_mean = np.nanmean(c['redchi2'])
                redchi2_std = np.nanstd(c['redchi2'])
                filtered_data = sigma_clip(c['redchi2'].values,
                                           sigma_lower=None,sigma_upper = 3,
                                           maxiters=None,
                                           cenfunc='mean',
                                           masked=True,
                                           copy=False)
                c = c[~filtered_data.mask]
                c = c.drop(c[c['redchi2'] > redchi2_mean + 3 * redchi2_std].index)
                
            # drop if the counts are negative - account for mismatched source or very faint source
            c = c.drop(c[c['flux_star'] <= 0.0].index)

            c = c.drop(c[np.isnan(c['flux_star'])].index)
            
            # Remove high error catalog sources
            c = c.drop(c[c['cat_'+str(use_filter)+'_err'] >1].index)
            
            if autophot_input['photometry']['use_local_stars']:
                c = c[c['dist2target'] <= local_radius]

             # =============================================================================
            # Aperture correction
            # =============================================================================
            if not do_ap:

                ap_corr = 0
                ap_corr_err = 0
                
            else:
                ap_corr = ap_corr_base
                ap_corr_err = ap_corr_base_err


            c['inst_'+str(use_filter)] = calc_mag(c['flux_star'],autophot_input['gain'],0) + ap_corr
            
            
            # Error in instrumental Magnitude
            c_SNR_err = SNR_err(c.SNR.values)

            c['inst_'+str(use_filter)+'_err'] = np.sqrt(c_SNR_err**2 + ap_corr_err**2)

            # =============================================================================
            # Find Zeropoint
            # =============================================================================

            zp_measurement, c = get_zeropoint(c,image=image,headinfo=headinfo,
                                          fpath = autophot_input['fpath'],
                                          use_filter = use_filter,
                                          
                                          matching_source_SNR_limit = autophot_input['zeropoint']['matching_source_SNR_limit'],
                                          GAIN = autophot_input['gain'],
                                          fwhm = autophot_input['fwhm'],
                                          zp_sigma = autophot_input['zeropoint']['zp_sigma'],
                                          zp_use_fitted = autophot_input['zeropoint']['zp_use_fitted'],
                                          zp_use_mean = autophot_input['zeropoint']['zp_use_mean'],
                                          zp_use_max_bin = autophot_input['zeropoint']['zp_use_max_bin'],
                                          zp_use_median = autophot_input['zeropoint']['zp_use_median'],
                                          zp_use_WA = autophot_input['zeropoint']['zp_use_WA'],
                                           plot_ZP_image_analysis = autophot_input['zeropoint']['plot_ZP_image_analysis'],
                                          # plot_ZP_image_analysis = False,
                                          plot_ZP_vs_SNR = autophot_input['zeropoint']['plot_ZP_vs_SNR'])
  

            # =============================================================================
            # Plot of image with sources used and target
            # =============================================================================

            if autophot_input['plot_source_selection']:
                plt.ioff()

                fig_source_check = plt.figure(figsize = set_size(500,aspect = 1))

                vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image)

                ax = fig_source_check.add_subplot(111)

                ax.imshow(image,
                          vmin = vmin,
                          vmax = vmax,
                           # interpolation = 'None',
                          origin = 'lower',
                          aspect = 'equal',
                          cmap = 'Greys')

                ax.scatter(c.x_pix,c.y_pix,
                           marker = '+',
                           s = 25,
                           color = 'red',
                           label = 'Recentering [%d]' % len(c.x_pix),
                           zorder = 2,
                           linewidths=0.1)

                ax.scatter(c.x_pix_cat,c.y_pix_cat,
                           marker = 's',
                           s = 25,
                           color = 'green',
                           facecolor = 'None',
                           label = 'Sequence Stars [%d]' % len(c.x_pix_cat),
                           linewidths=0.1,
                           zorder = 3)

                ax.scatter([autophot_input['target_x_pix']],[autophot_input['target_y_pix']],
                            marker = 'H',
                            facecolor = 'None',
                            edgecolor = 'gold',
                            s = 25,
                            linewidths=0.1,
                            label = 'Target: %s' %  tname
                            )

                if not do_ap:
                    ax.scatter(psf_MODEL_sources.x_pix,psf_MODEL_sources.y_pix,
                               marker = 'o',
                               s = 25,
                               color = 'blue',
                               facecolor = 'None',
                               label = 'PSF Sources [%d]' % len(psf_MODEL_sources),
                               linewidths=0.1,
                               zorder = 3)

                if autophot_input['photometry']['use_local_stars']:

                    local_radius_circle = plt.Circle( ( target_x_pix, target_y_pix ), autophot_input['local_radius'],
                                                     color = 'red',
                                                     ls = '--',
                                                     lw = 0.5,
                                                     label = 'Local Radius [%d px]' % autophot_input['local_radius'],
                                                     fill=False)
                    ax.add_patch( local_radius_circle)

                if autophot_input['preprocessing']['mask_sources']:
                    for X_mask,Y_mask,R_mask in autophot_input['preprocessing']['mask_sources_XY_R']:

                        ax.scatter(X_mask, Y_mask ,
                                   color = 'red',
                                   marker = 'X',
                                   label = 'Masked Galaxy')

                ax.set_xlim(0,image.shape[1])
                ax.set_ylim(0,image.shape[0])

                ax.set_xlabel('X Pixel')
                ax.set_ylabel('Y Pixel')

                lines_labels = [ax.get_legend_handles_labels() for ax in fig_source_check.axes]
                handles,labels = [sum(i, []) for i in zip(*lines_labels)]

                by_label = dict(zip(labels, handles))

                fig_source_check.legend(by_label.values(), by_label.keys(),
                                        bbox_to_anchor=(0.5, 0.89),
                                        loc='lower center',
                                        ncol = 2,
                                        frameon=False)

            
                fig_source_check.savefig(autophot_input['write_dir'] + '/' +'source_check_'+str(autophot_input['base'].split('.')[0])+'.pdf',
                                         format = 'pdf',bbox_inches='tight')

                plt.close(fig_source_check)
           
            if autophot_input['catalog']['plot_catalog_nondetections']:

                c_tmp = c
                # non_detections = c[c.SNR<3]['cat_'+use_filter].values

                catalog_magnitudes = np.arange(c_tmp['cat_'+use_filter].min(),c_tmp['cat_'+use_filter].max(),0.15)

                detection_percents = []

                for i in catalog_magnitudes:
                    c_i = c_tmp[abs( c_tmp['cat_'+use_filter].values - i)<0.1]
                    if len(c_i)<=1:
                        p = 1
                    else:
                        p = np.sum(c_i.SNR.values>=autophot_input['limiting_magnitude']['lim_SNR']) / len(c_i) 
                    # print(i,p)
                    detection_percents.append([i,100 * p])
                plt.ioff()
                fig = plt.figure(figsize=set_size(250,1.5))

                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212,sharex = ax1)

                ax1.plot([i[0] for i in detection_percents],
                         [i[1] for i in detection_percents],
                         marker = 'o',
                         ls = '-',
                         color = 'black')

                ax1.fill_between([i[0] for i in detection_percents], 
                                 [i[1] for i in detection_percents], 100,
                                 color = 'green',
                                 alpha = 0.5,
                                 hatch = '////',
                                 edgecolor = 'none',
                                 label = 'Non detections')
                ax1.fill_between([i[0] for i in detection_percents], [i[1] for i in detection_percents], 0,
                                 color = 'red',

                                 alpha = 0.5,
                                 hatch = '\\\\',
                                 edgecolor = 'none',
                                 label = 'detections')

                ax2.scatter(c['cat_'+use_filter].values,
                            c['SNR'].values,
                            marker = '.',
                            # ls = '-',
                            color = 'blue')
                ax2.axhline(3,label = 'SNR=3',ls = ':',color = 'black')
                ax2.set_ylim(0.1,None)
                # ax2.plot( xx, yy)
                ax2.set_yscale('log')

                plt.setp( ax1.get_xticklabels(), visible=False)
                ax1.set_ylabel('Sources Detected [ % ]')
                ax1.xaxis.set_major_locator(MultipleLocator(1))
                ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
                ax1.legend(loc = 'best',frameon = True,facecolor="white",edgecolor = 'none')
                ax2.legend(loc = 'best',frameon = True,facecolor="white",edgecolor = 'none')

                ax2.set_xlabel('Catalog Magnitude [ mag ]')
                ax2.set_ylabel('SNR')

                figname = os.path.join(autophot_input['write_dir'],'catalog_nondetections_'+autophot_input['base']+'.pdf')

                fig.savefig(figname,
                            format = 'pdf',
                            bbox_inches='tight'
                            )

                plt.close(fig)
        
            # =============================================================================
            # Limiting Magnitude
            # =============================================================================
        
            
            # Create copy of image to avoid anything being written to original image
            image_copy = image.copy()
            # Bin size in magnitudes
            b_size = 0.25
            lim_err =  SNR_err(autophot_input['limiting_magnitude']['lim_SNR'])
            # Sort values by filter
            c_mag_lim = c.sort_values(by = [str('cat_'+use_filter)])
            # x values: autophot magnitudes
            x = c_mag_lim['cat_'+str(use_filter)].values
            x_err = c_mag_lim['cat_'+str(use_filter)+'_err'].values
            # y values: absolute differnece between catalog magnitude and autophots
            y =  c_mag_lim[str(use_filter)].values - c_mag_lim[str('cat_'+use_filter)].values
            y_err =  np.sqrt(c_mag_lim[str(use_filter)+'_err'].values**2 + c_mag_lim[str('cat_'+use_filter)+'_err'].values**2)
            # remove nans
            idx = (np.isnan(x)) |  (np.isnan(y))
            x = x[~idx]
            y = y[~idx]
            y_err = y_err[~idx]
            x_err = x_err[~idx]
            if len(x)  <= 1 :
                logging.warning('Magnitude diagram not found')
                catalog_mag_limit = np.nan
            else:

                plt.ioff()
                fig_magnitude = plt.figure(figsize = set_size(500,aspect = 0.5))

                grid = GridSpec(1, 2 ,
                                wspace=0.05, hspace=0,
                                width_ratios = [1,0.25])
                ax1 = fig_magnitude.add_subplot(grid[0 , 0])
                ax2 = fig_magnitude.add_subplot(grid[0, 1],sharey = ax1)
                s, edges, _ = binned_statistic(x,y,
                                                statistic='median',
                                                bins=np.linspace(np.nanmin(x),np.nanmax(x),int((np.nanmax(x)-np.nanmin(x))/b_size)))
                bin_centers = np.array(edges[:-1]+np.diff(edges)/2)
                ax1.hlines(s,edges[:-1],edges[1:], color="black")
                ax1.scatter(bin_centers, s,
                            c="green",
                            marker = 's',
                            label = 'Median Bins',
                            zorder = 10)
                markers, caps, bars = ax1.errorbar(x,y,
                                                    xerr = x_err,
                                                    yerr = y_err,
                                                    color = 'red',
                                                    ecolor = 'black',
                                                    capsize = 0.5,
                                                    marker = 'o',
                                                    ls = '',
                                                    # label = ,
                                                    zorder = 90)
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
                ax1.axhline(lim_err,
                            label = 'SNR error (%d)' % autophot_input['limiting_magnitude']['lim_SNR'],
                            linestyle = '--',
                            color = 'black')
                ax2.axhline(lim_err,
                            linestyle = '--',
                            color = 'black')
                ax1.axhline(-1*lim_err,linestyle = '--',color = 'black')
                ax2.axhline(-1*lim_err,linestyle = '--',color = 'black')
                ax1.set_ylim(-0.5,0.5)
                ax1.set_ylabel(r'$M_%s - M_{%s,cat}$ [ mag ]' % (use_filter,use_filter))
                ax1.set_xlabel(r'$M_{%s,cat}$ [ mag ]' % use_filter)

                catalog_mag_limit = np.nan

                # Find magnitude where all over where all proceeding magnitude bins are greater than the error SNR cutoff
                for i in range(len(s)):
                    t = s[i]
                    if abs(t)>lim_err:
                        for j in range(i+1,len(s)):
                            if abs(s[j]) < lim_err:
                                break
                        else:

                            catalog_mag_limit = bin_centers[i]

                            text = 'Limit = %.1f [ mag ]' % catalog_mag_limit

                            ax1.annotate(text, xy=(catalog_mag_limit,0),
                                        xytext = (catalog_mag_limit,-0.15),
                                        va = 'center',
                                        ha = 'center',
                                        color = 'red',
                                        xycoords = ax1.get_xaxis_transform(),  
                                        arrowprops=dict(arrowstyle="->", color='red'),
                                        annotation_clip=False)
                            break

                # Set magnitude limit with inall i'th value here
                # mag_limit = bin_centers[i]
                n, bins, patches = ax2.hist(y,
                                            # weights = weights,
                                            bins = 'auto',
                                            facecolor = 'green',
                                            label = 'Zeropoint Distribution',
                                            density = True,
                                            orientation = 'horizontal')
                ax2.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
                ax2.set_xlabel('Probability Density')
                ax2.yaxis.tick_right()

                ax1.legend(fancybox=True,
                            ncol = 4,
                            bbox_to_anchor=(0, 1.01, 1, 0),
                            loc = 'lower center',
                            frameon=False
                            )
                ax1.axhline(0,alpha = 0.5,
                            color = 'black',
                            ls = ':',zorder = 0 )
                ax1.axhspan(lim_err, 1, alpha=0.3, color='gray')
                ax1.axhspan(-lim_err,-1, alpha=0.3, color='gray')
                ax1.text(0.5,lim_err+0.075,
                    'Over Luminous',
                    va = 'bottom',
                    ha = 'center',
                    transform=ax1.get_yaxis_transform(),
                    color = 'black',
                    rotation = 0)
                ax1.text(0.5,-lim_err-0.075,
                    'Under Luminous',
                    va = 'top',
                    ha = 'center',
                    transform=ax1.get_yaxis_transform(),
                    color = 'black',
                    rotation = 0)

                ax1.xaxis.set_major_locator(MultipleLocator(1))
                ax1.xaxis.set_minor_locator(MultipleLocator(0.25))

                fig_magnitude.savefig(os.path.join(cur_dir,'zeropoint_accuracy_'+str(base.split('.')[0])+'.pdf'),
                                        bbox_inches='tight')
                plt.close(fig_magnitude)

            if not np.isnan(catalog_mag_limit):
                logging.info('Approx. catalog limiting magnitude: %s [ mag ]' % str(round(catalog_mag_limit,3)))

            elif do_ap or autophot_input['photometry']['do_ap_phot']:

                # if ap_phot selected - sets magnitude limit to nan <- will fix

                catalog_mag_limit = np.nan
                
            image_copy = image.copy()

            # if np.isnan(catalog_mag_limit):
            #     mag_limit = 20
                
            # =============================================================================
            # Get Template
            # =============================================================================
            # Initally assume subtraction is not ready
            subtraction_ready = False
            template_found = False

            if autophot_input['template_subtraction']['do_subtraction']:
                try:
                    # Pan_starrs template images use some old WCS keywrods, astropy can handle them just for cleaniness
                    warnings.filterwarnings('ignore')
                    ra = target_coords.ra.degree
                    dec = target_coords.dec.degree

                    # Get pixel scale - output in deg
                    image_scale = np.max(wcs.utils.proj_plane_pixel_scales(w1))

                    # size of image in arcseonds
                    size = round(image_scale * 3600 * np.nanmax(image.shape))

                    # avoid case-sensitive nature of python
                    if use_filter in ['g','r','i','z','u']:
                        use_filter_template = use_filter+'p'

                    else:
                        use_filter_template = use_filter

                    expected_template_folder = os.path.join(autophot_input['fits_dir'] , 'templates')
                    expected_filter_template_folder  = os.path.join(expected_template_folder, use_filter_template + '_template')

                    if autophot_input['template_subtraction']['use_user_template']:


                        logging.info('Looking for User template in %s' % expected_template_folder)
                        if not os.path.exists(expected_template_folder):
                            logging.info('Templates folder not found - check filepath is correct')

                        else:
                            
                            if not os.path.exists(expected_filter_template_folder ):
                                logging.info('Cannot find template for filter: %s' % use_filter_template.replace('p',''))
                            else:

                                list_dir = os.listdir(expected_filter_template_folder)
                                fits_list_dir = [i for i in list_dir if i.split('.')[-1] in ['fits','fts','fit']]

                                if len(fits_list_dir) >1:
                                    fits_list_dir = [i for i in fits_list_dir if 'PSF_model_' not in i and i.endswith(fname_ext)]
#                                
                                fpath_template = os.path.join(expected_filter_template_folder,  fits_list_dir[0])
                                template_found = True
                                logging.info('Template filepath: %s ' % fpath_template)

                    if autophot_input['template_subtraction']['get_template'] and not template_found:
                        logging.info('Searching for template ...')

                        if autophot_input['catalog'] == '2mass':
                            try:
                                
                                # https://astroquery.readthedocs.io/en/latest/skyview/skyview.html
                                hdu  = SkyView.get_images(target_coords,survey = ['2MASS-'+use_filter_template.upper()],coordinates = 'ICRS',radius = size * u.arcsec)
                                fits.writeto(fpath.replace(fname_ext,'_template')+'no_rot'+fname_ext,
                                 hdu[0][0].data,
                                  headinfo, overwrite=True,
                                  output_verify = 'silentfix+ignore')
                            except Exception as e:
                                    logging.exception(e)

                        # Template retrival from panstarrs
                        if autophot_input['catalog']['use_catalog'] == 'pan_starrs' or autophot_input['catalog']['use_catalog'] == 'skymapper':
                            # arcsec per pxiel for PS1
                            pan_starrs_pscale = 0.25 

                            # Check if the file has been downloaded before
                            
                            # use_filter_template + '_template/'+'template_'+use_filter+'_retrieved'+fname_ext
                            expected_filter_template_file = os.path.join(expected_filter_template_folder,use_filter+'_template_retrieved'+fname_ext)

                            if os.path.isfile(expected_filter_template_file):
                                logging.info('Found previously retrieved template: %s' % expected_filter_template_file)
                                template_found = True
                                fpath_template = expected_filter_template_file

                            else:
                                
                                logging.info('Searching for template on PanSTARRS')
                                fitsurl = get_pstars(float(ra), float(dec), size=int(size/pan_starrs_pscale), filters=use_filter)
                                with fits.open(fitsurl[0],ignore_missing_end = True,lazy_load_hdus = True) as hdu:
                                    try:
                                        hdu.verify('silentfix+ignore')
                                        headinfo_template = hdu[0].header
                                        template_found  = True
                                        # save templates into original folder under the name template
                                        pathlib.Path(expected_template_folder).mkdir(parents = True, exist_ok=True)
                                        fits.writeto(expected_filter_template_file,
                                                     hdu[0].data,
                                                     headinfo_template,
                                                     overwrite=True,
                                                     output_verify = 'silentfix+ignore')
                                        if os.path.isfile(expected_filter_template_file):
                                            logging.info('Retrieved template saved as: %s' % expected_filter_template_file)
                                            fpath_template = expected_filter_template_file
                                            template_found = True
                                    except Exception as e:
                                        logging.exception(e)


                    if not template_found:
                        logging.info('Template not found - cannot perform image subtraction')
                    else:
                        with fits.open(fpath_template,ignore_missing_end = True,lazy_load_hdus = True) as hdu:
                            headinfo_template = hdu[0].header
                            try:
                                if autophot_input['template_subtraction']['use_astroalign']:
                                    try:
                                        
                                        logging.info('Aligning via Astro Align')
                                        
                                        aligned_template, footprint = aa.register(hdu[0].data.astype(float),
                                                                                image.astype(float))
                                        aligned_template[footprint] = 0

                                    except Exception as e:

                                        logging.exception(e)
                                        autophot_input['template_subtraction']['use_astroalign'] = False


                                        
                                if not autophot_input['template_subtraction']['use_astroalign']:
                                    try:
                                        logging.info('Aligning via WCS with reproject_interp')
                                        
                                        aligned_template, footprint = reproject_interp(hdu[0], headinfo, order = 1)
                                        aligned_template[~footprint.astype(bool)] = 0

                                    except Exception as e:
                                        logging.info('Could not align images: %s' % e)
                                        # TODO: make this not crash everything
                                        raise Exception

                                # cutout template around transient and match image size

                                aligned_template_no_zeroes,good_templates_slices = trim_zeros_slices(aligned_template)

                                # Lets see where the target location is in relation to the template cutout
                                ny, nx = aligned_template.shape
                                x = np.arange(nx) # x an y so they are distance from center, assuming array is "nx" long (as opposed to 1. which is the other common choice)
                                y = np.arange(ny) 
                                Y, X = np.meshgrid(x, y)
                                
                                distance_grid = pix_dist(X,target_x_pix,Y,target_y_pix)
                                distance_grid[good_templates_slices] = 0

                                distance_to_zero = distance_grid[distance_grid>0]

                                if len(distance_to_zero) ==0:
                                    distance_to_zero  = [np.nan]

                                if np.isnan(np.nanmin(distance_to_zero)):

                                    logging.info('Template larger than image, cropping')

                                    crop_size_y = np.floor(aligned_template.shape[0])
                                    crop_size_x = np.floor(aligned_template.shape[1])

                                else:

                                    logging.info('Template smaller than image, cropping to exlcude zeros')
                                    crop_size_x = 2*abs(np.nanmin(distance_to_zero))
                                    crop_size_y = 2*abs(np.nanmin(distance_to_zero))

                                aligned_template = Cutout2D(aligned_template,
                                                            (np.floor(target_x_pix),np.floor(target_y_pix)),
                                                            (int(crop_size_y),int(crop_size_x)),
                                                            # mode = 'strict'
                                                            )

                                image_template_size =  Cutout2D(image,
                                                                (np.floor(target_x_pix),np.floor(target_y_pix)),
                                                                (int(crop_size_y),int(crop_size_x)),
                                                                wcs=w1,
                                                                # mode = 'strict'
                                                                )

                                logging.info('Trimmed template shape:(%d %d)' %  (aligned_template.data.shape[0],aligned_template.data.shape[1]))
                                logging.info('Trimmed image shape:(%d %d)' %  (image_template_size.data.shape[0],image_template_size.data.shape[1]))

                                if image_template_size.data.shape[0]!=aligned_template.shape[0] or image_template_size.data.shape[1]!=aligned_template.shape[1]:
                                    sys.exit('Templates and image not aligned correctly')
                                    # raise Exception()

                                # Update to WCS info of closeup image that has been trimmed to match template
                                
                                w1 = WCS(image_template_size.wcs.to_header())

                                target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree,
                                                                              target_coords.dec.degree,
                                                                              1)
                                autophot_input['target_x_pix'] = target_x_pix
                                autophot_input['target_y_pix'] = target_y_pix

                                fpath_template = fpath.replace(fname_ext,'_template')+fname_ext
                                fpath = fpath.replace(fname_ext,'_image_cutout')+fname_ext
                                
                                # Write aligned image with cutout to file
                                fits.writeto(fpath_template,
                                             aligned_template.data,
                                             headinfo_template,
                                             overwrite=True,
                                             output_verify = 'silentfix+ignore')

                                # Write aligned template to file
                                fits.writeto(fpath,
                                             image_template_size.data,
                                             image_template_size.wcs.to_header(),
                                             overwrite=True,
                                             output_verify = 'silentfix+ignore')

                                subtraction_ready = True

                                # This is for *** - 0 = Good pixel, 1 = mask pixel 
                                footprint = abs(1-footprint)
                            except Exception as e:
                                logging.exception(e)
                    warnings.filterwarnings("default")
                    if os.path.isfile(fpath.replace(fname_ext,'_template')+fname_ext):

                        logging.info('Template saved as: %s' %os.path.basename(fpath.replace(fname_ext,'_template')))
                except Exception as e:
                    logging.error('Error with Template aquisiton: %s ' % e)
                    logging.exception(e)
            # Check for template image
            # if autophot_input['get_template']:
            #     if not os.path.isfile(fpath.replace(fname_ext,'_template')+fname_ext):
            #         logging.info('Template file NOT found')
            #     else:
            #         fpath_template = fpath.replace(fname_ext,'_template')+fname_ext
            #         subtraction_ready = True
            # =============================================================================
            # Image subtraction using HOTPANTS
            # =============================================================================
            autophot_input['subtraction_ready'] = subtraction_ready
            if autophot_input['template_subtraction']['do_subtraction'] and subtraction_ready:

                hdu = fits.PrimaryHDU(footprint.astype(int))
                hdul = fits.HDUList([hdu])
                footprint_loc = os.path.join(autophot_input['write_dir'],'align_footprint_'+autophot_input['base']+fname_ext)

                hdul.writeto(footprint_loc,
                      overwrite=True,
                      output_verify = 'silentfix+ignore')

                #  This is where the template files are found
                autophot_input['template_dir'] = os.path.join(autophot_input['fits_dir'],'templates/'+ use_filter_template + '_template')

                fpath_sub = subtract(file = fpath,
                                     template = fpath_template,
                                     image_fwhm = image_fwhm,
                                     footprint = footprint,
                                     use_zogy = autophot_input['template_subtraction']['use_zogy'],
                                     hotpants_exe_loc = autophot_input['template_subtraction']['hotpants_exe_loc'],
                                     hotpants_timeout = autophot_input['template_subtraction']['hotpants_timeout'],
                                     template_dir = autophot_input['template_dir'],
                                     psf = PSF_MODEL,
                                     mask_border = False,
                                     pix_bound = autophot_input['source_detection']['pix_bound'],
                                     remove_sat = autophot_input['source_detection']['remove_sat'],
                                     zogy_use_pixel = autophot_input['template_subtraction']['zogy_use_pixel'])

                
            if autophot_input['template_subtraction']['do_ap_on_sub'] and subtraction_ready:
                    do_ap = True
                    autophot_input['do_ap_phot'] = True
                    logging.info('\nPerforming aperture photometry on subtracted image\nSwitching to local median background fit')

                    autophot_input['fitting']['remove_bkg_surface'] = False
                    autophot_input['fitting']['remove_bkg_local'] =  True
                    autophot_input['fitting']['remove_bkg_poly'] =  False

            # =============================================================================
            # Perform photometry on target
            # =============================================================================
            
            if subtraction_ready:
                image    = getimage(fpath_sub)
                logging.info('Target photometry on subtracted image')
            else:
                logging.info('Target photometry on original image')
            image_copy  = image.copy()
            target_x_pix_TNS, target_y_pix_TNS = w1.all_world2pix(autophot_input['target_ra'],
                                                                  autophot_input['target_dec'],
                                                                  1)
            target_close_up = image_copy[int(target_y_pix_TNS - autophot_input['scale']): int(target_y_pix_TNS + autophot_input['scale']),
                                         int(target_x_pix_TNS - autophot_input['scale']): int(target_x_pix_TNS + autophot_input['scale'])]

            target_close_up_median = np.nanmedian(target_close_up)

            xx,yy = np.meshgrid(np.arange(0,2*autophot_input['scale']),np.arange(0,2*autophot_input['scale']))

            # =============================================================================
            # Subtraction image
            # =============================================================================

            if autophot_input['template_subtraction']['save_subtraction_quicklook'] and subtraction_ready:
                plt.ioff()
                fig_sub = plt.figure(figsize = set_size(250,aspect = 1))
                ax1 = fig_sub.add_subplot(121)

                ax2 = fig_sub.add_subplot(122)
                plt.subplots_adjust(hspace=0.0,wspace=0.3)
                image_sub_tmp = fits.getdata(fpath_sub)

                vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image_sub_tmp)
                ax1.imshow(image_sub_tmp,
                           vmin = vmin,
                           vmax = vmax,
                           origin = 'lower',
                           aspect = 'equal',
                           )
                ax1.scatter([target_x_pix],[target_y_pix],marker = 'D',
                           facecolor = 'None',
                           color = 'GOLD',
                           linewidth = 0.5,
                           s = 25,
                           label =  tname)
                im = ax2.imshow(target_close_up,
                                origin = 'lower',
                                aspect = 'equal',
                                # cmap = 'Greys'
                                )

                ax1.set_title('Template subtracted image')
                ax2.set_title('Transient cutout')

                ax1.set_xlabel('X PIXEL')
                ax1.set_ylabel('Y PIXEL')
                ax2.set_xlabel('X PIXEL')
                ax2.set_ylabel('Y PIXEL')
                ax1.legend(loc = 'best',frameon = False)
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = fig_sub.colorbar(im, cax=cax)
                cb.ax.set_ylabel('Counts', rotation=270,labelpad = 10)
                figfname = 'subtraction_QUICKLOOK_'+os.path.basename(fpath).replace(fname_ext,'') +'.pdf'
                QL_sub_loc = os.path.join(cur_dir,figfname)
                fig_sub.savefig(QL_sub_loc,bbox_inches='tight')
                plt.close(fig_sub)
            # =============================================================================
            # Work on target location
            # =============================================================================
            dx = autophot_input['dx']
            dy = autophot_input['dy']
            pars = lmfit.Parameters()
            pars.add('A',
                     value = np.nanmax(target_close_up)*0.55,
                     min = 1e-6)
            pars.add('x0',value = target_close_up.shape[1]/2,
                     min = target_close_up.shape[1]/2 - dx,
                     max = target_close_up.shape[1]/2 + dx)
            pars.add('y0',value = target_close_up.shape[0]/2,
                     min = target_close_up.shape[0]/2 - dy,
                     max = target_close_up.shape[0]/2 + dy)
            pars.add('sky',value = np.nanmedian(target_close_up))

            if autophot_input['fitting']['use_moffat']:
                pars.add('alpha',value = autophot_input['image_params']['alpha'],
                         min = 0,
                         max = gauss_fwhm2sigma(autophot_input['source_detection']['max_fit_fwhm']),
                         vary = False)
                pars.add('beta',value = autophot_input['image_params']['beta'],
                         min = 0,
                         vary = autophot_input['fitting']['vary_moff_beta']
                        )
            else:
                pars.add('sigma',value = autophot_input['image_params']['sigma'],
                         min = 0,
                         max = gauss_fwhm2sigma(autophot_input['source_detection']['max_fit_fwhm']),
                         vary = False)
            if autophot_input['fitting']['use_moffat']:
                def residual(p):
                    p = p.valuesdict()
                    return (target_close_up - moffat_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(target_close_up.shape)).flatten()
            else:
                def residual(p):
                    p = p.valuesdict()
                    return (target_close_up - gauss_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(sigma=p['sigma'])).reshape(target_close_up.shape)).flatten()
            mini = lmfit.Minimizer(residual,
                                   pars,
                                   nan_policy = 'omit')
            result = mini.minimize(method = autophot_input['fitting']['fitting_method'])
            
            if autophot_input['fitting']['use_moffat']:
                fitting_model = moffat_2d
                fitting_model_fwhm = moffat_fwhm
            else:
                fitting_model = gauss_2d
                fitting_model_fwhm = gauss_sigma2fwhm
                
            if autophot_input['fitting']['use_moffat']:
                target_fwhm = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
            else:
                target_fwhm = fitting_model_fwhm(dict(sigma=result.params['sigma']))
                
            target_x_pix_corr =  result.params['x0'].value
            target_y_pix_corr =  result.params['y0'].value
            positions  = list(zip([target_x_pix_corr],[target_y_pix_corr]))
            target_counts,target_counts_err,target_max_pixel,target_bkg,target_bkg_std = measure_aperture_photometry(positions,
                                                             target_close_up,
                                                             ap_size = autophot_input['photometry']['ap_size']    * image_fwhm,
                                                             r_in   = autophot_input['photometry']['r_in_size']  * image_fwhm,
                                                             r_out  = autophot_input['photometry']['r_out_size'] * image_fwhm)

            if target_counts < 0:
                logging.info('Target has negative aperture counts, setting to 0 counts')
                target_counts = np.array([0])

            target_flux = (target_counts/exp_time)[0]
            target_bkg_flux = (target_bkg/exp_time)[0]
            target_bkg_std_flux = (target_bkg_std/exp_time)[0]
            target_height_flux = np.array(target_max_pixel/exp_time)[0]

            if subtraction_ready:
                logging.info('Setting target background to zero in template subtraction image')
                target_bkg_flux = 0 

            SNR_target = SNR(flux_star = target_flux ,
                             flux_sky = target_bkg_flux ,
                             exp_t = autophot_input['exp_time'],
                             radius = autophot_input['photometry']['ap_size']*autophot_input['fwhm'] ,
                             G  = autophot_input['gain'],
                             RN =  autophot_input['rdnoise'],
                             DC = 0 )
            
            if not do_ap and not autophot_input['photometry']['do_ap_phot'] :
                
                if approx_psf_mag - calc_mag(target_flux,autophot_input['gain'],0)  > -1:
                    
                    if not autophot_input['photometry']['force_psf']:
                        logging.warning('PSF not applicable')
                        logging.warning('target mag [%.3f] -  PSF mag [%.3f] > 1' % (calc_mag(target_flux,autophot_input['gain'],0),approx_psf_mag))
                        logging.info('set "force_psf" = True to fix')
                        do_ap = True
                        ap_corr = ap_corr_base
                        ap_corr_err = ap_corr_base_err
                        
            if autophot_input['target_photometry']['save_target_plot'] and (do_ap or (autophot_input['template_subtraction']['do_ap_on_sub'] and subtraction_ready)):
                
                    
                border_msg('Doing Aperture Photometry on Target')
                target_err = 0
                plot_aperture(close_up = target_close_up,
                              target_x_pix_corr=target_x_pix_corr,
                              target_y_pix_corr=target_y_pix_corr,
                              fwhm = autophot_input['fwhm'],
                              ap_size = autophot_input['photometry']['ap_size'],
                              r_in_size = autophot_input['photometry']['r_in_size'],
                              r_out_size = autophot_input['photometry']['r_out_size'],
                              write_dir = autophot_input['write_dir'],
                              base = autophot_input['base'],
                              background_value = target_bkg)
                
            # print(subtraction_ready , autophot_input['template_subtraction']['do_ap_on_sub'], do_ap)    
            else:
    
                border_msg('Performing PSF photometry on at target location')
                tagret_loc = pd.DataFrame(data = [[target_x_pix,target_y_pix]],
                                          columns = ['x_pix','y_pix'])
                
                
                c_psf_target,target_close_up = psf.fit(image = image,
                                                        sources = tagret_loc,
                                                        residual_table = r_table,
                                                        fwhm = autophot_input['fwhm'],
                                                        fpath = autophot_input['fpath'],
                                                        fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                        regriding_size = autophot_input['psf']['regriding_size'],
                                                        scale = autophot_input['scale'],
                                                        bkg_level = autophot_input['fitting']['bkg_level'],
                                                        remove_sat = autophot_input['source_detection']['remove_sat'],
                                                        sat_lvl = autophot_input['sat_lvl'],
                                                        use_moffat = autophot_input['fitting']['use_moffat'],
                                                        image_params = autophot_input['image_params'],
                                                        fitting_method = autophot_input['fitting']['fitting_method'],
                                                        # return_psf_model = autophot_input['psf']['return_psf_model'],
                                                        save_plot = autophot_input['target_photometry']['save_target_plot'],
                                                    
                                                        return_fwhm = True,
                                                        return_subtraction_image = autophot_input['psf']['return_subtraction_image'],
                                                        
                                                        return_closeup = True,
                                                        remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                        remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                        remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                        remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                                        
                                                        plot_PSF_residuals = autophot_input['psf']['plot_PSF_residuals'])
                c_psf_target =psf.do(df = c_psf_target,
                                        residual_image = r_table,
                                        ap_size = autophot_input['photometry']['ap_size'],
                                        fwhm = autophot_input['fwhm'],
                                        unity_PSF_counts = autophot_input['unity_PSF_counts'],
            
                                        use_moffat = autophot_input['fitting']['use_moffat'],
                                        image_params = autophot_input['image_params'])
                
                
                target_x_pix = c_psf_target['x_fitted']
                target_y_pix = c_psf_target['y_fitted']

                target_counts = c_psf_target.psf_counts

                target_flux = np.array(target_counts/exp_time)[0]
                target_bkg_flux = np.array(c_psf_target.bkg/exp_time)[0]
                target_bkg_std_flux = np.array(c_psf_target.noise/exp_time)[0]
                target_height_flux = np.array(c_psf_target.H_psf/exp_time)[0]

                target_err = np.array(c_psf_target.psf_counts_err/exp_time)[0]

                if subtraction_ready:
                    logging.info('Setting target background to zero in template subtraction image')
                    target_bkg_flux = 0 

                SNR_target = SNR(flux_star = target_flux ,
                                 flux_sky = target_bkg_flux ,
                                 exp_t = autophot_input['exp_time'],
                                 radius = autophot_input['photometry']['ap_size']*autophot_input['fwhm'] ,
                                 G  = autophot_input['gain'],
                                 RN =  autophot_input['rdnoise'],
                                 DC = 0 )
                target_fwhm = c_psf_target['target_fwhm'].values[0]
                
     

            logging.info('Approximate Target SNR: %.1f' % SNR_target)

            # Detection probability - i.e. what is the likelyhood that this detection is assocaited with a noise spike
            

            target_beta = beta_value(n=3,
                                     sigma = target_bkg_std_flux,
                                     f_ul = target_height_flux)

            # =============================================================================
            # Limiting Magnitude
            # =============================================================================
             

            lmag_check = True
            expand_scale =  1.5+(int(np.ceil((autophot_input['limiting_magnitude']['inject_source_location'] * autophot_input['fwhm']) + autophot_input['scale'])))
            close_up_expand = image_copy[int(target_y_pix - expand_scale): int(target_y_pix + expand_scale),
                                         int(target_x_pix - expand_scale): int(target_x_pix + expand_scale)]

            model = PSF_MODEL
            if SNR_target > autophot_input['limiting_magnitude']['lmag_check_SNR'] and not autophot_input['limiting_magnitude']['force_lmag'] and target_beta > 0.98:
                    lmag_prob = np.nan
                    lmag_inject = np.nan
                    output.update({'lmag_prob':lmag_prob})
                    output.update({'lmag_inject':lmag_inject})
                    logging.info('SNR = %.f - skipping limiting magnitude' % SNR_target)
                    lmag_check = False
                    
            else:

                logging.info('Discrepancy in FWHM of %.1f pixels' % abs(target_fwhm - image_fwhm))
                logging.info('Detection Probability %.1f %%' % abs(target_beta*100))
                
                if autophot_input['limiting_magnitude']['probable_limit']:
                    # print(autophot_input['image_params'])
                    lmag_prob_inst = limiting_magnitude_prob(image = close_up_expand,
                                                                            model = model,
                                                                            r_table = r_table,
                                                                            fpath = autophot_input['fpath'],
                                                                            lim_SNR = autophot_input['limiting_magnitude']['lim_SNR'],
                                                                            bkg_level = autophot_input['fitting']['bkg_level'],
                                                                            fwhm = autophot_input['fwhm'],
                                                                            ap_size = autophot_input['photometry']['ap_size'],
                                                                            exp_time = autophot_input['exp_time'],
                                                                            gain = autophot_input['gain'],
                                                                            image_params = autophot_input['image_params'],
                                                                            regriding_size = autophot_input['psf']['regriding_size'],
                                                                            fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                                            inject_source_cutoff_sources = autophot_input['limiting_magnitude']['inject_source_cutoff_sources'],
                                                                            inject_source_location = autophot_input['limiting_magnitude']['inject_source_location'],
                                                                            inject_source_on_target = autophot_input['limiting_magnitude']['inject_source_on_target'],
                                                                            inject_source_random = autophot_input['limiting_magnitude']['inject_source_random'],
                                                                            inject_source_add_noise = autophot_input['limiting_magnitude']['inject_source_add_noise'],
                                                                            use_moffat = autophot_input['fitting']['use_moffat'],
                                                                            unity_PSF_counts = autophot_input['unity_PSF_counts'],
                                                                            print_progress = True,
                                                                            remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                                            remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                                            remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                                            remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                                                            subtraction_ready = autophot_input['subtraction_ready'],
                                                                            injected_sources_use_beta = autophot_input['limiting_magnitude']['injected_sources_use_beta'])

                    lmag_prob = lmag_prob_inst + zp_measurement[0]
                    print('Probable Limiting Magnitude: %.1f [mag]' % lmag_prob)
                else:
                    lmag_prob = np.nan

                if autophot_input['limiting_magnitude']['inject_sources']:
                    if not np.isnan(lmag_prob):
                        lmag_guess = lmag_prob[0]
                    else:
                        lmag_guess = None
                    
             
                    lmag_inject_inst = inject_sources(image = close_up_expand,
                                                                    fwhm = autophot_input['fwhm'],
                                                                    fpath = autophot_input['fpath'],
                                                                    exp_time = autophot_input['exp_time'],
                                                                    ap_size = autophot_input['photometry']['ap_size'],
                                                                    scale = autophot_input['scale'],
                                                                    zeropoint = zp_measurement[0],
                                                                    r_in_size = autophot_input['photometry']['r_in_size'],
                                                                    r_out_size = autophot_input['photometry']['r_out_size'],
                                                                    injected_sources_use_beta = autophot_input['limiting_magnitude']['injected_sources_use_beta'],
                                                                    beta_limit = autophot_input['limiting_magnitude']['beta_limit'],
                                                                    gain = autophot_input['gain'],
                                                                    rdnoise = autophot_input['rdnoise'],
                                                                    inject_lmag_use_ap_phot = autophot_input['limiting_magnitude']['inject_lmag_use_ap_phot'],
                                                                    use_moffat = autophot_input['fitting']['use_moffat'],
                                                                    image_params = image_params,
                                                                    fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                                    regriding_size = autophot_input['psf']['regriding_size'],
                                                                    lim_SNR = autophot_input['limiting_magnitude']['lim_SNR'],
                                                                    bkg_level = autophot_input['fitting']['bkg_level'],
                                                                    inject_source_recover_dmag = autophot_input['limiting_magnitude']['inject_source_recover_dmag'],
                                                                    inject_source_recover_fine_dmag = autophot_input['limiting_magnitude']['inject_source_recover_fine_dmag'],
                                                                    inject_source_mag = autophot_input['limiting_magnitude']['inject_source_mag'],
                                                                    inject_source_recover_nsteps = autophot_input['limiting_magnitude']['inject_source_recover_nsteps'],
                                                                    inject_source_recover_dmag_redo = autophot_input['limiting_magnitude']['inject_source_recover_dmag_redo'],
                                                                    inject_source_cutoff_sources = autophot_input['limiting_magnitude']['inject_source_cutoff_sources'],
                                                                    inject_source_cutoff_limit = autophot_input['limiting_magnitude']['inject_source_cutoff_limit'],
                                                                    subtraction_ready = autophot_input['subtraction_ready'],
                                                                    unity_PSF_counts = unity_PSF_counts,
                                                                    inject_source_add_noise = autophot_input['limiting_magnitude']['inject_source_add_noise'],
                                                                    inject_source_location = autophot_input['limiting_magnitude']['inject_source_location'],
                                                                    injected_sources_additional_sources = autophot_input['limiting_magnitude']['injected_sources_additional_sources'],
                                                                    injected_sources_additional_sources_position = autophot_input['limiting_magnitude']['injected_sources_additional_sources_position'],
                                                                    injected_sources_additional_sources_number = autophot_input['limiting_magnitude']['injected_sources_additional_sources_number'],
                                                                    plot_injected_sources_randomly = autophot_input['limiting_magnitude']['plot_injected_sources_randomly'],
                                                                    injected_sources_save_output = autophot_input['limiting_magnitude']['injected_sources_save_output'],
                                                                    model = model,
                                                                    r_table = r_table,
                                                                    print_progress = True,
                                                                    
                                              
                                                                    lmag_guess = lmag_guess,
                                                             
                                                                    fitting_method = autophot_input['fitting']['fitting_method'],
                                                                    remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                                    remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                                    remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                                    remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'])


                    lmag_inject = lmag_inject_inst + zp_measurement[0]
                else:
                    lmag_inject = np.nan
                    
                output.update({'lmag_prob':lmag_prob})
                output.update({'lmag_inject':lmag_inject})

        # =============================================================================
        # Check limiting magnitudes of catalog nondetections      
        # =============================================================================

            if autophot_input['limiting_magnitude']['check_catalog_nondetections']:
                border_msg('Performing catalog non detections analysis')
    
                sample_size = 100
                counter = 0
    
                c_nondetect = c[c.SNR<25].sample(sample_size)
                c_nondetect["inject_lmag"] = [np.nan] * len(c_nondetect)
    
                for index,catalog_x_pix,catalog_y_pix,catalog_magnitude in zip(c_nondetect.index,c_nondetect.x_pix.values,c_nondetect.y_pix.values,c_nondetect['cat_'+use_filter]):
                    logging.info('\nPerforming analysis on  source %d / %d' % (counter,sample_size))
    
                    catalog_close_up_expand = image_copy[int(catalog_y_pix - expand_scale): int(catalog_y_pix + expand_scale),
                                                 int(catalog_x_pix - expand_scale): int(catalog_x_pix + expand_scale)]
    
                    catalog_lmag_prob_inst = limiting_magnitude_prob(image = catalog_close_up_expand,
                                                                            model = model,
                                                                            r_table = r_table,
                                                                            fpath = autophot_input['fpath'],
                                                                            lim_SNR = autophot_input['limiting_magnitude']['lim_SNR'],
                                                                            bkg_level = autophot_input['fitting']['bkg_level'],
                                                                            fwhm = autophot_input['fwhm'],
                                                                            ap_size = autophot_input['photometry']['ap_size'],
                                                                            exp_time = autophot_input['exp_time'],
                                                                            gain = autophot_input['gain'],
                                                                            image_params = autophot_input['image_params'],
                                                                            regriding_size = autophot_input['psf']['regriding_size'],
                                                                            fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                                            inject_source_cutoff_sources = autophot_input['limiting_magnitude']['inject_source_cutoff_sources'],
                                                                            inject_source_location = autophot_input['limiting_magnitude']['inject_source_location'],
                                                                            inject_source_on_target = autophot_input['limiting_magnitude']['inject_source_on_target'],
                                                                            inject_source_random = autophot_input['limiting_magnitude']['inject_source_random'],
                                                                            inject_source_add_noise = autophot_input['limiting_magnitude']['inject_source_add_noise'],
                                                                            use_moffat = autophot_input['fitting']['use_moffat'],
                                                                            unity_PSF_counts = autophot_input['unity_PSF_counts'],
                                                                            print_progress = False,
                                                                            remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                                            remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                                            remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                                            remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                                                            subtraction_ready = autophot_input['subtraction_ready'],
                                                                            injected_sources_use_beta = autophot_input['limiting_magnitude']['injected_sources_use_beta'])
                    
                    catalog_lmag_prob = catalog_lmag_prob_inst + zp_measurement[0]
                    
                    
                    catalog_lmag_inject_inst =       inject_sources(image = catalog_close_up_expand,
                                                    fwhm = autophot_input['fwhm'],
                                                    fpath = autophot_input['fpath'],
                                                    exp_time = autophot_input['exp_time'],
                                                    ap_size = autophot_input['photometry']['ap_size'],
                                                    scale = autophot_input['scale'],
                                                    zeropoint = zp_measurement[0],
                                                    r_in_size = autophot_input['photometry']['r_in_size'],
                                                    r_out_size = autophot_input['photometry']['r_out_size'],
                                                    injected_sources_use_beta = autophot_input['limiting_magnitude']['injected_sources_use_beta'],
                                                    beta_limit = autophot_input['limiting_magnitude']['beta_limit'],
                                                    gain = autophot_input['gain'],
                                                    rdnoise = autophot_input['rdnoise'],
                                                    inject_lmag_use_ap_phot = autophot_input['limiting_magnitude']['inject_lmag_use_ap_phot'],
                                                    use_moffat = autophot_input['fitting']['use_moffat'],
                                                    image_params = image_params,
                                                    fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                    regriding_size = autophot_input['psf']['regriding_size'],
                                                    lim_SNR = autophot_input['limiting_magnitude']['lim_SNR'],
                                                    bkg_level = autophot_input['fitting']['bkg_level'],
                                                    inject_source_recover_dmag = autophot_input['limiting_magnitude']['inject_source_recover_dmag'],
                                                    inject_source_recover_fine_dmag = autophot_input['limiting_magnitude']['inject_source_recover_fine_dmag'],
                                                    inject_source_mag = autophot_input['limiting_magnitude']['inject_source_mag'],
                                                    inject_source_recover_nsteps = autophot_input['limiting_magnitude']['inject_source_recover_nsteps'],
                                                    inject_source_recover_dmag_redo = autophot_input['limiting_magnitude']['inject_source_recover_dmag_redo'],
                                                    inject_source_cutoff_sources = autophot_input['limiting_magnitude']['inject_source_cutoff_sources'],
                                                    inject_source_cutoff_limit = autophot_input['limiting_magnitude']['inject_source_cutoff_limit'],
                                                    subtraction_ready = autophot_input['subtraction_ready'],
                                                    unity_PSF_counts = unity_PSF_counts,
                                                    inject_source_add_noise = autophot_input['limiting_magnitude']['inject_source_add_noise'],
                                                    inject_source_location = autophot_input['limiting_magnitude']['inject_source_location'],
                                                    injected_sources_additional_sources = autophot_input['limiting_magnitude']['injected_sources_additional_sources'],
                                                    injected_sources_additional_sources_position = autophot_input['limiting_magnitude']['injected_sources_additional_sources_position'],
                                                    injected_sources_additional_sources_number = autophot_input['limiting_magnitude']['injected_sources_additional_sources_number'],
                                                    plot_injected_sources_randomly = autophot_input['limiting_magnitude']['plot_injected_sources_randomly'],
                                                    injected_sources_save_output = autophot_input['limiting_magnitude']['injected_sources_save_output'],
                                                    model = model,
                                                    r_table = r_table,
                                                    print_progress = False,
                                                    
                              
                                                    lmag_guess = catalog_lmag_prob[0],
                                             
                                                    fitting_method = autophot_input['fitting']['fitting_method'],
                                                    remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                    remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                    remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                    remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'])

                    
    
                    catalog_lmag_inject = catalog_lmag_inject_inst + zp_measurement[0]
    
                    logging.info('\nCatalog Magnitude: %.3f [mag]\nInjected:%.3f [mag]\nProbable:%.3f [mag]\n' % (catalog_magnitude,catalog_lmag_inject,catalog_lmag_prob))
    
                    c_nondetect.at[index, "inject_lmag"] = catalog_lmag_inject
                    c_nondetect.at[index, "prob_lmag"] = catalog_lmag_prob
                    counter+=1
                    # logging.info('')
    
                c_nondetect.round(6).to_csv(autophot_input['write_dir']+'catalog_non_detection_analysis_'+str(base.split('.')[0])+'_filter_'+str(use_filter)+'.csv',index = False)
                
                
            # =============================================================================
            #  TODO: wrong place for this -  Apply extinction airmass correction
            # =============================================================================

            if autophot_input['extinction']['apply_airmass_extinction']:
                

                if 'extinction' not in tele_autophot_input[telescope]:
                    airmass_correction = np.nan

                else:
                    airmass_correction = find_airmass_extinction(tele_autophot_input[telescope]['extinction'],headinfo,autophot_input)
                    find_airmass_extinction(extinction_dictionary = tele_autophot_input[telescope]['extinction'],
                                            headinfo = headinfo,
                                            image_filter = autophot_input['image_filter'],
                                            airmass_key = AIRMASS_key)
                output.update({'airmass_ext':float(airmass_correction)})
                
            # =============================================================================
            # Error on target magnitude
            # =============================================================================

            # Error due to SNR of target
            SNR_error = SNR_err(SNR_target)

            fit_error  = calc_mag(target_flux,autophot_input['gain'],0) - calc_mag(target_flux+target_err,autophot_input['gain'],0)

            if autophot_input['error']['target_error_compute_multilocation'] and not do_ap  :

                # fit_error_multiloc = compute_multilocation_err(close_up_expand,
                                                                # autophot_input,
                                                                # xfit = close_up_expand.shape[1]/2,
                                                                # yfit = close_up_expand.shape[0]/2,
                                                                # Hfit = c_psf_target['H_psf'].values[0],
                                                                # MODEL = model,
                                                                # r_table = r_table
                                                                # )

                fit_error_multiloc = compute_multilocation_err(image = close_up_expand,
                                                          fwhm = autophot_input['fwhm'],
                                                          PSF_model = model,
                                                          image_params = image_params,
                                                          exp_time = autophot_input['exp_time'],
                                                          fpath = autophot_input['fpath'],
                                                          scale = scale,
                                                          unity_PSF_counts = unity_PSF_counts,
                                                          target_error_compute_multilocation_number = autophot_input['error']['target_error_compute_multilocation_number'],
                                                          target_error_compute_multilocation_position = autophot_input['error']['target_error_compute_multilocation_position'],
                                                          use_moffat = autophot_input['fitting']['use_moffat'],
                                                          fitting_method = autophot_input['fitting']['fitting_method'],
                                                          ap_size = autophot_input['photometry']['ap_size'],
                                                          fitting_radius = autophot_input['fitting']['fitting_radius'],
                                                          regriding_size = autophot_input['psf']['regriding_size'],
                                                          xfit = close_up_expand.shape[1]/2,
                                                          yfit = close_up_expand.shape[0]/2,
                                                          Hfit = c_psf_target['H_psf'].values[0],
                                                          r_table = r_table,
                                                          remove_bkg_local = autophot_input['fitting']['remove_bkg_local'],
                                                          remove_bkg_surface = autophot_input['fitting']['remove_bkg_surface'],
                                                          remove_bkg_poly = autophot_input['fitting']['remove_bkg_poly'],
                                                          remove_bkg_poly_degree = autophot_input['fitting']['remove_bkg_poly_degree'],
                                                          bkg_level = autophot_input['fitting']['bkg_level'])
            
                fit_error = np.sqrt(fit_error_multiloc**2 + fit_error[0]**2)

            else:

                fit_error = fit_error[0]
                
            target_mag_err = SNR_error + fit_error
            location_offset = float(pix_dist(target_x_pix,target_x_pix_TNS,target_y_pix,target_y_pix_TNS))

            # =============================================================================
            # Output
            # =============================================================================

            logging.info('Pixel Offset: %.3f' % location_offset)
            location_offset_dict = {'pixel_offset':location_offset}
            detection_beta = {'beta':target_beta }

            # Don't include aperture correction unless aperture photometry is used
            if not do_ap:
                ap_corr = 0
                ap_corr_err = 0 
            else:
                ap_corr = ap_corr_base
                ap_corr_err = ap_corr_base_err
                
            # TODO: add in corrections
            mag_target = calc_mag(target_flux,autophot_input['gain'],zp_measurement[0]) + ap_corr
            mag_inst={use_filter+'_inst':calc_mag(target_flux,autophot_input['gain'],0)}

            mag_inst_err={use_filter+'_inst_err':target_mag_err}

            ap_corr_out = {'aperature_correction':ap_corr_base}
            ap_corr_err_out = {'aperature_correction_err':ap_corr_base_err}
            zp={'zp_'+use_filter:zp_measurement[0]}

            zp_err={'zp_'+use_filter+'_err':zp_measurement[1]}
                
            fwhm_out={'fwhm':image_fwhm,
                      'fwhm_err':image_fwhm_err}
            target_fwhm_out = {'target_fwhm':target_fwhm}
            SNR_dict = {'SNR':SNR_target}
            time_exe = {'time':str(datetime.datetime.now())}
            target_locx = {'xpix':float(target_x_pix)}
            target_locy = {'ypix':float(target_y_pix)}
            mag_target_dict = {use_filter:mag_target}
            mag_err = np.sqrt(target_mag_err**2 + zp_measurement[1]**2 + ap_corr_err**2)
            mag_target_err_dict = {use_filter+'_err':mag_err}
            if do_ap:
                output.update({'method':'ap'})
            else:
                output.update({'method':'psf'})
            if subtraction_ready:
                output.update({'subtraction':True})
            else:
                output.update({'subtraction':False})
            if not lmag_check:
                logging.info('Limiting Magnitude: skipped')
                detection_beta = {'beta':1}
            else:
                logging.info('Probablistic Limiting Magnitude: %.3f' % lmag_prob)
                logging.info('Injected Limiting Magnitude: %.3f' % lmag_inject)
                
            output.update(target_locx)
            output.update(target_locy)
            output.update(mag_inst)
            output.update(mag_inst_err)
            output.update(zp)
            output.update(zp_err)
            output.update(mag_target_dict)
            output.update(mag_target_err_dict)
            output.update(SNR_dict)
            output.update(time_exe)
            output.update(fwhm_out)
            output.update(target_fwhm_out)
            output.update(detection_beta)
            output.update(location_offset_dict)
            output.update(ap_corr_out)
            output.update(ap_corr_err_out)
            output.update({'time_taken':round(time.time() - start_time,1)})

            # =============================================================================
            # Print message to tell about source detection
            # =============================================================================
            if lmag_inject < mag_target or np.isnan(mag_target):
                lim_mag_check = False
            else:
                lim_mag_check = True
            if abs(target_fwhm - image_fwhm) < 0.5:
                fwhm_check = True
            else:
                fwhm_check = False
            # =============================================================================
            # Print final message
            # =============================================================================
            # logging.info(target_bkg_flux,target_bkg_std)
            logging.info('Target Detection probability: %d %%' % (target_beta*100))

            # Print interesting outputs
            logging.info('Target flux: %.3f +/- %.3f [counts/s]'% (target_flux,target_err))
            logging.info('Noise: %.3f [counts/s]' % (target_bkg_std_flux*aperture_area))
            logging.info('Target SNR: %.3f +/- %.3f' % (SNR_target,SNR_error))
            logging.info('Instrumental Magnitude: %.3f +/- %.3f' % (calc_mag(target_flux,autophot_input['gain'],0)[0],fit_error))
            logging.info('Zeropoint: %.3f +/- %.3f' % (zp_measurement[0],zp_measurement[1]))
            logging.info('Target Magnitude: %.3f +/- %.3f ' % (mag_target,mag_err))
            if fwhm_check and lim_mag_check:
                logging.info('\n*** Transient well detected ***\n')
            elif not lim_mag_check :
                logging.info('\n*** Image is magnitude limited ***\n')
            elif not fwhm_check:
                logging.info('\n*** Detected with FWHM discrepancy: %.3f pixels ***\n' % abs(target_fwhm - image_fwhm))
            '''
            Calibration file used in reduction
            - used in color calibration
            '''
            c.round(6).to_csv(autophot_input['write_dir']+'image_calib_'+str(base.split('.')[0])+'_filter_'+str(use_filter)+'.csv',index = False)
            output_file = os.path.join(cur_dir,'out.csv')
            for key,value in output.items():
                if isinstance(value,list):
                    output[key] = value[0]
                if output[key] == np.nan:
                    output[key] = 999

            # print(output)
            target_output = pd.DataFrame(output,columns=output.keys(), index=[0])
            target_output.round(6).to_csv(output_file,index=False)
            # =============================================================================
            # Do photometry on all sources
            # =============================================================================
            # if autophot_input['do_all_phot'] or autophot_input['remove_all_sources']:
            #     logging.info(' \n--- Perform photometry on all sources in field ---')
            #     _,df_all,_= get_fwhm(image,autophot_input,
            #                          sigma_lvl = autophot_input['do_all_phot_sigma'],
            #                          fwhm = image_fwhm)
            #     ra_all,dec_all = w1.all_pix2world(df_all.x_pix.values,df_all.y_pix.values,1 )
            #     df_all['RA'] = ra_all
            #     df_all['DEC'] = dec_all
            #     photfile = 'phot_filter_%s_sigma_%d_%s.csv' % (use_filter,autophot_input['do_all_phot_sigma'],str(base.split('.')[0]))
            #     if do_ap:
            #         positions  = list(zip(df_all.x_pix.values,df_all.y_pix.values))
            #         target_counts,target_maxpixel,target_bkg,target_bkg_std = measure_aperture_photometry(positions,
            #                                     image,
            #                                     radius = autophot_input['photometry']['ap_size']    * image_fwhm,
            #                                     r_in   = autophot_input['photometry']['r_in_size']  * image_fwhm,
            #                                     r_out  = autophot_input['photometry']['r_out_size'] * image_fwhm)
            #         source_flux = (target_counts/exp_time)
            #         source_bkg_flux = (target_bkg/exp_time)
            #         source_noise_flux = target_bkg_std/exp_time
            #         SNR_sources = SNR(flux_star = source_flux ,
            #                     flux_sky = source_bkg_flux,
            #                     exp_t = autophot_input['exp_time'],
            #                     radius = autophot_input['photometry']['ap_size']*autophot_input['fwhm'] ,
            #                     G  = autophot_input['gain'],
            #                     RN =  autophot_input['rdnoise'],
            #                     DC = 0 )
            #         df_all['snr'] = SNR_sources
            #         df_all[use_filter] = calc_mag(source_flux,autophot_input['gain'],zp_measurement[0] ) + ap_corr
            #         mag_err = SNR_err(SNR_sources)
            #         df_all[use_filter+'_err'] =  np.sqrt(mag_err**2 + zp_measurement[1]**2)
            #     else:
            #         positions = df_all[['x_pix','y_pix']]
            #         psf_sources,_ = psf.fit(image,
            #                                 positions,
            #                                 r_table,
            #                                 autophot_input,
            #                                 # image_fwhm
            #                                 )
            #         psf_sources_phot,_ = psf.do(psf_sources,
            #                                     r_table,
            #                                     autophot_input,
            #                                     image_fwhm
            #                                     )
            #         sources_flux = np.array(psf_sources_phot.psf_counts/exp_time)
            #         sources_err = np.array(psf_sources_phot.psf_counts_err/exp_time)
            #         SNR_sources = np.array(sources_flux/sources_err)
            #         mag_err = SNR_err(SNR_sources)
            #         ra_all,dec_all = w1.all_pix2world(psf_sources_phot.x_pix.values,psf_sources_phot.y_pix.values,1 )
            #         df_all = pd.DataFrame([])
            #         df_all['RA'] = ra_all
            #         df_all['DEC'] = dec_all
            #         df_all['snr'] = SNR_sources
            #         df_all['flux_inst'] = psf_sources['H_psf']
            #         df_all[use_filter] = calc_mag(sources_flux,autophot_input['gain'],zp_measurement[0] )
            #         mag_err = SNR_err(SNR_sources)
            #         df_all[use_filter+'_err'] =  np.sqrt(mag_err**2 + zp_measurement[1]**2)
            #     try:
            #         all_phot_loc = autophot_input['write_dir']+photfile
            #         df_all.to_csv(all_phot_loc,index = False)
            #         logging.info('Photometry of all sources saved as: %s' % str(base.split('.')[0])+'.csv')
            #     except Exception as e:
            #         logging.info('Warning - %s \n Table could not be saved to csv' % e)
            #         # flog.close()
            #         pass
            #     if autophot_input['remove_all_sources']:
            #         logging.info(' \n--- Removing all sources in field ---')
            #         df_all['x_pix'] = psf_sources_phot.x_pix.values
            #         df_all['y_pix'] = psf_sources_phot.y_pix.values
            #         psf.fit(image,
            #                 df_all,
            #                 r_table,
            #                 autophot_input,
            #                 # image_fwhm,
            #                 return_psf_model = False,
            #                 return_subtraction_image = True)
            logging.info('Time Taken [ %s ]: %ss' % (str(os.getpid()),round(time.time() - start)))
            logging.info('Sucess: %s :: PID %s \n'%(str(base),str(os.getpid())))
            console.close()
            
            gc.collect()
            return output,base
        # Parent try/except statement for loop
        except Exception as e:
            logging.exception(e)
            logging.critical('Failure: '+ str(base) + ' - PID: '+str(os.getpid()))
            # # console.close()
            # logging.info(cur_dir)
            # fail_dir = os.path.join(cur_dir,'fail')
            # logging.info(fail_dir)
            # pathlib.Path(fail_dir).mkdir(parents = True, exist_ok=True)
            # shutil.move(cur_dir,fail_dir)
            # logging.info('Moving contents to fail folder')
            return None,fpath
    except Exception as e:
        logging.exception(e)
        logging.critical('Fatal Error')
        logging.info('Failure: '+str(base) + ' - PID: '+str(os.getpid()))
        # fail_dir = os.path.join(new_output_dir,'fail')
        # logging.info(fail_dir)
        # pathlib.Path(fail_dir).mkdir(parents = True, exist_ok=True)
        # if os.path.exists(fail_dir):
        #     os.remove(fail_dir)
        # shutil.move(cur_dir,fail_dir)
        # logging.info('Moving contents to fail folder')
        files = (file for file in os.listdir(os.getcwd())
             if os.path.isfile(os.path.join(os.getcwd(), file)))
        for file in files:
             if fpath == file:
                 break
             if os.path.basename(fpath).split('.')[0] in file:
                   shutil.move(os.path.join(os.getcwd(), file),
                    os.path.join(cur_dir, file))
        try:
            console.close()
            c.to_csv('table_ERROR_'+str(base.split('.')[0])+'_filter_'+str(use_filter)+'.csv',index=False)
        except:
            pass

        gc.collect()
        return None,fpath