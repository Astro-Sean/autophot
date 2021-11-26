def prepare_templates(fpath,
                      wdir,
                      write_dir,
                      tele_autophot_input ,
                      get_fwhm = True,
                      build_psf = True,
                      clean_cosmic = True,
                      use_astroscrappy = True,
                      solve_field_exe_loc = None,
                      use_lacosmic = False,
                      use_filter = None,
                      redo_wcs = True,
                      target_ra = None,
                      target_dec = None,
                      search_radius = 0.5,
                      cpu_limit= 180,
                      downsample = 2,
                      threshold_value = 25,
                      ap_size = 1.7,
                      inf_ap_size = 2.5,
                      r_in_size = 2,
                      r_out_size = 3,
                      use_moffat = True,
                      psf_source_no = 10,
                      fitting_method = 'least_sqaure',
                      regriding_size  = 10,
                      fitting_radius = 1.3
                      
                      ):
    
    import os
    import logging
    from astropy.io import fits
    from autophot.packages.functions import getheader,getimage
    from autophot.packages import psf
    from autophot.packages.aperture import do_aperture_photometry
    from autophot.packages.call_astrometry_net import AstrometryNetLOCAL
    from autophot.packages.check_wcs import updatewcs,removewcs
    from autophot.packages.find import get_fwhm
        
    base = os.path.basename(fpath)
    write_dir = os.path.dirname(fpath)
    base = os.path.splitext(base)[0]


    logger = logging.getLogger(__name__)


    logger.info('Preparing templates')
    logger.info('Write Directory: %s' % write_dir )


    if base.startswith('PSF_model'):
        logger.info('ignoring PSF_MODEL in file: %s' % base)
        return

    dirpath = os.path.dirname(fpath)
    
    image = getimage(fpath)
    headinfo = getheader(fpath)
    
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



    if 'rdnoise' in tele_autophot_input[telescope][inst_key][inst]:

        rdnoise_key = tele_autophot_input[telescope][inst_key][inst]['rdnoise']

        if rdnoise_key is None:
            rdnoise = 0 
        elif rdnoise_key in headinfo:
            rdnoise = headinfo[rdnoise_key]
        else:
            rdnoise = 0 
         
    else:
        logging.info('Read noise key not found for template file')
        rdnoise = 0
        
    logging.info('Read Noise: %.1f [e^- /pixel]' % rdnoise)


    if 'GAIN' in tele_autophot_input[telescope][inst_key][inst]:

        GAIN_key = tele_autophot_input[telescope][inst_key][inst]['GAIN']

        if GAIN_key is None:
            GAIN = 1
        elif GAIN_key in headinfo:
            GAIN = headinfo[GAIN_key]
        else:
            GAIN=1
        
  
    logging.info('Template GAIN: %.1f [e^- /count]' % GAIN)
        
    
    # print(list(headinfo.keys()))
    # try:
    for i in ['EXPTIME','EXP_TIME','TIME-INT']:
        if i in list(headinfo.keys()):
            template_exp_time = headinfo[i]
            break
    # raise Exception()
    # except Exception as e:
    #     template_exp_time = 1
        
    if isinstance(template_exp_time, str):
       template_exp_time_split = template_exp_time.split('/')
       if len(template_exp_time_split)>1:
           template_exp_time = float(template_exp_time_split[0])
       else:
           template_exp_time = float(template_exp_time)
           
       
    logging.info('Template Exposure Time: %.1f [s]' % template_exp_time)
       
    if telescope == 'MPI-2.2':
        if use_filter in ['J','H','K']:
            logging.info('Detected GROND IR - setting pixel scale to 0.3')
            pixel_scale = 0.3
        else:
            logging.info('Detected GROND Optical - setting pixel scale to 0.16')
            pixel_scale = 0.16
    else:

        pixel_scale   = tele_autophot_input[telescope][inst_key][inst]['pixel_scale']


    
    # Write new header
    updated_header = getheader(fpath)
    updated_header['GAIN'] = GAIN
    
    updated_header['exp_time'] = template_exp_time
    updated_header['GAIN'] = GAIN
    updated_header['RDNOISE'] = rdnoise
    
    fits.writeto(fpath,
                 image,
                 updated_header,
                 overwrite = True,
                 output_verify = 'silentfix+ignore')



    

    if clean_cosmic:

        # if 'CRAY_RM'  not in header:

        from autophot.packages.call_crayremoval import run_astroscrappy
        
        if 'CRAY_RM'  not in updated_header:
            headinfo = getheader(fpath)
            # image with cosmic rays
            image_old = fits.PrimaryHDU(image)
            
            image = run_astroscrappy(image_old,
                                     gain = GAIN,
                                     use_astroscrappy =use_astroscrappy,
                                     use_lacosmic = use_lacosmic)
        
            # Update header and write to new file
            updated_header['CRAY_RM'] = ('T', 'Comsic rays w/astroscrappy ')
            fits.writeto(fpath,
                         image,
                         updated_header,
                         overwrite = True,
                         output_verify = 'silentfix+ignore')
            logging.info('Cosmic rays removed - image updated')
        
        else:
        
            logging.info('Cosmic sources pre-cleaned - skipping!')
        
        
    if redo_wcs:

        
        # Run local instance of Astrometry.net - returns filepath of wcs file
        astro_check = AstrometryNetLOCAL(fpath,
                                        solve_field_exe_loc = solve_field_exe_loc,
                                        pixel_scale = pixel_scale,
                                        ignore_pointing = False,
                                        target_ra = target_ra,
                                        target_dec = target_dec,
                                        search_radius = search_radius,
                                        downsample = downsample,
                                        cpulimit = cpu_limit)
    
        old_header = getheader(fpath)
    
        try:
            old_header = removewcs(headinfo,delete_keys = True)
    
            # Open wcs fits file with wcs values
            new_wcs  = fits.open(astro_check,ignore_missing_end = True)[0].header
    
            # script used to update per-existing header file with new wcs values
            header_updated = updatewcs(old_header,new_wcs)
    
            # update header to show wcs has been checked
            header_updated['UPWCS'] = ('T', 'WCS by APT')
    
            
            os.remove(astro_check)
    
        except Exception as e:
            
            logger.info('Error with template WCS: %s' % e)
    
        # Write new header
        fits.writeto(fpath,image,
                     header_updated,
                     overwrite = True,
                     output_verify = 'silentfix+ignore')



    if get_fwhm or build_psf:
 


        template_fwhm,df,scale,image_params = get_fwhm(image,
                                                   write_dir,
                                                   base,
                                                   threshold_value = threshold_value,
                                                   # fwhm_guess = autophot_input['source_detection']['fwhm_guess,
                                                   # bkg_level = autophot_input['fitting']['bkg_level'],
                                                   # max_source_lim = autophot_input['source_detection']['max_source_lim'],
                                                   # min_source_lim = autophot_input['source_detection']['min_source_lim'],
                                                   # int_scale = autophot_input['source_detection']['int_scale'],
                                                   # fudge_factor = autophot_input['source_detection']['fudge_factor'],
                                                   # fine_fudge_factor = autophot_input['source_detection']['fine_fudge_factor'],
                                                   # source_max_iter = autophot_input['source_detection']['source_max_iter'],
                                                   # sat_lvl = autophot_input['sat_lvl'],
                                                   # lim_SNR = autophot_input['limiting_magnitude']['lim_SNR'],
                                                   # scale_multipler = autophot_input['source_detection']['scale_multipler'],
                                                   # sigmaclip_FWHM = autophot_input['source_detection']['sigmaclip_FWHM'],
                                                   # sigmaclip_FWHM_sigma = autophot_input['source_detection']['sigmaclip_FWHM_sigma'],
                                                   # sigmaclip_median = autophot_input['source_detection']['sigmaclip_median'],
                                                   # isolate_sources = autophot_input['source_detection']['isolate_sources'],
                                                   # isolate_sources_fwhm_sep = autophot_input['source_detection']['isolate_sources_fwhm_sep'],
                                                   # init_iso_scale = autophot_input['source_detection']['init_iso_scale'],
                                                   # remove_boundary_sources = autophot_input['source_detection']['remove_boundary_sources'],
                                                   # pix_bound = autophot_input['source_detection']['pix_bound'],
                                                   # sigmaclip_median_sigma = autophot_input['source_detection']['sigmaclip_median_sigma'],
                                                   # save_FWHM_plot = autophot_input['source_detection']['save_FWHM_plot'],
                                                   # plot_image_analysis = autophot_input['source_detection']['plot_image_analysis'],
                                                   # save_image_analysis = autophot_input['source_detection']['save_image_analysis'],
                                                   # use_local_stars_for_FWHM = autophot_input['photometry']['use_local_stars_for_FWHM'],
                                                    prepare_templates = True,
                                                   # image_filter = autophot_input['image_filter'],
                                                   # target_name = autophot_input['target_name'],
                                                   # target_x_pix = None,
                                                   # target_y_pix = None,
                                                   # local_radius = autophot_input['photometry']['local_radius'],
                                                   # mask_sources = autophot_input['preprocessing']['mask_sources'],
                                                   # mask_sources_XY_R = None,
                                                   # remove_sat = autophot_input['source_detection']['remove_sat'],
                                                    use_moffat = use_moffat ,
                                                   # default_moff_beta = autophot_input['fitting']['default_moff_beta'],
                                                   # vary_moff_beta = autophot_input['fitting']['vary_moff_beta'],
                                                   # max_fit_fwhm = autophot_input['source_detection']['max_fit_fwhm'],
                                                   # fitting_method = autophot_input['fitting']['fitting_method']
                                                   )
        # autophot_input['fwhm'] = image_fwhm
        # autophot_input['scale'] = scale
        # autophot_input['image_params'] = image_params
        # header['FWHM'] = image_fwhm
        
        

        df.to_csv(os.path.join(dirpath,'calib_template.csv'),index = False)



    if build_psf:

        df_PSF = do_aperture_photometry(image = image,
                                        dataframe = df,
                                        fwhm = template_fwhm,
                                        ap_size = ap_size,
                                        inf_ap_size =inf_ap_size,
                                        r_in_size =r_in_size,
                                        r_out_size = r_out_size)

        psf.build_r_table(base_image = image,
                            selected_sources = df_PSF,
                            fwhm = template_fwhm,
                            exp_time = template_exp_time,
                            image_params = image_params,
                            fpath = fpath,
                            GAIN = GAIN,
                            rdnoise = rdnoise,
                            use_moffat = use_moffat,
                            vary_moff_beta = False,
                            fitting_radius = fitting_radius,
                            regrid_size = regriding_size,
                            # use_PSF_starlist = autophot_input['psf']['use_PSF_starlist'],
                            use_local_stars_for_PSF = False,
                            prepare_templates = True,
                            scale = scale,
                            ap_size = ap_size,
                            r_in_size = r_in_size,
                            r_out_size = r_out_size,
                            # local_radius = autophot_input['photometry']['local_radius'],
                            # bkg_level = autophot_input['fitting']['bkg_level'],
                            psf_source_no = psf_source_no,
                            # min_psf_source_no = autophot_input['psf']['min_psf_source_no'],
                            construction_SNR = 5,
                            remove_bkg_local = True,
                            remove_bkg_surface = False,
                            remove_bkg_poly = False,
                            remove_bkg_poly_degree = 1,
                            # fit_PSF_FWHM = autophot_input['psf']['fit_PSF_FWHM'],
                            # max_fit_fwhm = autophot_input['source_detection']['max_fit_fwhm'],
                            fitting_method = fitting_method,
                            save_PSF_stars = False,
                            # plot_PSF_model_residuals = autophot_input['psf']['plot_PSF_model_residuals'],
                            save_PSF_models_fits = True)


        pass


    return










def subtract(file,template,image_fwhm,use_zogy = False,hotpants_exe_loc = None,
             hotpants_timeout=45,
             template_dir = None,psf = None, mask_border = False, pix_bound = None,footprint = None,
             remove_sat  = False,zogy_use_pixel = False):

    import subprocess
    import os
    import sys
    import numpy as np
    from pathlib import Path
    import signal
    import time
    from autophot.packages.functions import getimage,getheader
    from astropy.io import fits
    import logging
    import warnings
    from astropy.wcs import WCS

    logger = logging.getLogger(__name__)
    
    base = os.path.basename(file)
    write_dir = os.path.dirname(file)
    base = os.path.splitext(base)[0]

    
    logger.info('\nImage subtracion')
    if hotpants_exe_loc is None and not use_zogy:
        use_zogy = True

        logger.info('HOTPANTS selected but exe file location not found, trying PyZogy')
        hotpants_exe_loc = True
        
    if use_zogy: # Check if zogy is available 
    
        try:
            from PyZOGY.subtract import run_subtraction 
            use_zogy = True
        except ImportError as e:
            logger.info('PyZogy selected but not installed: %s' % e)
            use_zogy = False
            
    if use_zogy and not hotpants_exe_loc:
        warnings.warn('No suitable template subtraction package found/nPlease check installation instructions!,/n returning original image')
        return np.nan
        
    
    try:

        # convolve_image = False
        # smooth_template = False
        
        # Get file extension and template data
        fname_ext = Path(file).suffix

        # Open image and template
        file_image     = getimage(file)
        
        image_header = getheader(file)
        
        original_wcs = WCS(image_header)

        # header = getheader(file)
        template_image = getimage(template)
        # template_header = getheader(template)

        # Where the subtraction will be written
        output_fpath = str(file.replace(fname_ext,'_subtraction'+fname_ext))
        
        # Create footprint
        footprint = np.zeros(file_image.shape).astype(bool)
        
        # footprint_template = np.zeros(template_image.shape).astype(bool)
        
        footprint[ ( np.isnan(file_image)) | np.isnan(template_image) ] = 1
            
        # footprint = abs(footprint)
        if mask_border:
            
            if not (pix_bound is None):
                pix_bound = pix_bound
                
            footprint[pix_bound: - pix_bound,
                      pix_bound: - pix_bound] = False
        
        hdu = fits.PrimaryHDU(footprint.astype(int))
        hdul = fits.HDUList([hdu])

        footprint_loc = os.path.join(write_dir,'footprint_'+base+fname_ext)
        
        hdul.writeto(footprint_loc,
                      overwrite=True,
                      output_verify = 'silentfix+ignore')

        check_values_image = file_image[~footprint]

        check_values_template = template_image[~footprint]

        if  remove_sat  :

            image_max = [np.nanmax(check_values_image) if np.nanmax(check_values_image) < 2**16 else  + 2**16][0]

            template_max = [np.nanmax(check_values_template) if np.nanmax(check_values_template) < 2**16 else  2**16][0]
            
        else:
            
            image_max = np.nanmax(check_values_image)

            template_max = np.nanmax(np.nanmax(check_values_template))
        
        if use_zogy:
            try:

                # Get filename for saving
                base = os.path.splitext(os.path.basename(file))[0]
                
                logger.info('Performing image subtraction using PyZOGY')
                
                # PyZOGY_log = write_dir + base + '_ZOGY.txt'
                # original_stdout = sys.stdout # Save a reference to the original standard output
    
                   
                image_psf = os.path.join(write_dir,'PSF_model_'+base.replace('_image_cutout','')+'.fits')
    
                from glob import glob
                template_psf = glob(os.path.join(template_dir,'PSF_model_*'))[0]
                
                logger.info('Using Image : %s' % file)
                logger.info('Using Image PSF: %s' % image_psf)
                logger.info('Using Template : %s' % template)
                logger.info('Using Template PSF: %s' % template_psf)
                
                logger.info('\nRunning Zogy...\n')
                
                # logger.info(image_max,template_max)
                
                diff = run_subtraction(science_image = file,
                                       reference_image = template,
                                       science_psf = image_psf,
                                       reference_psf = template_psf,
                                       reference_mask = footprint,
                                       science_mask = footprint,
                                       show = False,
                                       # sigma_cut = 3,
                                       normalization = "science",
                                       science_saturation = 10+image_max,
                                       reference_saturation = 10+template_max,
                                       n_stamps = 1,
                                       max_iterations = 10,
                                       use_pixels  = zogy_use_pixel
                                        # size_cut = True
                                        )
             
                hdu = fits.PrimaryHDU(diff[0])
                hdul = fits.HDUList([hdu])
                hdul.writeto(str(file.replace(fname_ext,'_subtraction'+fname_ext)),
                             overwrite = True,
                             output_verify = 'silentfix+ignore')
            except Exception as e:
                logger.info('Pyzogy Failed [%s] - trying HOTPANTS' % e)
                use_zogy = False
                
                

        if not use_zogy :
            
            logger.info('Performing image subtraction using HOTPANTS')

            # Get filename for saving
            base = os.path.splitext(os.path.basename(file))[0]

            # Location of executable for hotpants
            exe = hotpants_exe_loc

            # =============================================================================
            # Argurments to send to HOTPANTS process - list of tuples
            # =============================================================================

            # Arguments to pass to HOTPANTS
            
            include_args = [
                    # Input image
                            ('-inim',   str(file)),
                    # Template Image
                            ('-tmplim', str(template)),
                    # Output image name
                            ('-outim',  str(file.replace(fname_ext,'_subtraction'+fname_ext))),
                    # Image lower limits
                            ('-il',     str(np.nanmin(check_values_image))),
                    # Template lower limits
                            ('-tl',     str(np.nanmin(check_values_template))),
                    # Template upper limits
                            ('-tu',     str(template_max+10)),
                    # Image upper limits
                            ('-iu',     str(image_max+10)),
                    # Image mask
                            ('-imi',    str(footprint_loc)),
                    # Template mask
                            ('-tmi',    str(footprint_loc)),
                    # Image gain
                            # ('-ig',     str(autophot_input['GAIN'])),
                    # Template gain
                            # ('-tg',     str(t_header['gain'])),
                    # Normalise to image[i]
                            ('-n',  'i'),
                    # spatial order of kernel variation within region  
                            ('-ko', '1'),
                    # Verbosity - set to as output is sent to file
                            ('-v' , ' 0') ,
                    # number of each region's stamps in x dimension
                            ('-nsx' , '11') ,
                    # number of each re5ion's stamps in y dimension
                            ('-nsy' , '11'), 
                    # number of each region's stamps in x dimension
                    # RMS threshold forgood centroid in kernel fit
                            ('-ft' , '20') ,
                    # threshold for sigma clipping statistics
                            ('-ssig' , '5.0') ,
                    # high sigma rejection for bad stamps in kernel fit
                            ('-ks' , '3.0'),
                    # convolution kernel half width
                            # ('-r' , str(1.5*image_FWHM)) ,
                    # number of centroids to use for each stamp 
                            # ('-nss' , str(5))
        
                            ]

            args= [str(exe)]

            for i in include_args:
                args[0] += ' ' + i[0] + ' ' + i[1]
            # =============================================================================
            # Call subprocess using executable location and option prasers
            # =============================================================================

            start = time.time()
            
            HOTPANTS_log = write_dir + base + '_HOTterPANTS.txt'
            
            # logger.info(args, file=open(HOTPANTS_log, 'w'))

    
            with  open(HOTPANTS_log, 'w')  as FNULL:
                
                pro = subprocess.Popen(args,shell=True, stdout=FNULL, stderr=FNULL)
                print('ARGUMENTS:', args, file=FNULL)

                # Timeout
                pro.wait(hotpants_timeout)

                try:
                    # Try to kill process to avoid memory errors / hanging process
                    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                    logger.info('HOTPANTS PID killed')
                    logger.info(args)
                except:
                    pass

            logger.info('HOTPANTS finished: %ss' % round(time.time() - start) )
            
        # =============================================================================
        # Check that subtraction file has been created
        # =============================================================================
        if os.path.isfile(output_fpath):

            file_size = os.path.getsize(str(file.replace(fname_ext,'_subtraction'+fname_ext)))

            if file_size == 0:
                
                logger.info('File was created but nothing written')

                return np.nan
            
            else:
                
                logger.info('Subtraction saved as %s' % os.path.splitext(os.path.basename(file.replace(fname_ext,'_subtraction'+fname_ext)))[0])
                
                original_wcs
                
                template_header = getheader(output_fpath)
                template_image = getimage(output_fpath)
                template_header.update(original_wcs.to_header())
                
                fits.writeto(output_fpath,
                            template_image,
                            template_header,
                            overwrite = True,
                             output_verify = 'silentfix+ignore')
                
                
                return output_fpath
            
        if not os.path.isfile(output_fpath):
            
            logger.info('File was not created')
            
            return np.nan

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.info(exc_type, fname, exc_tb.tb_lineno,e)
        
        try:
                # Try to kill process to avoid memory errors / hanging process
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            logger.info('HOTPANTS PID killed')
        except:
            pass

        return np.nan