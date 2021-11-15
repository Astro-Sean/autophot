def prepare_templates(fpath,
                      autophot_input,
                      get_fwhm = True,
                      clean_cosmic = True,
                      redo_wcs = True,
                      build_psf = True):

    
    
    import os
    import numpy as np
    from astropy.io import fits
    from autophot.packages.functions import getheader,getimage
    from autophot.packages.call_yaml import yaml_autophot_input as cs

    write_dir = os.path.dirname(fpath)
    base = ''.join(os.path.basename(fpath).split('.')[:-1])


    wdir = autophot_input['wdir']
    print('Preparing templates')
    print('Write Directory: %s' % write_dir )
    
    print('Base Directory: %s' %  base)

    if base.startswith('PSF_model'):
        print('ignoring PSF_MODEL in file: %s' % base)
        return

    dirpath = os.path.dirname(fpath)
    
    image = getimage(fpath)
    
    header = getheader(fpath)
    
    try:
        telescope = header['TELESCOP']
    except:
        telescope  = 'UNKNOWN'
        header['TELESCOP'] = 'UNKNOWN'

    if telescope  == '':
        telescope  = 'UNKNOWN'
        header['TELESCOP'] = 'UNKNOWN'

    inst_key = 'INSTRUME'
    inst = header[inst_key]

    tele_autophot_input_yml = 'telescope.yml'
    teledata = cs(os.path.join(wdir,tele_autophot_input_yml))
    tele_autophot_input = teledata.load_vars()
    
    GAIN=1
    
    if 'GAIN' in tele_autophot_input[telescope][inst_key][inst]:
                
        GAIN_key = tele_autophot_input[telescope][inst_key][inst]['GAIN']
 
        if GAIN_key is None:
            GAIN = 1
            
        elif GAIN_key in header:
            
            GAIN = header[GAIN_key]
    else:
        
        GAIN=1
        
    

    try:
        for i in ['EXPTIME','EXP_TIME','TIME-INT']:
            if i in header:
                exp_time = header[i]
                break
        raise Exception
    except:
        exp_time = 1
        
    header['exp_time'] = exp_time
    # autophot_input['exp_time'] = exp_time


    # autophot_input['GAIN'] = 1
    RDNOISE = 1
    # GAIN = GAIN
    exp_time = exp_time
    
    # Write new header
    header = getheader(fpath)
    header['GAIN'] = GAIN
    
    autophot_input['exp_time'] = exp_time
    autophot_input['GAIN'] = GAIN
    autophot_input['RDNOISE'] = 0    
    
    fits.writeto(fpath,
                 image,
                 header,
                 overwrite = True,
                 output_verify = 'silentfix+ignore')

    if isinstance(exp_time, str):
       exp_time = exp_time.split('/')
       exp_time = float(exp_time[0])

    

    if clean_cosmic:

        # if 'CRAY_RM'  not in header:

        from autophot.packages.call_crayremoval import run_astroscrappy
        

        image= run_astroscrappy(fits.PrimaryHDU(image),
                                gain = GAIN,
                                use_astroscrappy = autophot_input['use_astroscrappy'],
                                use_lacosmic = autophot_input['use_lacosmic'])
        
        
    if redo_wcs:
        
        
        import numpy as np
        from autophot.packages.call_astrometry_net import AstrometryNetLOCAL
        from autophot.packages.check_wcs import updatewcs,removewcs
        
        # Run local instance of Astrometry.net - returns filepath of wcs file
        astro_check = AstrometryNetLOCAL(fpath,
                                         NAXIS1 = autophot_input['NAXIS1'],
                                         NAXIS2 = autophot_input['NAXIS2'],
                                         solve_field_exe_loc = autophot_input['solve_field_exe_loc'],
                                         pixel_scale = autophot_input['pixel_scale'],
                                         try_guess_wcs = autophot_input['try_guess_wcs'],
                                         ignore_pointing = autophot_input['ignore_pointing'],
                                          target_ra = autophot_input['target_ra'],
                                          target_dec = autophot_input['target_dec'],
                                         search_radius = autophot_input['search_radius'],
                                         downsample = autophot_input['downsample'],
                                         cpulimit = autophot_input['cpulimit'],
                                         solve_field_timeout = autophot_input['solve_field_timeout']);
    
        old_header = getheader(fpath)
    
        try:
    
            # Open wcs fits file with wcs values
            new_wcs  = fits.open(astro_check,ignore_missing_end = True)[0].header
    
            # script used to update per-existing header file with new wcs values
            header_updated = updatewcs(old_header,new_wcs)
    
            # update header to show wcs has been checked
            header_updated['UPWCS'] = ('T', 'WCS by APT')
    
            updated_wcs = True
            
            os.remove(astro_check)
    
        except Exception as e:
            
            print('Error with template WCS: %s' % fpath)
    
        # Write new header
        fits.writeto(fpath,image,
                     header_updated,
                     overwrite = True,
                     output_verify = 'silentfix+ignore')
        header = getheader(fpath)



    if get_fwhm or build_psf:
 
        from autophot.packages.find import get_fwhm

        image_fwhm,df,scale,image_params = get_fwhm(image,
                                                       write_dir,
                                                       base,
                                                       threshold_value = autophot_input['threshold_value'],
                                                       fwhm_guess = autophot_input['fwhm_guess'],
                                                       bkg_level = autophot_input['bkg_level'],
                                                       max_source_lim = autophot_input['max_source_lim'],
                                                       min_source_lim = autophot_input['min_source_lim'],
                                                       int_scale = autophot_input['int_scale'],
                                                       fudge_factor = autophot_input['fudge_factor'],
                                                       fine_fudge_factor = autophot_input['fine_fudge_factor'],
                                                       source_max_iter = autophot_input['source_max_iter'],
                                                       sat_lvl = autophot_input['sat_lvl'],
                                                       lim_SNR = autophot_input['lim_SNR'],
                                                       scale_multipler = autophot_input['scale_multipler'],
                                                       sigmaclip_FWHM = autophot_input['sigmaclip_FWHM'],
                                                       sigmaclip_FWHM_sigma = autophot_input['sigmaclip_FWHM_sigma'],
                                                       sigmaclip_median = autophot_input['sigmaclip_median'],
                                                       isolate_sources = autophot_input['isolate_sources'],
                                                       isolate_sources_fwhm_sep = autophot_input['isolate_sources_fwhm_sep'],
                                                       init_iso_scale = autophot_input['init_iso_scale'],
                                                       remove_boundary_sources = autophot_input['remove_boundary_sources'],
                                                       pix_bound = autophot_input['pix_bound'],
                                                       sigmaclip_median_sigma = autophot_input['sigmaclip_median_sigma'],
                                                       save_FWHM_plot = autophot_input['save_FWHM_plot'],
                                                       plot_image_analysis = autophot_input['plot_image_analysis'],
                                                       save_image_analysis = autophot_input['save_image_analysis'],
                                                       use_local_stars_for_FWHM = autophot_input['use_local_stars_for_FWHM'],
                                                       prepare_templates = autophot_input['prepare_templates'],
                                                       image_filter = autophot_input['image_filter'],
                                                       target_name = autophot_input['target_name'],
                                                       # target_x_pix = autophot_input['target_x_pix'],
                                                       # target_y_pix = autophot_input['target_y_pix'],
                                                       # local_radius = autophot_input['local_radius'],
                                                       # mask_sources = autophot_input['mask_sources'],
                                                       # mask_sources_XY_R = autophot_input['mask_sources_XY_R'],
                                                       remove_sat = autophot_input['remove_sat'],
                                                       use_moffat = autophot_input['use_moffat'],
                                                       default_moff_beta = autophot_input['default_moff_beta'],
                                                       vary_moff_beta = autophot_input['vary_moff_beta'],
                                                       max_fit_fwhm = autophot_input['max_fit_fwhm'],
                                                       fitting_method = autophot_input['fitting_method'],
                                                       
                                                       
                                                       )
        autophot_input['fwhm'] = image_fwhm
        autophot_input['scale'] = scale
        autophot_input['image_params'] = image_params
        header['FWHM'] = image_fwhm
        
        

        df.to_csv(os.path.join(dirpath,'calib_template.csv'),index = False)



    if build_psf:
        from autophot.packages import psf
        from autophot.packages.aperture import do_aperture_photometry

        df = do_aperture_photometry(image = image,
                                        dataframe = df,
                                        fwhm= autophot_input['fwhm'],
                                        ap_size = autophot_input['ap_size'],
                                        inf_ap_size = autophot_input['inf_ap_size'],
                                        r_in_size = autophot_input['r_in_size'],
                                        r_out_size = autophot_input['r_out_size'])

        r_table,fwhm_fit,psf_heights,autophot_input = psf.build_r_table(image,
                                                                        df,
                                                                        autophot_input,
                                                                        image_fwhm)

        pass


    return







def point_source_align(template_image,image_sources,template_sources):
    
    import numpy as np
    import itertools
    import operator

    aligned_template = np.nan

    image_sources = image_sources.head(10)


    set_B = [list(image_sources['x_pix'].values),list(image_sources['y_pix'].values)]


    set_A = [list(template_sources['x_pix'].values),list(template_sources['y_pix'].values)]

    print(set_A)

    # print(set_A,set_B)

    set_Ax, set_Ay = (set_A)
    set_A = []
    for i in range(len(set_Ax)):
        set_A.append([set_Ax[i],set_Ay[i]])
    # normalize set B data to set A format
    set_Bx, set_By = (set_B)
    set_B = []
    for i in range(len(set_Bx)):
        set_B.append([set_Bx[i],set_By[i]])



    # create index combinations of both lists
    set_A_tri = list(itertools.combinations(range(len(set_A)), 3))
    set_B_tri = list(itertools.combinations(range(len(set_B)), 3))

    # print( set_A_tri )
    # print( set_B_tri )
    # print( len(set_A_tri) )
    # print( len(set_B_tri) )

    def distance(x1,x2,y1,y2):
        return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

    def tri_sides(set_x, set_x_tri):

        triangles = []
        for i in range(len(set_x_tri)):

            point1 = set_x_tri[i][0]
            point2 = set_x_tri[i][1]
            point3 = set_x_tri[i][2]

            point1x, point1y = set_x[point1][0], set_x[point1][1]
            point2x, point2y = set_x[point2][0], set_x[point2][1]
            point3x, point3y = set_x[point3][0], set_x[point3][1]

            len1 = distance(point1x,point1y,point2x,point2y)
            len2 = distance(point1x,point1y,point3x,point3y)
            len3 = distance(point2x,point2y,point3x,point3y)

            min_side = min(len1,len2,len3)
            len1/=min_side
            len2/=min_side
            len3/=min_side
            t=[len1,len2,len3]
            t.sort()
            triangles.append(t)

        return triangles

    A_triangles = tri_sides(set_A, set_A_tri)
    B_triangles = tri_sides(set_B, set_B_tri)



    def list_subtract(list1,list2):

        return np.absolute(np.array(list1)-np.array(list2))

    sums = []
    threshold = 1
    print('working')
    for i in range(len(A_triangles)):
        for j in range(len(B_triangles)):
            k = sum(list_subtract(A_triangles[i], B_triangles[j]))
            if k < threshold:
                sums.append([i,j,k])
    # sort by smallest sum
    sums = sorted(sums, key=operator.itemgetter(2))

    # print( sums )
    print( 'winner %s' % sums[0])
    match_A = set_A_tri[sums[0][0]]
    match_B = set_B_tri[sums[0][1]]
    print( 'triangle A %s matches triangle B %s' % (match_A, match_B) )

    match_A_pts = []
    match_B_pts = []
    for i in range(3):
        match_A_pts.append(set_A[match_A[i]])
        match_B_pts.append(set_B[match_B[i]])

    print( 'triangle A has points %s' % match_A_pts )
    print( 'triangle B has points %s' % match_B_pts )


    print('\n Matching')

    print('Image -> Catalog')
    for i in range(len(match_A_pts)):
        print('%s -> %s'% (match_A_pts[i],match_B_pts[i]))

    print('\nCatalog Matching')


    import astroalign as aa

    # Align template with Image

    # transform, (source_list, target_list) = aa.find_transform(np.array(match_A_pts), np.array(match_B_pts))
    transform = aa.estimate_transform('affine', np.array(match_B_pts), np.array(match_A_pts))

    aligned_template = aa.apply_transform(transform, template_image.astype(float),template_image.astype(float))


    return aligned_template[0]


def subtract(file,template,autophot_input,psf = None, mask_border = False, pix_bound = None,footprint = None ):
    '''
    
    Function for performing template subtractioon Astronmical images using AutoPhot
    
    
    :param file: DESCRIPTION
    :type file: TYPE
    :param template: DESCRIPTION
    :type template: TYPE
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param psf: DESCRIPTION, defaults to None
    :type psf: TYPE, optional
    :param mask_border: DESCRIPTION, defaults to True
    :type mask_border: TYPE, optional
    :param pix_bound: DESCRIPTION, defaults to None
    :type pix_bound: TYPE, optional
    :raises NotImplementedError: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    import subprocess
    import os
    import sys
    import numpy as np
    from pathlib import Path
    import signal
    import time
    from autophot.packages.functions import  getheader,getimage
    from astropy.io import fits
    import logging
    import warnings

    logger = logging.getLogger(__name__)
    
    
    # By default, use HOTPANTS:
    use_hotpants = True
        
        
    
    if autophot_input['solve_field_exe_loc'] is None:
        use_hotpants = False
        if not  autophot_input['use_zogy']:
            print('HOTPANTS exe location not found, trying PyZogy')
            autophot_input['use_zogy'] = True
        
    if autophot_input['use_zogy']:
        try:
            from PyZOGY.subtract import run_subtraction 
            use_hotpants = False
        except ImportError as e:
            print('PyZogy selected but not installed: %s' % e)
            autophot_input['use_zogy'] = False
            
    if not use_hotpants and not autophot_input['use_zogy']:
        warnings.warn('No suitable template subtraction package found/nPlease check installation instructions!,/n returning original image')
        return np.nan
        
        
        

    try:

        # convolve_image = False
        smooth_template = False
        
        # Get file extension and template data
        fname_ext = Path(file).suffix

        # Open image and template
        file_image     = getimage(file)

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
                pix_bound = autophot_input['pix_bound']
                
            footprint[autophot_input['pix_bound']: - autophot_input['pix_bound'],
                      autophot_input['pix_bound']: - autophot_input['pix_bound']] = False
        
        hdu = fits.PrimaryHDU(footprint.astype(int))
        hdul = fits.HDUList([hdu])

        footprint_loc = os.path.join(autophot_input['write_dir'],'footprint_'+autophot_input['base']+fname_ext)
        
        hdul.writeto(footprint_loc,
                      overwrite=True,
                      output_verify = 'silentfix+ignore')

        check_values_image = file_image[~footprint]

        check_values_template = template_image[~footprint]

        if autophot_input['remove_sat']:

            image_max = [np.nanmax(check_values_image) if np.nanmax(check_values_image) < 2**16 else  + 2**16][0]

            template_max = [np.nanmax(check_values_template) if np.nanmax(check_values_template) < 2**16 else  2**16][0]
            
        else:
            
            image_max = np.nanmax(check_values_image)

            template_max = np.nanmax(np.nanmax(check_values_template))


        t_header = getheader(template)

        image_FWHM = autophot_input['fwhm']

        # template_FWHM = t_header['FWHM']s
        
        if autophot_input['use_zogy']:
            try:
            
                autophot_input['use_hotpants'] = False
                # raise NotImplementedError('Zogy not yet implemented, shwich to HOTPANTS')
                
                # Get filename for saving
                base = os.path.splitext(os.path.basename(file))[0]
                
                logger.info('Performing image subtraction using PyZOGY')
                
                PyZOGY_log = autophot_input['write_dir'] + base + '_ZOGY.txt'
                # original_stdout = sys.stdout # Save a reference to the original standard output
    
                   
                image_psf = os.path.join(autophot_input['write_dir'],'PSF_model_'+autophot_input['base']+'.fits')
    
                from glob import glob
                template_psf = glob(os.path.join(autophot_input['template_dir'],'PSF_model_*'))[0]
                
                logger.info('Using Image : %s' % file)
                logger.info('Using Image PSF: %s' % image_psf)
                logger.info('Using Template : %s' % template)
                logger.info('Using Template PSF: %s' % template_psf)
                
                logger.info('\nRunning Zogy...\n')
                
                # print(image_max,template_max)
                
                diff = run_subtraction(science_image = file,
                                       reference_image = template,
                                       science_psf = image_psf,
                                       reference_psf = template_psf,
                                       reference_mask = footprint,
                                       # science_mask = footprint,
                                   
                                        show = False,
                                       # sigma_cut = 3,
                                       normalization = "science",
                                       science_saturation = 10+image_max,
                                       reference_saturation = 10+template_max,
                                       n_stamps = 1,
                                       max_iterations = 10,
                                       
                                       use_pixels  = True
                                        # size_cut = True
                                        )
             
                hdu = fits.PrimaryHDU(diff[0])
                hdul = fits.HDUList([hdu])
                hdul.writeto(str(file.replace(fname_ext,'_subtraction'+fname_ext)),
                             overwrite = True,
                             output_verify = 'silentfix+ignore')
            except Exception as e:
                print('Pyzogy Failed [%s] - trying HOTPANTS' % e)
                use_hotpants = True
                
                

        if use_hotpants:
            
            logger.info('Performing image subtraction using HOTPANTS')

            # Get filename for saving
            base = os.path.splitext(os.path.basename(file))[0]

            # Location of executable for hotpants
            exe = autophot_input['hotpants_exe_loc']

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
            
            HOTPANTS_log = autophot_input['write_dir'] + base + '_HOTterPANTS.txt'
            
            # print(args, file=open(HOTPANTS_log, 'w'))

    
            with  open(HOTPANTS_log, 'w')  as FNULL:
                
                
                
                

                pro = subprocess.Popen(args,shell=True, stdout=FNULL, stderr=FNULL)
                print('ARGUMENTS:', args, file=FNULL)

                # Timeout
                pro.wait(autophot_input['hotpants_timeout'])

                try:
                    # Try to kill process to avoid memory errors / hanging process
                    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                    print('HOTPANTS PID killed')
                    print(args)
                except:
                    pass

            print('HOTPANTS finished: %ss' % round(time.time() - start) )
            
        # =============================================================================
        # Check that subtraction file has been created
        # =============================================================================
        if os.path.isfile(output_fpath):

            file_size = os.path.getsize(str(file.replace(fname_ext,'_subtraction'+fname_ext)))

            if file_size == 0:
                
                print('File was created but nothing written')

                return np.nan
            
            else:
                
                print('Subtraction saved as %s' % os.path.splitext(os.path.basename(file.replace(fname_ext,'_subtraction'+fname_ext)))[0])
                
                return output_fpath
            
        if not os.path.isfile(output_fpath):
            
            print('File was not created')
            
            return np.nan

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        
        try:
                # Try to kill process to avoid memory errors / hanging process
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            print('HOTPANTS PID killed')
        except:
            pass

        return np.nan