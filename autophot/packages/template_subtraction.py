
def prepare_templates(fpath,syntax,
                      get_fwhm = True,
                      clean_cosmic = True,
                      redo_wcs = False,
                      build_psf = True):
    '''
    
    :param fpath: DESCRIPTION
    :type fpath: TYPE
    :param syntax: DESCRIPTION
    :type syntax: TYPE
    :param get_fwhm: DESCRIPTION, defaults to True
    :type get_fwhm: TYPE, optional
    :param clean_cosmic: DESCRIPTION, defaults to True
    :type clean_cosmic: TYPE, optional
    :param redo_wcs: DESCRIPTION, defaults to False
    :type redo_wcs: TYPE, optional
    :param build_psf: DESCRIPTION, defaults to True
    :type build_psf: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    import os
    from astropy.io import fits
    from autophot.packages.functions import getheader,getimage

    syntax['write_dir'] = os.path.dirname(fpath)
    syntax['base'] = ''.join(os.path.basename(fpath).split('.')[:-1])

    if syntax['base'].startswith('PSF_model'):
        print('ignoring PSF_MODEL in file: %s' % syntax['base'])
        return

    dirpath = os.path.dirname(fpath)
    image = getimage(fpath)
    header = getheader(fpath)

    try:
        for i in ['EXPTIME','EXP_TIME','TIME-INT']:
            if i in header:
                exp_time = header[i]
                break
        raise Exception
    except:
        exp_time = 1


    syntax['gain'] = 1

    if isinstance(exp_time, str):
       exp_time = exp_time.split('/')
       exp_time = float(exp_time[0])

    syntax['exp_time'] = exp_time

    if clean_cosmic:

        if 'CRAY_RM'  not in header:

            from autophot.packages.call_crayremoval import run_astroscrappy

            image,syntax = run_astroscrappy( fits.PrimaryHDU(image),syntax)


    if get_fwhm or build_psf:

        import autophot.packages.find as find

        mean_fwhm,df,syntax = find.fwhm(image,syntax)

        header['FWHM'] = mean_fwhm

        df.to_csv(os.path.join(dirpath,'calib_template.csv'),index = False)

    if redo_wcs:
        print('Not yet implemented')


    if build_psf:
        from autophot.packages import psf

        df = find.phot(image,syntax,df,mean_fwhm)

        r_table,fwhm_fit,psf_heights,syntax = psf.build_r_table(image,df,syntax,mean_fwhm)

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


def subtract(file,template,syntax,psf = None, mask_border = True, pix_bound = None ):
    '''
    
    Function for performing template subtractioon Astronmical images using AutoPhot
    
    
    :param file: DESCRIPTION
    :type file: TYPE
    :param template: DESCRIPTION
    :type template: TYPE
    :param syntax: DESCRIPTION
    :type syntax: TYPE
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

    logger = logging.getLogger(__name__)

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
        
        footprint = np.ones(file_image.shape).astype(bool)
        
        if mask_border:
            
            if not (pix_bound is None):
                pix_bound = syntax['pix_bound']
                
            footprint[syntax['pix_bound']: - syntax['pix_bound'],
                      syntax['pix_bound']: - syntax['pix_bound']] = False
        

        footprint = footprint +  (np.isnan(template_image)) | (np.isnan(file_image)) | (np.isinf(file_image)) | (np.isinf(template_image))
        
        
            

        hdu = fits.PrimaryHDU(footprint.astype(int))
        hdul = fits.HDUList([hdu])

        footprint_loc = os.path.join(syntax['write_dir'],'footprint_'+syntax['base']+fname_ext)
        # print(footprint_loc)
        hdul.writeto(footprint_loc,
                      overwrite=True,
                      output_verify = 'silentfix+ignore')

        check_values_image = file_image[~footprint]

        check_values_template = template_image[~footprint]

        if syntax['remove_sat']:

            image_max = [np.nanmax(check_values_image) if np.nanmax(check_values_image) < 2**16 else -500 + 2**16][0]

            template_max = [np.nanmax(check_values_template) if np.nanmax(check_values_template) < 2**16 else  2**16][0]
        else:
            image_max = np.nanmax(check_values_image)

            template_max = np.nanmax(np.nanmax(check_values_template))


        t_header = getheader(template)

        image_FWHM = syntax['fwhm']

        template_FWHM = t_header['FWHM']
        
        if syntax['use_zogy']:
            # syntax['use_hotpants'] = True
            # raise NotImplementedError('Zogy not yet implemented, shwich to HOTPANTS')
            
            logger.info('Performing image subtraction using PyZOGY')

            from PyZOGY.subtract import run_subtraction

            # template_dir = os.path.basename(template)
            # template_base = ''.join(template_dir.split('.')[:-1])

            image_psf = os.path.join(syntax['write_dir'],'PSF_model_'+syntax['base']+'.fits')

            from glob import glob
            template_psf = glob(os.path.join(syntax['template_dir'],'PSF_model_*'))[0]
            
            diff = run_subtraction(file, template, image_psf, template_psf,
                                    show = False,
                                    sigma_cut = 3,
                                    normalization = "science",
                                    science_saturation = syntax['sat_lvl'],
                                    # science_mask = footprint_loc,
                                    # reference_mask = footprint_loc
                                    )

            hdu = fits.PrimaryHDU(diff[0])
            hdul = fits.HDUList([hdu])
            hdul.writeto(str(file.replace(fname_ext,'_subtraction'+fname_ext)),
                         overwrite = True,
                         output_verify = 'silentfix+ignore')

        if syntax['use_hotpants']:
            
            logger.info('Performing image subtraction using HOTPANTS')

            # Get filename for saving
            base = os.path.splitext(os.path.basename(file))[0]

            # Location of executable for hotpants
            exe = syntax['hotpants_exe_loc']

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
                    # Image lower values
                            ('-il',     str(np.nanmin(check_values_image))),
                    # Template lower values
                            ('-tl',     str(np.nanmin(check_values_template))),
                    # Template upper
                            ('-tu',     str(template_max)),
                    # Image upper
                            ('-iu',     str(image_max)),
                    # Image mask
                            ('-imi',    str(footprint_loc)),
                    # Template mask
                            # ('-tmi',    str(footprint_loc)),
                    # Image gain
                            # ('-ig',     str(syntax['gain'])),
                    # Template gain
                    #         ('-tg',     str(t_header['gain'])),
                    # Normalise to image[i]
                            ('-n',      'i'),
                    # Kernal size
                            ('-r',      str(syntax['fwhm'])),
                    # Background order fitting
                            # ('-bgo' ,   str(2)),

                            ]

            args= [str(exe)]

            if smooth_template:
                
                print('Adjusting fitting to smooth template')
                fwhm_match = np.sqrt(image_FWHM**2 + template_FWHM**2)
                sigma_match = fwhm_match/(2 * np.sqrt(2 * np.log(2)))

                add_flag = ('-ng', '3 6 %.3f 4 %.3f 2 %.3f' % (0.5*sigma_match, sigma_match, 2*sigma_match))

                include_args= include_args + [add_flag]

            
            for i in include_args:
                args[0] += ' ' + i[0] + ' ' + i[1]
            # =============================================================================
            # Call subprocess using executable location and option prasers
            # =============================================================================

            start = time.time()
            
            HOTPANTS_log = syntax['write_dir'] + base + '_HOTterPANTS.txt'
            
            # print(args, file=open(HOTPANTS_log, 'w'))

    
            with  open(HOTPANTS_log, 'w')  as FNULL:
                
                
                
                

                pro = subprocess.Popen(args,shell=True, stdout=FNULL, stderr=FNULL)
                print('ARGUMENTS:', args, file=FNULL)

                # Timeout
                pro.wait(syntax['hotpants_timeout'])

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