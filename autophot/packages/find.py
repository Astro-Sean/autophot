#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def create_circular_mask(h, w, center=None, radius=None):
    '''
    Create mask centered within a image with height *h* and width *w* centered on
    *center* with a radius *radius*
    
    :param h: height of image
    :type h: int
    :param w: width of image
    :type w: int
    :param center: pixel location of mask, if none, mask out center of image
    defaults to None
    :type center: tuple with x, y pixel position, optional
    :param radius: radius of mask in pixels, if none, use the smallest distance
    between the center and image walls defaults to None
    :type radius: float, optional
    :return: image with shape h,w with masked regions
    :rtype: 2D array with height *h* and width *w*

    '''
    import numpy as np

    if center is None:
        # use the middle of the image
        center = (int(w/2), int(h/2))
        
    if radius is None: 
        # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    
    
    return mask

    
def combine(dictionaries):
    '''
    Combine list of dictionaies 
    
    :param dictionaries: List of dictionaries
    :type dictionaries: Dict
    :return: Combined dictionary with values equal to a list of original dictionary values
    :rtype: dict

    '''
    combined_dict = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            combined_dict.setdefault(key, []).append(value)
    return combined_dict
    

def get_fwhm(image, wdir, base, threshold_value = 25, fwhm_guess = 5,
             bkg_level = 3, max_source_lim = 1000, min_source_lim = 2, 
             int_scale = 25, fudge_factor = 5, fine_fudge_factor = 0.1,
             source_max_iter = 50, sat_lvl = 65536, lim_threshold_value = 3, scale_multipler = 4,
             sigmaclip_FWHM_sigma = 3, 
             sigmaclip_median_sigma = 3, 
             isolate_sources_fwhm_sep = 5, init_iso_scale = 25, 
             pix_bound = 25, save_FWHM_plot = False, image_analysis = False,
             use_local_stars_for_FWHM = False, prepare_templates = False, 
             vary_moff_beta = False, default_moff_beta= 4.765,
             max_fit_fwhm = 30, fitting_method = 'least_square', 
             local_radius = 1000, mask_sources_XY_R = [], 
             remove_sat = True, use_moffat = True,
             target_name = None, target_x_pix = None, target_y_pix = None, 
             scale = None, use_catalog = None, sigma_lvl = None, fwhm = None
             ):
    '''
    
        Robust function to find FWHM in an image. 
    
    The Full Width Half Maximum (FWHM) of point sources in an image is determined
    by the astronomical seeing when the image was taken, as well as the telescope
    and instrument optics.
    
    AutoPHOT needs to adapt to the number of point sources in an images. A deep
    image with a large field of view (FoV) will have considerably more sources than
    a shallow image with a small FoV. Too few sources may lead to poorly sampled
    (or incorrect) values of the FWHM, while too many sources may indicate the
    detection threshold is too low (i.e. background noise is detected as a source)
    and needlessly increases the computation time.
    
    
    :param image: Image containing sources
    :type image: 2D array
    :param wdir: File Path location for where to save plots and tables
    :type wdir: str
    :param base: Name of file to distinctly label plots and tables
    :type base: str
    :param threshold_value: Initial threshold value for which to search for sources. The detection criteria is initial set to look for sources
    :math:`threshold\_value \times \sigma_{bkg}` where:math:`\sigma_{bkg}` is the standard deviation of the image. This value is updated during execution , defaults to 25
    :type threshold_value: float, optional
    :param fwhm_guess: initial guess for the FWHM, this is updated once any source are found, defaults to 5
    :type fwhm_guess: float, optional
    :param bkg_level: The number of standard deviations to use for both the lower and upper clipping limit when determining the background level,  defaults to 3
    :type bkg_level: float, optional
    :param max_source_lim: Maximum number of sources to search for. If more sources are found, increase threshold value, defaults to 1000
    :type max_source_lim: int, optional
    :param min_source_lim: Minimum amount of sources to search for. If less than this value are found, an error is raise, defaults to 2
    :type min_source_lim: int, optional
    :param int_scale: Initial size of the cutout to place around sources. This value is updated, defaults to 25
    :type int_scale: int, optional
    :param fudge_factor: Large step size when increasing/decreasing the:math:`threshold\_value`, defaults to 5
    :type fudge_factor: float, optional
    :param fine_fudge_factor: If the code runs into an issue when there is too large of a change in sources detected per step size, we assumed that we are now detecting background noise. In this case we switch to this value and change the :math:`threshold\_value`, by a small increment. This is also used if the :math:`threshold\_value` drops below :math:`5\sigma_bkg` defaults to 0.1
    :type fine_fudge_factor: float, optional
    :param source_max_iter: Backstop to inhibit the source detection algorithm to execute for too long. The code my take a long time if initial too many sources are found and you start of with too long of an initial :math:`threshold\_value`, or if it cannot find any sources about the background level defaults to 50
    :type source_max_iter: Init, optional
    :param sat_lvl: Counts level above which any detected source is deemed saturated and discarded, defaults to 65536
    :type sat_lvl: float, optional
    :param lim_theshold_value: If the threshold_value decreases below this value, use :math:`fine\_fudge\_factor`, defaults to 5
    :type lim_theshold_value: float, optional
    :param scale_multipler: Integer times the FWHM used when creating the new cutout size, defaults to 4
    :type scale_multipler: int, optional
    :param sigmaclip_FWHM_sigma: DESCRIPTION, defaults to 3
    :type sigmaclip_FWHM_sigma: TYPE, optional
    :param sigmaclip_median_sigma: DESCRIPTION, defaults to 3
    :type sigmaclip_median_sigma: TYPE, optional
    :param isolate_sources_fwhm_sep: Isolate detected sources by this amount times the FWHM, defaults to 5
    :type isolate_sources_fwhm_sep: float, optional
    :param init_iso_scale: Initial distance to isolated detected sources by, defaults to 25
    :type init_iso_scale: float, optional
    :param pix_bound: Ignore sources near the edge of the image. Value given in pixels, defaults to 25
    :type pix_bound: float, optional
    :param save_FWHM_plot: If True, plot a distribution of the FWHM values, defaults to False
    :type save_FWHM_plot: bool, optional
    :param image_analysis: If True, plot a distribution of the FWHM values across the image and save this information as a *csv* file, defaults to False
    :type image_analysis: bool, optional
    :param use_local_stars_for_FWHM: If True, use stars within *local_radius pixels to determine the FWHM. This is useful if there is a spread of FWHM values across the image, defaults to False
    :type use_local_stars_for_FWHM: bool, optional
    :param prepare_templates: IF True, preform FWHM measurements on template image, defaults to False
    :type prepare_templates: bool, optional
    :param fitting_method: Fitting method when measuring the FWHM, defaults to 'least_square'
    :type fitting_method: str, optional
    :param local_radius: Distance from target location to search for stars and find FWHM value, defaults to 1000
    :type local_radius: float, optional
    :param mask_sources_XY_R: List of tuples containing sources to be masked in formation [(x_pix,y_pix,radius)], defaults to None
    :type mask_sources_XY_R: List of Tuples, optional
    :param remove_sat: If True, remove sources that have a maximum brightness greater than *sat_lvl*, defaults to True
    :type remove_sat: bool, optional
    :param use_moffat: If True, use a moffat function for FWHM fitting defaults to
    :type use_moffat: bool, optional
    :param default_moff_beta: Default value for the Moffat function exponent, defaults to 4.765
    :type default_moff_beta: float, optional
    :param vary_moff_beta: If True, allowing the :math:`\beta` exponent in the Moffat function to vary. This may become unstable, defaults to False
    :type vary_moff_beta: bool, optional
    :param max_fit_fwhm: Maximum FWHM allowed when fitting analytical function, defaults to 30
    :type max_fit_fwhm: float, optional
    :param target_x_pix: X pixel coordinate of target. If given, exclude the general location of the target when fitting for the FWHM defaults to None
    :type target_x_pix: float, optional
    :param target_y_pix: Y pixel coordinate of target.If given, exclude the general location of the target when fitting for the FWHM, defaults to None
    :type target_y_pix: float, optional
    :param scale: If known, preset the cutout size to this value. Cutout size = (:math:`2\times scale`, :math:`2 \times scale`), defaults to None
    :type scale: int, optional
    :param use_catalog: If True, use a catalog containing the columns *x_pix* and *y_pix* instead of using source detection. This variable source correspond tothe filepath of the catalog *csv* file, defaults to None
    :type use_catalog: str, optional
    :return: Returns the image FWHM, a dataframe containing information on thefitted sources, the updated cutout scale and the :math:`image\_params` dictionary containing information on the best fitting analytical model
    :rtype: List of objects
    

    '''


    from astropy.stats import sigma_clipped_stats
 
    from photutils.detection import DAOStarFinder

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip
    import logging
    import os
    from lmfit import Model
    
    from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    from autophot.packages.functions import set_size,pix_dist,border_msg

    logger = logging.getLogger(__name__)
    
    border_msg('Finding Full Width Half Maximum')
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    
   # mean, median, std = sigma_clipped_stats(image,
   #                                      sigma=bkg_level,
   #                                      maxiters=3)
   
    if not (sigma_lvl is None):
        
        min_source_no  = 0
        max_source_no  = np.inf
        
    else:
        
        max_source_no = max_source_lim
        min_source_no = min_source_lim

    if sigma_lvl is None:

        int_fwhm = fwhm_guess
        
    else:
        
        threshold_value = sigma_lvl
        
        if not ( fwhm is None ):
            int_fwhm = fwhm

        else:
            int_fwhm = fwhm_guess

    logging.info('\nSearching for FWHM')

    if not (fwhm is None):
        int_scale = scale
        int_fwhm = fwhm

    if use_moffat:
        
        logging.info('Using Moffat Profile for fitting')

        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        
        logging.info('Using Gaussian Profile for fitting')

        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm

    if use_local_stars_for_FWHM and not prepare_templates and not (target_x_pix is None):

        mask = np.zeros(image.shape).astype(int)

        h, w = mask.shape

        mask_circular = create_circular_mask(h, w,
                                             center = (target_x_pix,target_y_pix),
                                             radius = local_radius )
        mask[~mask_circular] = 1

    else:
        
        mask = np.zeros(image.shape)

    if len(mask_sources_XY_R)>0 and not prepare_templates:

        h, w = mask.shape
        for X_mask,Y_mask,R_mask in mask_sources_XY_R:
            
            mask_circular = create_circular_mask(h, w,
                                             center = (X_mask,Y_mask),
                                             radius = R_mask )
            mask[mask_circular] = 1


    image_params = []
    isolated_sources = []

    try:
        
        image_mean, image_median, image_std = sigma_clipped_stats(image,
                                                sigma = bkg_level,
                                                maxiters = 3)
        
        if image_median > 60000 and remove_sat:
            
            logger.info('High background level [%d counts] - ignoring saturated  stars' % image_median)

            remove_sat = False

        if not (sigma_lvl is None):
            
            logger.debug('Image stats: Mean %.3f :: Median %.3f :: std %.3f' % (image_mean,image_median,image_std))


        # decrease
        m = 0

        # increase
        n = 0

        # backstop
        failsafe = 0

        decrease_increment = False
        
        # check if we can find a few more sources
        check = False
        check_len = -np.inf

        search_image = image.copy()

        iso_temp = []
        
 

        # from astropy.stats import SigmaClip
        # from photutils.background import Background2D, MedianBackground
        
        
        # sigma_clip_bkg = SigmaClip(sigma=3.)

        # bkg_estimator = MedianBackground()

        # bkg = Background2D(image, (50, 50),
        #                    # filter_size=(3, 3),
        #                    sigma_clip=sigma_clip_bkg,
        #                    bkg_estimator=bkg_estimator)

        # image_median =  bkg.background
        
        update_guess = False
        
        brightest = 25
        
        
        # =============================================================================
        #  Setup FWHM fitting model
        # =============================================================================

        # shape = int(2*int_scale),int(2*int_scale)
        
        x_pix = np.arange(0, 2 * int_scale)
        y_pix = np.arange(0, 2 * int_scale)
        
        xx, yy = np.meshgrid(x_pix, y_pix)
        
        
        
                            
        if use_moffat:
            def fwhm_fitting_residual(x,x0,y0,sky,A,alpha,beta):
                
                M = moffat_2d((xx,yy),x0,y0,sky,A,dict(alpha=alpha,beta=beta))
                  
                return  M
        else:
            def fwhm_fitting_residual(x,x0,y0,sky,A,sigma):
                
                G = gauss_2d((xx,yy),x0,y0,sky,A,dict(sigma=sigma))
                
                return G

   
        
        fwhm_fitting_model = Model(fwhm_fitting_residual)
        
        fwhm_fitting_model.set_param_hint('x0',
                                          value = int_scale,
                                          min   = 0,
                                          max   = 2*int_scale)
    
        fwhm_fitting_model.set_param_hint('y0',
                                          value = int_scale,
                                          min   = 0,
                                          max   = 2*int_scale)
  
        if use_moffat:
            fwhm_fitting_model.set_param_hint('alpha',
                                              value = 3,
                                              min = 0,
                                              max = 30)
            
            fwhm_fitting_model.set_param_hint('beta',
                                              value = default_moff_beta,
                                              min = 0,
                                              vary = vary_moff_beta  )

        else:
            fwhm_fitting_model.set_param_hint('sigma',
                                              value = 3,
                                              min = 0,
                                              max = gauss_fwhm2sigma(max_fit_fwhm) )
                                    
        
        
        if  (use_catalog is None):
            using_catalog_sources = False
        else:
            using_catalog_sources = True
            
        try:
            # Remove target by masking with area with that of median image value - just so it's not picked up
            if not (target_x_pix is None) and not (target_x_pix is None) and not (fwhm is None) and not prepare_templates:

                logger.info('Target location : (x,y) -> (%.3f,%.3f)' % (target_x_pix , target_y_pix))

                if abs(target_x_pix)>=search_image.shape[1] or abs(target_y_pix)>=search_image.shape[0]:
                    logger.warning('Target not located on images')
                    raise Exception ( ' *** EXITING - Target pixel coordinates outside of image [%s , %s] *** ' % (int(target_y_pix), int(target_y_pix)))

                else:
                    search_image[int(target_y_pix)-int_scale: int(target_y_pix) + int_scale,
                                 int(target_x_pix)-int_scale: int(target_x_pix) + int_scale] =  image_median * np.ones((int(2*int_scale),int(2*int_scale)))
        except:
            print('Target position not defined - ignoring for now')
            
        while True:
            try:
                
                # Use sources given in Catalog/Sequence stars
                if using_catalog_sources:
                    
                    sources = pd.read_csv(use_catalog,sep = ' ')
                   
                    sources = sources.rename(columns={'x_pix': 'xcentroid', 'y_pix': 'ycentroid'})

                
                    if len(sources) == 0:
                        using_catalog_sources = not using_catalog_sources
                        
                if not using_catalog_sources:
                    
                    if failsafe > source_max_iter:
                        logger.info(' Source detection gives up!')
                        break
                        # raise Exception ( ' *** EXITING - Cannot find enough sources in the image ***')
                    else:
                        failsafe +=1

                    # check if threshold value is still good
                    threshold_value_check = threshold_value + n - m

                    # if <=0 reverse previous drop and and fine_fudge factor
                    if threshold_value_check  < lim_threshold_value and not decrease_increment:
                        
                            logger.warning('Threshold value has gone below background limit [%d sigma] - increasing by smaller increment ' % lim_threshold_value)

                            # revert previous decrease
                            decrease_increment = True
                            
                            m = fudge_factor
                            
                    elif threshold_value_check  < lim_threshold_value and decrease_increment:
                        
                        raise Exception('FWHM detection failed - cannot find suitable threshold value ')
                        
                    else:
                        
                        threshold_value = round(threshold_value + n - m,3)

                    # if threshold goes negative use smaller fudge factor
                    if decrease_increment:
                        fudge_factor = fine_fudge_factor


                    daofind = DAOStarFinder(fwhm      = int_fwhm,
                                            threshold = threshold_value*image_std,
                                            sharplo   =  0.2,sharphi = 1.0,
                                            roundlo   = -1.0,roundhi = 1.0,
                                            exclude_border = True,
                                            brightest = brightest
                                            )
                    
                    sources = daofind(search_image - image_median,
                                      mask = mask.astype(bool))

                    if sources is None:
                        logger.warning('Sources == None at %.1f sigma - decresing threshold' % threshold_value)
                        m = fudge_factor
                        continue

                    sources = sources.to_pandas()
                    
                
                logger.info('\nNumber of sources before cleaning [ %.1f sigma ]: %d ' % (threshold_value,len(sources)))
            
                
                    
                if len(sources) == 0 and not using_catalog_sources:
                    logger.warning('No sources')
                    m = fudge_factor
                    continue
                    
                # TODO: Make sure this works  
                if check and len(sources) >= 10 and not using_catalog_sources:
                    pass
                
                elif check and len(sources) > check_len and not using_catalog_sources:
                    
                    print('More sources found - trying again')
                    
                    m = fudge_factor
                    check_len = len(sources)
                    check = False
                    
                    continue
                
                
                # Last ditch attempt to recover a few more sources
                elif len(sources) < 5 and not check and not using_catalog_sources:
                    logging.info('Sources found but attempting to go lower')
                    m = fudge_factor
                    check_len = len(sources)
                    check = True
                    continue
                
                    
                    
                
                
                if not update_guess:
                    
                    logging.info('Updating search FWHM value')
                    
                    from random import sample
                    new_fwhm_guess = []
                    
                    no_sources = 10 if len(sources.index.values)>10 else len(sources.index.values)
                    
          
                    
                    
                    for i in sample(list(sources.index.values), no_sources):
                        try:
                            # print('\rFitting source for FWHM: %d/%d'%(i+1,len(isolated_sources.index)),end = ' ',flush=True)
                            
                            idx = sources.index.values[i]
    
                            x0 = float(sources['xcentroid'].loc[[idx]])
                            y0 = float(sources['ycentroid'].loc[[idx]])
                            
        
                            close_up = search_image[int(y0 - int_scale): int(y0 + int_scale),
                                                    int(x0 - int_scale): int(x0 + int_scale)]
                            
                        
                            fwhm_fitting_model.set_param_hint('A',
                                                              value = np.nanmax(close_up),
                                                              min = 1e-3,
                                                              max = 1.5*np.nanmax(close_up))
                            
                            fwhm_fitting_model.set_param_hint('sky',
                                                              value = np.nanmedian(close_up)
                                                              )
                                                  
                            fwhm_fitting_pars = fwhm_fitting_model.make_params()
              
               
                            result = fwhm_fitting_model.fit(data = close_up,
                                                            params = fwhm_fitting_pars,
                                                            x = np.ones(close_up.shape),
                                                            method = fitting_method,
                                                            nan_policy = 'omit')
                            
                            if use_moffat:
                                 
                                 source_image_params = dict(alpha=result.params['alpha'].value,
                                                            beta=result.params['beta'].value)
                                 fwhm_fit = fitting_model_fwhm(source_image_params)
 
                            else:
                                 # TODO account for if fitting fails and fitting returns NONE
                                 source_image_params = dict(sigma=result.params['sigma'].value)
                                 
                                 fwhm_fit = fitting_model_fwhm(source_image_params)
                            
                            new_fwhm_guess.append(fwhm_fit)
                            brightest = None
                            m = 0
                            
                        except Exception as e:
                            print(e)
                            pass
                        
                    
                    update_guess = True
                    # print(new_fwhm_guess)
                    int_fwhm = np.nanmean(new_fwhm_guess)
                    print('Updated guess for FWHM: %.1f pixels ' % int_fwhm)
                    continue
                    

                if len(sources) > max_source_no and not using_catalog_sources:

                    logger.warning('Too many sources - increasing threshold')
                    
                    if n==0:
                        
                        threshold_value *=2

                    elif m != 0:
                        
                        decrease_increment = True
                        n = fine_fudge_factor
                        fudge_factor = fine_fudge_factor

                    else:
                        n = fudge_factor

                    continue

                elif len(sources) > 5000 and m !=0 and not using_catalog_sources:

                    logger.warning('Picking up noise - increasing threshold')
                    fudge_factor = fine_fudge_factor
                    n = fine_fudge_factor
                    m = 0
                    decrease_increment = True

                    continue

                elif len(sources) < min_source_no and not decrease_increment and not using_catalog_sources:

                    logger.warning('Too few sources - decreasing threshold')
                    m = fudge_factor

                    continue

                elif len(sources) == 0 and not using_catalog_sources:

                    logger.warning('No sources - decreasing threshold')
                    m = fudge_factor

                    continue
                
 

                with_boundary = len(sources)

                sources = sources[sources['xcentroid'] < image.shape[1] - pix_bound ]
                sources = sources[sources['xcentroid'] > pix_bound ]
                sources = sources[sources['ycentroid'] < image.shape[0] - pix_bound ]
                sources = sources[sources['ycentroid'] > pix_bound ]
                
                if with_boundary - len(sources) >0:

                    logger.info('Removed %d sources near boundary' % (with_boundary - len(sources)))

                #  Interested in these sources
                x = np.array(sources['xcentroid'])
                y = np.array(sources['ycentroid'])

                if len(sources) < min_source_no and not using_catalog_sources:

                    logger.warning('Less than min source after boundary removal')
                    m = fudge_factor

                    continue

                if sigma_lvl != None or len(sources) < 10:

                    isolated_sources = pd.DataFrame({'x_pix':x,'y_pix':y})
                    
                not_isolated = 0
                
                # TODO: No need for this loop - change to generator
                for idx in range(len(x)):
                    

                    try:

                        x0 = x[idx]
                        y0 = y[idx]
                
                        dist = np.sqrt((x0-np.array(x))**2+(y0-np.array(y))**2)

                        dist = dist[np.where(dist>0)]

                        if len(dist) == 0:
                            dist = [0]

                        if min(dist) <= init_iso_scale  and not using_catalog_sources:

                            not_isolated+=1

                        else:
                            
                            df = np.array((float(x0),float(y0)))

                            iso_temp.append(df)
                            
                    except Exception as e:
                        logger.exception(e)
                        pass

                logger.info('Removed %d crowded sources' % ( not_isolated))

                if len(iso_temp) < min_source_no:
                    logger.warning('Less than min source available after isolating sources')
                    m = fudge_factor
                    continue

                isolated_sources= pd.DataFrame(data = iso_temp)
                isolated_sources.columns = ['x_pix','y_pix']
                isolated_sources.reset_index()
                
                #  x,y recenter lists
                x_rc = []
                y_rc = []

                fwhm_list=[]
                fwhm_list_err = []
                medianlst=[]


                image_copy = image.copy()

                saturated_source=0
                broken_closeup = 0
                not_fitted = 0
                high_fwhm = 0

                if remove_sat:

                    try:
                        # Look for predefined saturation level
                        saturation_lvl = sat_lvl

                    except:

                        saturation_lvl = 2**16
                        sat_lvl = saturation_lvl

                for i in range(len(isolated_sources.index)):
                    print('\rFitting source for FWHM: %d/%d'%(i+1,len(isolated_sources.index)),end = ' ',flush=True)
                    
                    idx = isolated_sources.index.values[i]

                    try:

                        x0 = float(isolated_sources['x_pix'].loc[[idx]])
                        y0 = float(isolated_sources['y_pix'].loc[[idx]])
                        
                        close_up = image_copy[int(y0 - int_scale): int(y0 + int_scale),
                                              int(x0 - int_scale): int(x0 + int_scale)]
                        
                        # Incorrect image size
                        if close_up.shape != (int(2*int_scale),int(2*int_scale)):

                            fwhm_list.append(np.nan)
                            fwhm_list_err.append(np.nan)
                            x_rc.append(x0)
                            y_rc.append(y0)
                            medianlst.append(np.nan)
                            # logger.warning('wrong close-up size [x: %.3f :: y %.3f]' % (x0,y0))
                            broken_closeup+=1

                            continue

                        # Saturdated or nan values in close-up
                        if (np.nanmax(close_up)>= sat_lvl or np.isnan(np.max(close_up))) and remove_sat:

                            saturated_source +=1
                            # logger.warning('Saturated source [x: %.3f :: y %.3f]' % (x0,y0))
                            fwhm_list.append(np.nan)
                            fwhm_list_err.append(np.nan)
                            medianlst.append(np.nan)
                            x_rc.append(x0)
                            y_rc.append(y0)

                            continue


                        

                        try:
                            
                            fwhm_fitting_model.set_param_hint('A',
                                                              value = np.nanmax(close_up),
                                                              min = 1e-3,
                                                              max = 1.5*np.nanmax(close_up))
                            
                            fwhm_fitting_model.set_param_hint('sky',
                                                              value = np.nanmedian(close_up)
                                                              )
                                                  
                            fwhm_fitting_pars = fwhm_fitting_model.make_params()
              
               
                            result = fwhm_fitting_model.fit(data = close_up,
                                                             params = fwhm_fitting_pars,
                                                             x = np.ones(close_up.shape),
                                                             method = fitting_method,
                                                             nan_policy = 'omit')
                            
                            A = result.params['A'].value
                            x_fitted = result.params['x0'].value
                            y_fitted = result.params['y0'].value
                            bkg_approx = result.params['sky'].value
  
                            # TODO: Find better way to ignore low SNR catalog sources if it is used
                            # if using_catalog_sources and A<5*std:
                                 
                            #      fwhm_list.append(np.nan)
                            #      fwhm_list_err.append(np.nan)
                            #      medianlst.append(np.nan)
                            #      x_rc.append(x0)
                            #      y_rc.append(y0)

                            #      continue

                            if remove_sat:
                                 
                                 # If the amplitude of a source is beyond the saturation level - remove it
                                if A >= sat_lvl:

                                    saturated_source +=1

                                    # logger.warning('Saturated source [x: %.3f :: y %.3f]' % (x0,y0))
                                    fwhm_list.append(np.nan)
                                    fwhm_list_err.append(np.nan)
                                    medianlst.append(np.nan)
                                    x_rc.append(x0)
                                    y_rc.append(y0)

                                    continue

                            if use_moffat:
                                 
                                 source_image_params = dict(alpha=result.params['alpha'].value,
                                                            beta=result.params['beta'].value)

                                 # source_image_params_STD = dict(alpha=result.params['alpha'].stderr,
                                 #                                beta=result.params['beta'].stderr)
                                 
                                 fwhm_fit = fitting_model_fwhm(source_image_params)
                                 # TODO add in moffat error 
                                 fwhm_fit_err = np.nan

                            else:
                                 
                                 # TODO account for if fitting fails and fitting returns NONE
                                 source_image_params = dict(sigma=result.params['sigma'].value)
                                 
                                 fwhm_fit = fitting_model_fwhm(source_image_params)
                                 
                                 fwhm_fit_err = np.nan
                   

                            if fwhm_fit >= max_fit_fwhm-1:

                                    high_fwhm +=1

                                    # logger.warning('Saturated source [x: %.3f :: y %.3f]' % (x0,y0))
                                    fwhm_list.append(np.nan)
                                    fwhm_list_err.append(np.nan)
                                    medianlst.append(np.nan)
                                    x_rc.append(x0)
                                    y_rc.append(y0)

                                    continue
                             
                            corrected_x = x_fitted - int_scale + x0
                            corrected_y = y_fitted - int_scale + y0
                             
                            # Add details to lists
                            image_params.append(source_image_params)
                            fwhm_list.append(fwhm_fit)
                            fwhm_list_err.append(fwhm_fit_err)
                            medianlst.append(bkg_approx)
                            x_rc.append(corrected_x)
                            y_rc.append(corrected_y)
                             

                        except Exception as e:
                            
                            logger.exception(e)

                            fwhm_list.append(np.nan)
                            fwhm_list_err.append(np.nan)
                            medianlst.append(np.nan)
                            x_rc.append(x0)
                            y_rc.append(y0)

                            continue

                    except Exception as e:

                        logger.exception(e)

                        continue
                # print(' - Done')

                if saturated_source != 0:
                    logging.info('Removed %d saturated sources' %  saturated_source)
                    
                if not_fitted != 0:
                    logging.info('Removed %d not fitted sources' %  not_fitted)
                    
                if high_fwhm != 0:
                    logging.info('Removed %d with high fwhm [limit: %d [pixels]' %  (high_fwhm,max_fit_fwhm))

                if broken_closeup != 0:
                    logging.info('Removed %d sourcess with incorrect cutouts' %  broken_closeup)

                # Add these values to the dataframe
                isolated_sources['FWHM'] = fwhm_list
                isolated_sources['FWHM_err'] = fwhm_list_err
                isolated_sources['median'] = medianlst
                
                
                seperations = [pix_dist(i[0],isolated_sources['x_pix'],i[1],isolated_sources['y_pix']) for i in list(zip(isolated_sources['x_pix'],isolated_sources['y_pix']))]
                isolated_sources['min_seperation'] = list([np.nanmin(i[i>0]) for i in seperations ])
                isolated_sources.reset_index(inplace = True,drop = True)
                
                    
                if sigma_lvl is None and not using_catalog_sources:

                    if len(isolated_sources['FWHM'].values) == 0:
                        logger.info('No sigma values taken')
                        continue

                    if len(isolated_sources) < min_source_no:

                        logger.warning('Less than min source after sigma clipping: %d' % len(isolated_sources))
                        threshold_value += m

                        if n ==0:

                            decrease_increment = True
                            n = fine_fudge_factor
                            fudge_factor = fine_fudge_factor

                        else:

                            n = fudge_factor

                        m = 0

                        continue


                    if not using_catalog_sources:
                        
                        from astropy.stats import  mad_std
                        

                        FWHM_mask = sigma_clip(isolated_sources['FWHM'].values,
                                                sigma=sigmaclip_FWHM_sigma,
                                                masked = True,
                                                maxiters=10,
                                                cenfunc = np.nanmedian,
                                                stdfunc = mad_std)
                        
                        if np.sum(FWHM_mask.mask)== 0 or len(isolated_sources)<5 or  using_catalog_sources:
                            isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
                            
                            fwhm_array =  isolated_sources['FWHM'].values
                            
                        else:
                            
                            fwhm_array =  isolated_sources[~FWHM_mask.mask]['FWHM'].values
                            isolated_sources['include_fwhm'] = ~FWHM_mask.mask
                            # isolated_sources[FWHM_mask]['fwhm'] = np.nan
                            logger.info('Removed %d FWHM outliers' % (np.sum(FWHM_mask.mask)))
                    else:
                        isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
                            
                            
                    if  not  using_catalog_sources:

                        median_mask = sigma_clip(isolated_sources['median'].values,
                                                  sigma=sigmaclip_median_sigma,
                                                  masked = True,
                                                  maxiters=10,
                                                  cenfunc = np.nanmedian,
                                                  stdfunc = mad_std)
                        
                        if np.sum(median_mask) == 0 or np.sum(~median_mask.mask)<5 or using_catalog_sources:
                            isolated_sources['include_median'] = [True] * len(isolated_sources)
                            # fwhm_array =  isolated_sources['FWHM'].values
                            pass
                        
                        else:
                            # fwhm_array =  isolated_sources[~FWHM_mask.mask]['FWHM'].values
                            isolated_sources['include_median'] = ~median_mask.mask
                            # isolated_sources[FWHM_mask]['fwhm'] = np.nan
                            logger.info('Removed %d median outliers' % (np.sum(median_mask.mask)))
                    else:
                        isolated_sources['include_median'] = [True] * len(isolated_sources)
                            
                            
                    logger.info('Useable sources found [ %d sigma ]: %d' % (threshold_value,len(isolated_sources)))
                    
                    image_fwhm =  np.nanmean(fwhm_array)
                    
                    if len(isolated_sources)>5:
                    
                        too_close = isolated_sources['min_seperation']<=isolate_sources_fwhm_sep * image_fwhm
                       
                        isolated_sources = isolated_sources[~too_close]
                        
                        logger.info('Removes %d sources within minimum seperation [ %d pixel ]' % (np.sum(too_close),(isolate_sources_fwhm_sep * image_fwhm)))
                    
                    
                else:

                    image_fwhm = fwhm

                break

            except Exception as e:
                logger.exception(e)
                break

        

        #  End of loop
        image_params_combine = combine(image_params)
        image_params_out={}

        for key,val in image_params_combine.items():
            val = np.array(val)
            image_params_out[key] = np.nanmedian(val)
            image_params_out[key+'_err'] = np.nanstd(val)
        
        if not  using_catalog_sources:
            idx = (isolated_sources['include_median']) & (isolated_sources['include_fwhm']) 
            
            if np.sum(idx) == len(isolated_sources):
                # If this removes all sources - ignore this step
                idx = [True]*len(isolated_sources)
        else:
            idx = [True]*len(isolated_sources)
            
            
        image_fwhm = np.nanmean(isolated_sources['FWHM'].values[idx])
        image_fwhm_err = np.nanstd(isolated_sources['FWHM'].values[idx])

        if image_fwhm_err > 2 and not prepare_templates:
            
            logging.warning('\nLarge error on FWHM - returning plots for user diagnostic')
            save_FWHM_plot = True
            image_analysis = True
            # save_image_analysis = True
            
        # Update and set image cutout scale
        scale = int(np.ceil(scale_multipler * image_fwhm)) + 0.5

        if save_FWHM_plot:
            
            # Histogram of FWHM values

            from scipy.stats import norm
           
            plt.ioff()

            fig = plt.figure(figsize = set_size(250,1))

            ax1 = fig.add_subplot(111)

            # Fit a normal distribution to the data:
            mu, std = norm.fit(isolated_sources['FWHM'].dropna().values)

            # Plot the histogram.
            ax1.hist(isolated_sources['FWHM'].values, 
                     bins='auto', 
                     density=True,
                     color = 'gray',
                     label = 'FWHM Distribution',
                     alpha = 0.5)

            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            ax1.plot(x, p, linewidth=0.5,label = 'PDF',color = 'r')

            ax1.set_xlabel(r'Full Width Half Maximum [pixels]')
            ax1.set_ylabel('Probability Denisty')

            ax1.legend(loc = 'best',
                       frameon = False)

            ax1.axvline(image_fwhm,color = 'black',ls = '--',label = ' FWHM')

            figname = os.path.join(wdir,'fwhm_histogram_'+base+'.pdf')

            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )
            
            plt.close(fig)


        if image_analysis:

            import matplotlib as mpl
            from astropy.visualization import  ZScaleInterval
            from matplotlib.gridspec import  GridSpec

            vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image)


            ncols = 3
            nrows = 3

            heights = [1,1,0.75]
            widths = [1,1,0.75]

            plt.ioff()

            fig = plt.figure(figsize = set_size(500,aspect = 1))

            grid = GridSpec(nrows, ncols ,wspace=0., hspace=0.,
                            height_ratios=heights,
                            width_ratios = widths
                            )

            ax1   = fig.add_subplot(grid[0:2, 0:2])
            ax1_B = fig.add_subplot(grid[2, 0:2])
            ax1_R = fig.add_subplot(grid[0:2, 2])

            ax1.imshow(image,
                      vmin = vmin,
                      vmax = vmax,
                      interpolation = 'nearest',
                      origin = 'lower',
                      aspect = 'auto',
                      cmap = 'Greys')

            if use_local_stars_for_FWHM and not prepare_templates:
                local_radius_circle = plt.Circle( ( target_x_pix, target_y_pix ), local_radius,
                                                     color = 'red',
                                                     ls = '--',
                                                     label = 'Local Radius [%d px]' % local_radius,
                                                     fill=False)
                ax1.add_patch( local_radius_circle)
                
            if len(mask_sources_XY_R)!=0:

                for X_mask,Y_mask,R_mask in mask_sources_XY_R:
                    masked_radius_circle = plt.Circle( ( X_mask, Y_mask ), R_mask,
                                                 color = 'green',
                                                 ls = ':',
                                                 label = 'Masked Region',
                                                 fill=False)
                    ax1.add_patch(masked_radius_circle)
                    
            if  not prepare_templates:
                ax1.scatter([target_x_pix],[target_y_pix],
                           marker = 'H',
                           s = 25,
                           facecolor = 'None',
                           edgecolor = 'gold')
                    
            ax1.set_xlim(0,image.shape[1])
            ax1.set_ylim(0,image.shape[0])

            cmap = plt.cm.jet


            ticks=np.linspace(isolated_sources['FWHM'].values.min(),isolated_sources['FWHM'].values.max(),10)
            
            norm = mpl.colors.BoundaryNorm(ticks, cmap.N)

            ax1.scatter(isolated_sources['x_pix'].values,
                         isolated_sources['y_pix'].values,
                         cmap = cmap,
                         norm = norm,
                         marker = "o",
                         alpha = 0.5,
                         # s = 25,
                         c = isolated_sources['FWHM'].values)
            
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            
            ax1_R.scatter(isolated_sources['FWHM'].values,isolated_sources['y_pix'].values,
                          cmap=cmap,
                          norm = norm,
                          marker = "o",
                          alpha = 0.5,
                          c = isolated_sources['FWHM'].values,
                          zorder = 1)

            ax1_R.errorbar(isolated_sources['FWHM'].values,
                         isolated_sources['y_pix'].values,
                         xerr = isolated_sources['FWHM_err'].values,
                         fmt="none",
                         marker=None,
                         color = 'black',
                         capsize = 0.5,
                         zorder = 0)

            ax1_B.scatter(isolated_sources['x_pix'].values,isolated_sources['FWHM'].values,
                          cmap=cmap,
                          norm = norm,
                          marker = "o",
                          alpha = 0.5,
                          c = isolated_sources['FWHM'].values,
                          zorder = 1)

            ax1_B.errorbar(isolated_sources['x_pix'].values,
                           isolated_sources['FWHM'].values,
                           yerr = isolated_sources['FWHM_err'].values,
                           fmt="none",
                           marker=None,
                           color = 'black',
                           capsize = 0.5,
                           zorder = 0)


            ax1_R.yaxis.set_label_position("right")
            ax1_R.yaxis.tick_right()


            ax1_R.set_ylabel('Y pixel')
            ax1_R.set_xlabel('FWHM [pixels]')

            ax1_B.set_ylabel('FWHM [pixels]')
            ax1_B.set_xlabel('X pixel')


            ax1_R.set_ylim(0,image.shape[0])
            ax1_B.set_xlim(0,image.shape[1])

            figname = os.path.join(wdir,'image_analysis_'+base+'.pdf')
            
            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )
            
            plt.close(fig)

            
            # Save FWHM analayis to file

            isolated_sources.round(3).to_csv(os.path.join(wdir,'image_analysis_'+base+'.csv'))


        return image_fwhm,isolated_sources,scale,image_params_out


    except Exception as e:
        
        # Failsafe exception

        logger.exception(e)

        return np.nan,np.nan,np.nan,np.nan









