def remove_background(source,
                       autophot_input,
                       xc=None,
                       yc=None,
                       use_image_seg_mask = False):
    '''
    
    :param source: DESCRIPTION
    :type source: TYPE
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param xc: DESCRIPTION, defaults to 0
    :type xc: TYPE, optional
    :param yc: DESCRIPTION, defaults to 0
    :type yc: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    # import sys,os
    import logging
    import warnings
    import numpy as np
    
    from astropy.stats import SigmaClip
    from astropy.stats import sigma_clipped_stats
    
    from astropy.modeling import models, fitting
    
    from photutils import CircularAperture
    
    # import warnings

    
    from photutils import (Background2D,
                           SExtractorBackground,
                           BkgIDWInterpolator,
                           BkgZoomInterpolator)
                           


    logger = logging.getLogger(__name__)

    source_size = 1.5*autophot_input['fwhm']
    
    # aperture_radius = autophot_input['ap_size'] * autophot_input['fwhm']
    # aperture_area = np.pi * aperture_radius **2

    box = int(np.ceil(autophot_input['fwhm']))
    filter_size = int(np.ceil(autophot_input['fwhm']))

    if xc is None and yc is None:
        
        yc = int(source.shape[1]/2)
        xc = int(source.shape[0]/2)
        

    sigma_clip = SigmaClip(sigma=3)
    
    # from photutils.segmentation import make_source_mask
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     mask_array = make_source_mask(source, 
    #                                   nsigma=autophot_input['bkg_level'],
    #                                   npixels=int(np.ceil(autophot_input['fwhm'])**2),
    #                                   filter_fwhm=autophot_input['fwhm'])
    # print(mask_array.astype(bool))
    positions = [xc,yc]
    
    aperture = CircularAperture(positions,r=source_size)
    
    masks = aperture.to_mask(method='center')
    
    mask_array = masks.to_image(shape=source.shape).astype(bool)
 
   
    # mask_array = mask_array.astype(int)
    # if use_image_seg_mask:
       
    


    try:
        if autophot_input['remove_bkg_surface']:

            
            background = Background2D(source,
                                        box_size = box,
                                        mask = mask_array,
                                        filter_size = filter_size,
                                        sigma_clip = sigma_clip,
                                        bkg_estimator = SExtractorBackground(sigma_clip=sigma_clip),
                                        # interpolator= BkgZoomInterpolator(order=1),
                                        # edge_method = 'pad'
                                        )


            surface = background.background
            
            surface_under_aperture = np.ma.array(surface, mask=1-mask_array)
            
            median_surface = np.ma.median(surface_under_aperture)

            source_background_free = source - surface
            
                     
            backgroundfree_image_outside_aperture = np.ma.array(source_background_free, mask=mask_array)
            
            _, _,noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                            cenfunc = np.nanmean,
                                            stdfunc = np.nanstd,
                                            sigma=autophot_input['bkg_level'],)

        elif autophot_input['remove_bkg_poly'] and not autophot_input['remove_bkg_local']:

            raise Exception('Not using poly fit anymore')
            surface_function_init = models.Polynomial2D(degree=autophot_input['remove_bkg_poly_degree'])

            fit_surface = fitting.LevMarLSQFitter()

            x = np.arange(0,source.shape[1])
            y = np.arange(0,source.shape[0])
            xx, yy = np.meshgrid(x,y)

            with warnings.catch_warnings():
                
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                surface_fit = fit_surface(surface_function_init, xx, yy, source)

            surface = surface_fit(xx,yy)
            
            # median_surface = np.median(surface * mask_array)
            
            surface_under_aperture = np.ma.array(surface, mask=1-mask_array)
            
            
            median_surface = np.ma.median(surface_under_aperture)
            
            source_background_free = source - surface
            
                     
            backgroundfree_image_outside_aperture = np.ma.array(source_background_free, mask=mask_array)
            
            _, _,noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                            cenfunc = np.nanmean,
                                            stdfunc = np.nanstd,
                                            sigma=autophot_input['bkg_level'],)
            
            


        else:
            source[mask_array.astype(bool)] == np.nan

            _, background, _ = sigma_clipped_stats(source,
                                           mask = mask_array,
                                           cenfunc = np.median,
                                           stdfunc = np.nanstd
                                           )

            surface = np.ones(source.shape) * background
            
            surface_under_aperture = np.ma.array(surface, mask= 1 - mask_array)
            
            median_surface = np.ma.median(surface_under_aperture)
        
            

            source_background_free = source - surface
            
            backgroundfree_image_outside_aperture = np.ma.array(source_background_free, mask=mask_array)
            
            _, _,noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                            cenfunc = np.nanmedian,
                                            stdfunc = np.nanstd,
                                            sigma=autophot_input['bkg_level'],)
            
            noise = np.nanstd(source_background_free)
            
    except Exception as e:
        logger.exception(e)
        
        
    noise = np.nanstd(backgroundfree_image_outside_aperture)
    return source_background_free, surface , median_surface, noise