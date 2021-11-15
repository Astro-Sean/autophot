def remove_background(source,fwhm = 7, xc=None, yc=None,
                      remove_bkg_local = True, remove_bkg_surface = False,
                      remove_bkg_poly  = False, remove_bkg_poly_degree = 1,
                      bkg_level = 3
                      ):
    '''
    
    :param source: DESCRIPTION
    :type source: TYPE
    :param autophot_input: DESCRIPTION
    :type autophot_input: TYPE
    :param fwhm: DESCRIPTION, defaults to 7
    :type fwhm: TYPE, optional
    :param xc: DESCRIPTION, defaults to None
    :type xc: TYPE, optional
    :param yc: DESCRIPTION, defaults to None
    :type yc: TYPE, optional
    :param remove_bkg_local: DESCRIPTION, defaults to True
    :type remove_bkg_local: TYPE, optional
    :param remove_bkg_surface: DESCRIPTION, defaults to False
    :type remove_bkg_surface: TYPE, optional
    :param remove_bkg_poly: DESCRIPTION, defaults to False
    :type remove_bkg_poly: TYPE, optional
    :param remove_bkg_poly_degree: DESCRIPTION, defaults to 1
    :type remove_bkg_poly_degree: TYPE, optional
    :param bkg_level: DESCRIPTION, defaults to 3
    :type bkg_level: TYPE, optional
    :param : DESCRIPTION
    :type : TYPE
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
    from photutils import (Background2D,SExtractorBackground,BkgIDWInterpolator,BkgZoomInterpolator)
                           

    logger = logging.getLogger(__name__)

    source_size = 1.5*fwhm
    box = int(np.ceil(fwhm))
    filter_size = int(np.ceil(fwhm))

    if xc is None and yc is None:
        yc = int(source.shape[1]/2)
        xc = int(source.shape[0]/2)
        

    sigma_clip = SigmaClip(sigma=bkg_level)

    positions = [xc,yc]
    
    aperture = CircularAperture(positions,r=source_size)
    
    masks = aperture.to_mask(method='center')
    
    mask_array = masks.to_image(shape=source.shape).astype(bool)
 

    try:
        if remove_bkg_surface:

            
            background = Background2D(source,
                                        box_size = box,
                                        mask = mask_array,
                                        filter_size = filter_size,
                                        sigma_clip = sigma_clip,
                                        bkg_estimator = SExtractorBackground(sigma_clip=sigma_clip),
                                        interpolator= BkgZoomInterpolator(order=1),
                                        )


            surface = background.background
            
            surface_under_aperture = np.ma.array(surface, mask=1-mask_array)
            
            median_surface = np.ma.median(surface_under_aperture)

            source_background_free = source - surface
            
                     
            backgroundfree_image_outside_aperture = np.ma.array(source_background_free, mask=mask_array)
            
            _, _,noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                            cenfunc = np.nanmean,
                                            stdfunc = np.nanstd,
                                            sigma=bkg_level,)

        elif remove_bkg_poly and not remove_bkg_local:

            surface_function_init = models.Polynomial2D(degree=remove_bkg_poly_degree)

            fit_surface = fitting.LevMarLSQFitter()

            x = np.arange(0,source.shape[1])
            y = np.arange(0,source.shape[0])
            xx, yy = np.meshgrid(x,y)

            with warnings.catch_warnings():
                
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                surface_fit = fit_surface(surface_function_init, xx, yy, source)

            surface = surface_fit(xx,yy)
            
            surface_under_aperture = np.ma.array(surface, mask=1-mask_array)
            
            
            median_surface = np.ma.median(surface_under_aperture)
            
            source_background_free = source - surface
            
            backgroundfree_image_outside_aperture = np.ma.array(source_background_free, mask=mask_array)
            
            _, _,noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                             cenfunc = np.nanmedian,
                                             stdfunc = np.nanstd,
                                             sigma=bkg_level)
            
            
        elif remove_bkg_local:
            
            source[mask_array.astype(bool)] == np.nan

            _, background, _ = sigma_clipped_stats(source,
                                           mask = mask_array,
                                           cenfunc = np.nanmedian,
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
                                            sigma=bkg_level,)
            
            noise = np.nanstd(source_background_free)
            
    except Exception as e:
        logger.exception(e)
        
        
    noise = np.nanstd(backgroundfree_image_outside_aperture)
    return source_background_free, surface , median_surface, noise