def rm_bkg(source,
           syntax,
           xc=0,
           yc=0):

    import logging
    import warnings
    import numpy as np
    from astropy.stats import SigmaClip
    
    from photutils import (Background2D,
                           SExtractorBackground
                           # MADStdBackgroundRMS,
                           # BkgIDWInterpolator
                           )
    # from photutils.background import MMMBackground
    # from photutils.background import BiweightLocationBackground
    # from photutils.background import StdBackgroundRMS
    # from photutils.segmentation import make_source_mask

    from astropy.modeling import models, fitting
    from photutils import CircularAperture
    # from photutils import SExtractorBackground
    
    
    # print('-->',source.shape)

    logger = logging.getLogger(__name__)

    source_size = 1.3*syntax['fwhm']
    box = int(np.floor(syntax['fwhm']))
    filter_size = int(np.ceil(syntax['fwhm']/2))
    
    # print('box:',box,'filter:',filter_size)
    
    
    if xc == 0 and yc ==0:
        yc = source.shape[0]/2
        xc = source.shape[1]/2
        
    # filter_size = 1


    sigma_clip = SigmaClip(sigma=syntax['bkg_level'])

    positions = [xc,yc]
    
    aperture = CircularAperture(positions,
                                r=source_size)
    
    masks = aperture.to_mask(method='center')
    
    mask_array = masks.to_image(shape=((source.shape[0], source.shape[1])))
    
    # mask_array = mask_array.astype(int)
    
    # mask_array = make_source_mask(source, 
    #                               nsigma=2,
    #                               npixels=5,
    #                               filter_fwhm=syntax['fwhm'])
    
    # print('\nshape:',source.shape)
    


    try:
        if syntax['remove_bkg_surface'] and not syntax['remove_bkg_poly']:

            

            

            bkg = Background2D(source,
                               box_size = box,
                               mask = mask_array,
                               filter_size = filter_size,
                               sigma_clip = sigma_clip,
                                bkg_estimator = SExtractorBackground(sigma_clip=sigma_clip),
                               # interpolator= BkgIDWInterpolator(),
                               edge_method = 'pad'
                               )


            surface = bkg.background

            # bkg_median = np.nanmedian(bkg.background_median)

            source_bkg_free = source - surface

        elif syntax['remove_bkg_poly']:


            surface_function_init = models.Polynomial2D(degree=syntax['remove_bkg_poly_degree'])

            fit_surface = fitting.LevMarLSQFitter()

            x = np.arange(0,source.shape[1])
            y = np.arange(0,source.shape[0])
            xx, yy = np.meshgrid(x,y)
            
         
            # print(mask_array.shape)

            # positions = [source.shape[1]/2,source.shape[0]/2]
            # aperture = CircularAperture(positions, r=source_size)
            # masks = aperture.to_mask(method='center')

            # mask_array = masks.to_image(shape=((source.shape[0], source.shape[1])))
            
            # mask_array = make_source_mask(source, 
            #                   nsigma=2,
            #                   npixels=5,
            #                   filter_fwhm=syntax['fwhm'])
            # 
            # print(mask_array)

            # source[mask_array.astype(bool)] == np.nan


            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                surface_fit = fit_surface(surface_function_init, xx, yy, source)

            surface = surface_fit(xx,yy)

            # bkg_median = np.nanmedian(surface)
            source_bkg_free = source - surface


        elif syntax['remove_bkg_local']:

            # pos = list(zip([xc],[yc]))

            # ap,bkg = ap_phot(pos,
            #                  source,
            #                  radius = syntax['ap_size'] * syntax['fwhm'],
            #                  r_in   = syntax['r_in_size'] * syntax['fwhm'],
            #                  r_out  = syntax['r_out_size'] * syntax['fwhm'])
            from astropy.stats import sigma_clipped_stats

            # positions = [source.shape[0]/2,source.shape[1]/2]
            # aperture = CircularAperture(positions, r=source_size)
            # masks = aperture.to_mask(method='center')

            # mask_array = masks.to_image(shape=((source.shape[0], source.shape[1])))

            source[mask_array.astype(bool)] == np.nan

            _, bkg,_ = sigma_clipped_stats(source,
                                           mask = mask_array,
                                           cenfunc = np.nanmedian,
                                           stdfunc = np.nanstd
                                           )

            surface = np.ones(source.shape) * bkg

            source_bkg_free = source - surface

        else:
            raise Exception('No Background removal process selected')


    except Exception as e:
        logger.exception(e)

    return source_bkg_free,surface