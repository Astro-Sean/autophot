
def rm_bkg(source,
                 syntax,
                 xc=0,
                 yc=0):

    import logging
    import warnings
    import numpy as np
    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground,MADStdBackgroundRMS, BkgIDWInterpolator
    from autophot.packages.aperture import ap_phot
    from astropy.modeling import models, fitting
    from photutils import CircularAperture
    from photutils import SExtractorBackground

    logger = logging.getLogger(__name__)

    # try:
    #     source_size = syntax['image_radius']
    # except:
    source_size = 2*syntax['fwhm']


    try:
        if syntax['psf_bkg_surface'] and not syntax['psf_bkg_poly']:

            sigma_clip = SigmaClip(sigma=syntax['lim_SNR'])

            bkg_estimator = MADStdBackgroundRMS()


            positions = [source.shape[0]/2,source.shape[1]/2]
            aperture = CircularAperture(positions, r=source_size)
            masks = aperture.to_mask(method='center')

            mask_array = masks.to_image(shape=((source.shape[0], source.shape[1])))

            bkg = Background2D(source,
                               box_size = (3, 3),
                               mask = mask_array,
                               filter_size=(3, 3),
                               sigma_clip=sigma_clip,
                               bkg_estimator=SExtractorBackground(),
                               interpolator= BkgIDWInterpolator(),
                               # edge_method = 'pad'
                               )


            surface = bkg.background

            # bkg_median = np.nanmedian(bkg.background_median)

            source_bkg_free = source - surface

        elif syntax['psf_bkg_poly']:


            surface_function_init = models.Polynomial2D(degree=syntax['psf_bkg_poly_degree'])

            fit_surface = fitting.LevMarLSQFitter()

            x = np.arange(0,source.shape[0])
            y = np.arange(0,source.shape[0])
            xx,yy= np.meshgrid(x,y)

            positions = [source.shape[0]/2,source.shape[1]/2]
            aperture = CircularAperture(positions, r=source_size)
            masks = aperture.to_mask(method='center')

            mask_array = masks.to_image(shape=((source.shape[0], source.shape[1])))

            source[mask_array.astype(bool)] == np.nan


            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                surface_fit = fit_surface(surface_function_init, xx, yy, source)

            surface = surface_fit(xx,yy)

            # bkg_median = np.nanmedian(surface)
            source_bkg_free = source - surface


        elif syntax['psf_bkg_local']:

            pos = list(zip([xc],[yc]))

            ap,bkg = ap_phot(pos,
                             source,
                             radius = syntax['ap_size'] * syntax['fwhm'],
                             r_in   = syntax['r_in_size'] * syntax['fwhm'],
                             r_out  = syntax['r_out_size'] * syntax['fwhm'])

            surface = np.ones(source.shape) * bkg

            source_bkg_free = source - surface


    except Exception as e:
        logger.exception(e)

    return source_bkg_free,surface

