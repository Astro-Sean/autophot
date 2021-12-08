def remove_background(image,fwhm = 7, xc=None, yc=None,
                      remove_bkg_local = True, remove_bkg_surface = False,
                      remove_bkg_poly  = False, remove_bkg_poly_degree = 1,
                      bkg_level = 3
                      ):
    '''
    
    Find and remove approximate background from cutout image. This package enables
    AutoPHOT to detected
    and remove the underlying background from a closeup cutout of an image for
    later PSF fitting. 
    
    This package enables three different types of background measurements and
    subtraction.
    
    1. Background surface fitting:
    
    Using `Background2D
    
    <https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html>`_,
    We fit
    the background  using (sigma-clipped) statistics in each box of a grid that
    covers the input image
    to create a low-resolution, and possibly irregularly-gridded, background map.
    This map /grid is
    updated using the image fwhm. The center of the image (where the target is
    suspected to be) is
    masked up
    
    2. Local median subtraction:
    
    We perform `sigma clipped statistics
    
    <https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html>`_
    on the close up
    image, which the target location masked, masked to find the median value for
    the background. This
    is then converted to a flat 2D image which a value equal to this median value.
    This is by far the
    fastest method of background estimation, although it can be an inappropriate
    assumption for a
    complex background e.g. with background contamination from a host galaxy.
    
    
    3. Polynomial fitting: 
    
    We fit the background surface using a `2D Polynomial Surface
    
    <https://docs.astropy.org/en/stable/api/astropy.modeling.polynomial.Polynomial2D.html>`_
    of a given
    degree. This can be useful if there is a smoothly varying background, however
    the computation time
    is similar to the background surface fitting and in most cases it is better to
    use that method.
    
    
    :param image: 2D array with target location at the center.
    :type image: 2D array
    :param fwhm: Full Width Half Maximum (FWHM) of an image. This is used to update
    the surface grid
    and masked regions of the image, defaults to 7
    :type fwhm: float, optional
    :param xc: X pixel location of the image. If not given, the target location is
    assumed to be in the
    center of the image, defaults to None
    :type xc: float, optional
    :param yc: Y pixel location of the image. If not given, the target location is
    assumed to be in the
    center of the image, defaults to None
    :type yc: float, optional
    :param remove_bkg_local: If True, use the local median of the image and
    subtract this to produce a
    background free image, see above, defaults to True
    :type remove_bkg_local: Boolean, optional
    :param remove_bkg_surface: If True, use the background fitted surface of the
    image and subtract
    this to produce a background free image, see above, defaults to False
    :type remove_bkg_surface: Boolean, optional
    :param remove_bkg_poly: DESCRIPTION, defaults to False
    :type remove_bkg_poly: If True, use the background polynomial surface of the
    image and subtract
    this to produce a background free image, see above, optional
    :param remove_bkg_poly_degree: If remove_bkg_poly is True, this is the degree
    of the polynomial
    fitted to the image, 1 = flat surface, 2 = 2nd order polynomial etc, defaults
    to 1
    :type remove_bkg_poly_degree: TYPE, optional
    :param bkg_level: The number of standard deviations, below which is assumed to
    be due to the
    background noise distribution, defaults to 3
    :type bkg_level: float, optional
    :return: Return the original image with the pre-selected background removed,
    the fitted background
    itself, the median value of this background at the position of the target
    (performed using crude
    aperture photometry) and the noise of this background surface (taken as the
    standard deviation)
    :rtype: Tuple
    
    '''

    import warnings
    import numpy as np
    
    from astropy.stats import SigmaClip
    from astropy.stats import sigma_clipped_stats
    from astropy.modeling import models, fitting
    
    from photutils import CircularAperture
    from photutils import Background2D,SExtractorBackground,BkgZoomInterpolator
    
    try:
        # assumed size (radius) of the source  = area to be masked
        source_size = 1.5*fwhm
        
        # box and filter size for  surface fitting
        box = int(np.ceil(fwhm))
        filter_size = int(np.ceil(fwhm))
    
        
        if xc is None and yc is None:
            yc = int(image.shape[1]/2)
            xc = int(image.shape[0]/2)
            
    
        sigma_clip = SigmaClip(sigma=bkg_level,cenfunc = 'median',stdfunc = 'std')
    
        positions = [xc,yc]
        
        # mask array excludign center location
        aperture = CircularAperture(positions,r=source_size)
        masks = aperture.to_mask(method='center')
        mask_array = masks.to_image(shape=image.shape).astype(bool)
     
        if remove_bkg_surface:
            
            # Fit surface background
            background = Background2D(image,
                                        box_size = box,
                                        mask = mask_array,
                                        filter_size = filter_size,
                                        sigma_clip = sigma_clip,
                                        bkg_estimator = SExtractorBackground(sigma_clip=sigma_clip),
                                        interpolator= BkgZoomInterpolator(order=1),
                                        )
    
            # fitted surface
            surface = background.background
    
            # background free image
            image_background_free = image - surface
            
        elif remove_bkg_poly and not remove_bkg_local:
            
            # fitted 2D polynomial to surface
            surface_function_init = models.Polynomial2D(degree=remove_bkg_poly_degree)
    
            fit_surface = fitting.LevMarLSQFitter()
    
            x = np.arange(0,image.shape[1])
            y = np.arange(0,image.shape[0])
            xx, yy = np.meshgrid(x,y)
    
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                
                surface_fit = fit_surface(surface_function_init, xx, yy, image)
    
            surface = surface_fit(xx,yy)
    
            
            image_background_free = image - surface
            
        elif remove_bkg_local:
            
            image[mask_array.astype(bool)] == np.nan
    
            _, background_value, _ = sigma_clipped_stats(image,
                                                   mask = mask_array,
                                                   cenfunc = np.nanmedian,
                                                   stdfunc = np.nanstd
                                               )
    
            surface = np.ones(image.shape) * background_value     
            
            image_background_free = image - surface
            

        # mask out target area in surface
        backgroundfree_image_outside_aperture = np.ma.array(image_background_free,
                                                            mask=mask_array)
        
        # get noise/STD of the surface, excluding the location of the target
        _, _, noise = sigma_clipped_stats(backgroundfree_image_outside_aperture,
                                         cenfunc = np.nanmedian,
                                         stdfunc = np.nanstd,
                                         sigma=bkg_level)
        
        # Find the median value of the surface inside the target location
        surface_inside_aperture = np.ma.array(surface,mask=abs(1-mask_array))
        
        _, median_surface, _= sigma_clipped_stats(surface_inside_aperture,
                                                  cenfunc = np.nanmedian,
                                                  stdfunc = np.nanstd,
                                                  sigma = bkg_level)
    
        
        return image_background_free, surface , median_surface, noise
    
    except Exception as e:
        
        print('Could not fit background: %s' % e)
        
        return np.nan, np.nan, np.nan, np.nan