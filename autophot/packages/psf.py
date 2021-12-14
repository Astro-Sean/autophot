#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def build_r_table(base_image, selected_sources, fwhm, exp_time, image_params,
                  fpath, GAIN=1, rdnoise=0, use_moffat = True, 
                  fitting_radius = 1.3, regrid_size = 10, use_PSF_starlist = False,
                  use_local_stars_for_PSF = False, prepare_templates = False,
                  scale = 25, ap_size = 1.7, r_in_size = 2, r_out_size = 3,
                  local_radius = 1500, bkg_level = 3, psf_source_no = 10,
                  min_psf_source_no =3, construction_SNR =25, remove_bkg_local = True,
                  remove_bkg_surface = False, remove_bkg_poly = False,
                  remove_bkg_poly_degree = 1,
                  fitting_method = 'least_sqaure', 
                  save_PSF_stars = False, plot_PSF_model_residual = False, 
                  save_PSF_models_fits = False
                  ):
    r'''
    
       Build the PSF model from bright, well isolated sources in the field.  AutoPHOT
    uses "well-behaved" sources to build the PSF model which will then be used to
    measure the amplitude of sources in the field. These sources must be have a high S/N,
    isolated from their neighbours and have a relatively smooth background. This is done by
    building a compound model comprised of an analytical component (such as Gaussian or
    Moffat) along with a numerical residual table obtained during the fitting process.  
    Bright isolated sources are located and fitted with an analytical function . The best
    fit location is noted and the analytic model is subtracted to leave a residual
    image . The residual image is resampled onto a finer pixel grid and shifted . The
    compound (analytic and residual) PSF model is then normalized to unity. This process is
    repeated for several (typically ~10 sources) bright isolated sources, to create an
    average residual image. The final step is to resample the average residual image back
    to to the original pixel scale.

    :param base_image: 2D array  containing sources
    :type base_image: 2D array
    :param selected_sources: DataFrame object containing *xcentroid* and *ycentroid* columns given the XY pixel location of a source. Dataframe may also containing *include_fwhm* and *include_median* which should contain either *True* or *False* on whether to include a given source based in it's FWHM and/or background median respectively.
    :type selected_sources: DataFrame
    :param fwhm: Full Width Half Maximum (FWHM) of an image
    :type fwhm: float
    :param exp_time: Exposure time ins seconds of an image. This is used to calculate the S/N of a potential source.
    :type exp_time: float
    :param image_params: Dictionary containing analytical model params. If a moffat is used, this dictionary should containing *alpha* and *beta* and their respective values, else if a gaussian is used, this dictionary should include *sigma* and its value.
    :type image_params: dict
    :param fpath: Filepath of *FITS* image used. 
    :type fpath: str
    :param GAIN: GAIN on CCD in :math:`e^{-} /  ADU` , defaults to 1.
    :type GAIN: float, optional
    :param rdnoise: Read noise of CCD in :math:`e^{-} /  pixel`, defaults to 0.
    :type rdnoise: float, optional
    :param use_moffat: If True, use a moffat function for FWHM fitting defaults to True
    :type use_moffat: bool, optional
    :param fitting_radius: zoomed region around location of best fit to focus fittinƒimage_params:g. This allows for the fitting to be concentrated on high S/N areas and not fit the low S/N wings of the PSF, defaults to 1.3
    :type fitting_radius: float, optional
    :param regrid_size: When expanding to larger pseudo-resolution, what zoom factor to use, defaults to 10
    :type regrid_size: int, optional
    :param use_PSF_starlist: If True, *selected_soures* has been provided by the USer and this variable is required to ignore several source cleaning steps, defaults to False
    :type use_PSF_starlist: bool, optional
    :param use_local_stars_for_PSF: if True, use sources within a radius given by *local_radius*, defaults to False
    :type use_local_stars_for_PSF: bool, optional
    :param prepare_templates: IF True, build PSF model on template image, defaults to False
    :type prepare_templates: bool, optional
    :param scale: Size of image cutout. Image cutout size = (:math:`2 \times scale`, :math:`2 \times scale`), defaults to 25
    :type scale: int, optional
    :param ap_size: Multiple of FWHM to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param r_in_size: Multiple of FWHM to be used as inner radius of background annulus, defaults to 1.9
    :type r_in_size: float, optional
    :param r_out_size: Multiple of FWHM to be used as outer radius of background annulus, defaults to 2.2
    :type r_out_size: float, optional
    :param local_radius: Radius to look for PSF sources around target position, defaults to 1500
    :type local_radius: int, optional
    :param bkg_level: The number of standard deviations, below which is assumed to be due to the background noise distribution, defaults to 3
    :type bkg_level: float, optional
    :param psf_source_no: Number of sources used to build PSF model, defaults to 10
    :type psf_source_no: int, optional
    :param min_psf_source_no: Minimum number  of sources used to build PSF model with. If less than this number of sources are available, PSF photometry is not used, defaults to 3
    :type min_psf_source_no: int, optional
    :param construction_SNR: Minimum S/N ratio require for a source to be used in the PSF model, defaults to 25
    :type construction_SNR: int, optional
    :param remove_bkg_local: If True, use the local median of the image and subtract this to produce a background free image, see above, defaults to True
    :type remove_bkg_local: Boolean, optional
    :param remove_bkg_surface: If True, use the background fitted surface of the image and subtract this to produce a background free image, see above, defaults to False
    :type remove_bkg_surface: Boolean, optional
    :type remove_bkg_poly: If True, use the background polynomial surface of the image and subtract this to produce a background free image, see above, optional
    :param remove_bkg_poly_degree: If remove_bkg_poly is True, this is the degree of the polynomial fitted to the image, 1 = flat surface, 2 = 2nd order polynomial etc, defaults to 1
    :param fitting_method: Fitting method when fitting the PSF model, defaults to 'least_square'
    :type fitting_method: str, optional
    :param save_PSF_stars: If True, save the information on the stars used to build the PSF model, defaults to False
    :type save_PSF_stars: bool, optional
    :param save_PSF_models_fits: If True, save a *FITS* image of the PSF model, normalised to unity, defaults to False
    :type save_PSF_models_fits: bool, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import lmfit
    import logging

    from photutils import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    
    from autophot.packages.functions import scale_roll
    from autophot.packages.functions import rebin
    from autophot.packages.functions import pix_dist
    from autophot.packages.functions import SNR,border_msg
    from autophot.packages.aperture  import measure_aperture_photometry
    from autophot.packages.functions import gauss_2d,gauss_sigma2fwhm
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    from autophot.packages.background import remove_background
    
    if not use_PSF_starlist:
        border_msg('Building PSF model using stars in the field')
    else:
        border_msg('Building PSF model using user defined sources')
        
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    
    base = os.path.basename(fpath)
    write_dir = os.path.dirname(fpath)
    base = os.path.splitext(base)[0]

    logger = logging.getLogger(__name__)

    image = base_image.copy()

    # Only fit to a small image with radius ~the fwhm
    fitting_radius = int(np.ceil( fitting_radius* fwhm))
    x_slice = np.arange(0,2*fitting_radius)
    xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

    # for matchinf each source residual image with will regrid the image for shifting later
    regrid_size = int(regrid_size)
    

    residual_table = []

    if regrid_size % 2 > 0:
        logger.info('regrid size must be even adding 1')
        regrid_size += 1

    if use_moffat:
        
        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm
        
        
    
    # FWHM/sigma fits
    fwhm_fit = []

    # what sources will be used
    construction_sources = []
    
    # Remove sources that don't have FWHM values
    if not use_PSF_starlist:
        selected_sources = selected_sources[~np.isnan(selected_sources['FWHM'])]
    
    if 'include_fwhm' in selected_sources and not use_PSF_starlist:
        if np.sum(selected_sources['include_fwhm'])>10:
            selected_sources = selected_sources[selected_sources['include_fwhm']]
        
    if 'include_median' in selected_sources and not use_PSF_starlist:
        if np.sum(selected_sources['include_median'])>10:
            selected_sources = selected_sources[selected_sources['include_median']]
        
    if use_local_stars_for_PSF and not prepare_templates and not use_PSF_starlist:

        # Use local stars given by 'use_acrmin' parameter
        selected_sources_test = selected_sources[selected_sources['dist'] <= local_radius]

        selected_sources  = selected_sources_test
        
    
        
    if not use_PSF_starlist:
        
        high_counts_idx = [i for i in selected_sources.counts_ap.sort_values(ascending = False).index]
        max_sources_used  = psf_source_no
        
    else:
        
        max_sources_used = len(selected_sources.index)
        high_counts_idx = selected_sources.index
        
    sources_used = 0
    n = 0
    failsafe = 0
    psf_mag = []
    sources_dict = {}
    
    
    try:
        while sources_used <= psf_source_no-1:

            if failsafe>25:
                logger.info('PSF - Failed to build psf')
                residual_table=None
                fwhm_fit = fwhm

            if n >= len(high_counts_idx):
                
                if sources_used  >= min_psf_source_no:
                    logger.info('Using worst case scenario number of sources')
                    break
                
                logger.info('\nRan out of sources for PSF model')
                residual_table=None
                fwhm_fit = fwhm
                break
            
            try:

                idx = high_counts_idx[n]

                n+=1

                # Inital guess at where psf source is is
                psf_image = image[int(selected_sources.y_pix[idx]-scale): int(selected_sources.y_pix[idx] + scale),
                                  int(selected_sources.x_pix[idx]-scale): int(selected_sources.x_pix[idx] + scale)]
                
                if not use_PSF_starlist:
                    
                    psf_fwhm_fitted = selected_sources.FWHM[idx]
                    
                else:
                    
                    psf_fwhm_fitted = np.nan
                    
                
                mean, median, std = sigma_clipped_stats(psf_image,
                                                        sigma = bkg_level,
                                                        maxiters = 10)
                
                daofind = DAOStarFinder(fwhm      = fwhm,
                                        threshold = 5 * std,
                                        sharplo   =  0.2,sharphi = 1.0,
                                        roundlo   = -1.0,roundhi = 1.0
                                        )

                sources = daofind(psf_image - median)
                
                if sources is None:
                    
                    logger.info('Cannot detect any point source - skipping')
                        
                    continue
                
                else:
                    
                    sources = sources.to_pandas()
                
                
                if len(sources)>1:
                    
                    # More than one source found! Might be detecting the same source - check if all points are withint one fwhm
                    dist = [max(pix_dist(x,sources['xcentroid'].values,y,sources['ycentroid'].values)) for x, y in zip(sources['xcentroid'], sources['ycentroid'])]


                    if all(x >  2*fwhm for x in dist) and not use_PSF_starlist:

                        logger.info('Found faint soures near PSF star at %d sigma - skipping '%bkg_level)
                        
                        continue
 
                if len(psf_image) == 0:
                    logger.info('PSF image ERROR')
                    continue
                
                if np.min(psf_image) == np.nan:
                    logger.info('PSF image ERROR (nan in image)')
                    continue

                psf_image = image[int(selected_sources.y_pix[idx]-scale): int(selected_sources.y_pix[idx]+scale),
                                  int(selected_sources.x_pix[idx]-scale): int(selected_sources.x_pix[idx]+scale)]
                
                psf_image_bkg_free, bkg_surface, background, noise = remove_background(psf_image,
                                                                                      remove_bkg_local = remove_bkg_local, 
                                                                                      remove_bkg_surface = remove_bkg_surface,
                                                                                      remove_bkg_poly   = remove_bkg_poly,
                                                                                      remove_bkg_poly_degree = remove_bkg_poly_degree,
                                                                                      bkg_level = bkg_level)
                                                                                      

                x = np.arange(0,2*scale)
                xx,yy= np.meshgrid(x,x)

                pars = lmfit.Parameters()
                pars.add('A',value = 0.75*np.nanmax(psf_image_bkg_free),
                         min=0)
                
                pars.add('x0',value = psf_image_bkg_free.shape[1]/2,
                         min = 0,
                         max =psf_image_bkg_free.shape[1] )
                
                pars.add('y0',value = psf_image_bkg_free.shape[0]/2,
                         min = 0,
                         max =psf_image_bkg_free.shape[0])

                if use_moffat:
                    pars.add('alpha',value = image_params['alpha'],
                             min = 0,
                             vary =  False )
                    pars.add('beta',value = image_params['beta'],
                             min = 0,
                             vary = False  )

                else:
                    pars.add('sigma', value = image_params['sigma'],
                             min = 0,
                             vary = False)
                             

                if use_moffat:
                    def residual(p):
                        p = p.valuesdict()
                        return (psf_image_bkg_free - moffat_2d((xx,yy),p['x0'],p['y0'],0,p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(psf_image_bkg_free.shape)).flatten()
                else:
                    def residual(p):
                      p = p.valuesdict()
                      return (psf_image_bkg_free - gauss_2d((xx,yy),p['x0'],p['y0'],0,p['A'],dict(sigma=p['sigma'])).reshape(psf_image_bkg_free.shape)).flatten()

                    

                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
                    
                    mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)
                    
                    result = mini.minimize(method = fitting_method)
                    
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                # global pixel coorindates base on best gaussian fit
                xc_global = xc - psf_image_bkg_free.shape[1]/2 + int(selected_sources.x_pix[idx])
                yc_global = yc - psf_image_bkg_free.shape[0]/2 + int(selected_sources.y_pix[idx])

                # recenter image absed on location of best fit x and y
                psf_image = image[int(yc_global-scale): int(yc_global + scale),
                                  int(xc_global-scale): int(xc_global + scale)]

                psf_image_bkg_free,_,bkg_median,noise = remove_background(psf_image,
                                                                          xc=None, yc=None,
                                                                          remove_bkg_local = remove_bkg_local, 
                                                                          remove_bkg_surface = remove_bkg_surface,
                                                                          remove_bkg_poly   = remove_bkg_poly,
                                                                          remove_bkg_poly_degree =   remove_bkg_poly_degree,
                                                                          bkg_level =bkg_level)
                                                                        

                psf_image_slice = psf_image_bkg_free[int(psf_image_bkg_free.shape[1]/2 - fitting_radius):int(psf_image_bkg_free.shape[1]/2 + fitting_radius) ,
                                                     int(psf_image_bkg_free.shape[0]/2 - fitting_radius):int(psf_image_bkg_free.shape[0]/2 + fitting_radius) ]

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmean(psf_image_slice),
                         min = 1e-6,
                         max = np.nanmax(psf_image_slice)*1.5 )
                
                pars.add('x0',value = psf_image_slice.shape[1]/2,
                         min = 1,
                         max = psf_image_slice.shape[1])
                
                pars.add('y0',value = psf_image_slice.shape[0]/2,
                         min = 1,
                         max = psf_image_slice.shape[0] )

                if use_moffat:

                    pars.add('alpha',value = image_params['alpha'],
                             min = 0,
                             vary =  False)

                    pars.add('beta',value = image_params['beta'],
                             min = 0,
                             vary = False  )

                else:

                    pars.add('sigma', value = image_params['sigma'],
                             min = 0,
                             vary =  False)
                             

                if use_moffat:
                    def residual(p):
                        p = p.valuesdict()
                        return abs(psf_image_slice  - moffat_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(psf_image_slice .shape)).flatten()
                else:
                    def residual(p):
                        p = p.valuesdict()
                        return abs(psf_image_slice - gauss_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(sigma=p['sigma'])).reshape(psf_image_slice.shape)).flatten()

                               
                import warnings
                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
                    
                    mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)
                    
                    result = mini.minimize(method = fitting_method)

                positions  = list(zip([xc_global],[yc_global]))

                psf_counts,psf_counts_error,psf_maxpixels,psf_bkg,psf_bkg_std = measure_aperture_photometry(positions,
                                                                 image,
                                                                 ap_size = ap_size    * fwhm,
                                                                 r_in   = r_in_size  * fwhm,
                                                                 r_out  = r_out_size * fwhm)
                
                PSF_flux = psf_counts/exp_time
                PSF_bkg_flux = psf_bkg/exp_time
                # psf_noise_flux = psf_bkg_std/exp_time

                if use_moffat:

                    PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:

                    PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))

                PSF_SNR = SNR(flux_star = PSF_flux ,
                              flux_sky = PSF_bkg_flux ,
                              exp_t = exp_time,
                              radius = ap_size*fwhm ,
                              G  = GAIN,
                              RN =  rdnoise,
                              DC = 0 )

                
                if np.isnan(PSF_SNR) or np.isnan(PSF_FWHM):
                    logger.debug('PSF Contruction source fitting error')
                    continue

                if PSF_SNR < construction_SNR and exp_time > 1:
                    logger.info('PSF constuction source too low: %s' % int(PSF_SNR))
                    logger.info('\nRan out of PSF sources above SNR=%d' % construction_SNR )
                    break
                else:

                    # print('\rPSF source %d / %d :: SNR: %d' % (int(PSF_SNR)),end = '')
                    pass

                # print(result.params)
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                H = result.params['A'].value
                # H_err = result.params['A'].stderr
                
                psf_mag.append(PSF_flux)

                chi2 = result.chisqr

                xc_correction =  xc - fitting_radius + scale
                yc_correction =  yc - fitting_radius + scale

                if use_moffat:

                    residual = psf_image_bkg_free - moffat_2d((xx,yy),xc_correction,yc_correction,
                                                              0,H,
                                                              dict(alpha=result.params['alpha'],beta=result.params['beta'])).reshape(psf_image_bkg_free.shape)
                    PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                    
                else:
                    residual = psf_image_bkg_free - gauss_2d((xx,yy),xc_correction,yc_correction,
                                                             0,H,
                                                             dict(sigma=result.params['sigma'])).reshape(psf_image_bkg_free.shape)
                    PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))
                    


                residual = residual / H
                
                # residual_counts_before = np.sum(residual)

                residual_regrid = np.repeat(np.repeat(residual, regrid_size, axis=0), regrid_size, axis=1)

                x_roll = scale_roll(fitting_radius,xc,regrid_size)
                y_roll = scale_roll(fitting_radius,yc,regrid_size)

                residual_roll = np.roll(np.roll(residual_regrid,y_roll,axis=0),x_roll,axis = 1)

                # residual_table += residual_roll
                residual_table.append(np.array(residual_roll))
                
                # residual_counts_after = np.sum(rebin(residual_roll, (int(2*scale),int(2*scale))))
                
        
                
                sources_used +=1
                

                sources_dict['PSF_%d'%sources_used] = {}

                # Save details about each source used to make the PSF
                sources_dict['PSF_%d'%sources_used]['x_pix']  = xc_global
                
                sources_dict['PSF_%d'%sources_used]['y_pix']  = yc_global
                sources_dict['PSF_%d'%sources_used]['H_psf'] = float(H)
                sources_dict['PSF_%d'%sources_used]['SNR']  = PSF_SNR

                sources_dict['PSF_%d'%sources_used]['fwhm'] = psf_fwhm_fitted
                sources_dict['PSF_%d'%sources_used]['chi2'] = chi2
                sources_dict['PSF_%d'%sources_used]['x_best'] = xc_correction
                sources_dict['PSF_%d'%sources_used]['y_best'] = yc_correction

                sources_dict['PSF_%d'%sources_used]['close_up'] = psf_image_bkg_free
                sources_dict['PSF_%d'%sources_used]['residual'] = residual
                sources_dict['PSF_%d'%sources_used]['regrid'] = residual_regrid
                sources_dict['PSF_%d'%sources_used]['roll'] = residual_roll
                sources_dict['PSF_%d'%sources_used]['x_roll'] = x_roll
                sources_dict['PSF_%d'%sources_used]['y_roll'] = y_roll

                logger.info('\rResidual table updated: %d / %d ' % (sources_used,max_sources_used) )
                logger.info(' - SNR: %d :: FWHM: %.3f :: FWHM fitted %.3f' % (PSF_SNR,PSF_FWHM,psf_fwhm_fitted))

                

                fwhm_fit.append(PSF_FWHM)

            except Exception as e:
                
                logger.exception(e)

                logger.error('** Fitting error - trying another source**')
                failsafe+=1
                n+=1

                continue

        if sources_used < min_psf_source_no:
            logger.info('BUILDING PSF: Not enough useable sources found')
            return None,None,construction_sources.append([np.nan]*5)
        
        # Get the mean of the residual tavles
        # residual_table = np.nanmean(np.dstack(residual_table),axis = -1)
        # print(residual_table)
        residual_table = sum(residual_table)/sources_used

        # regrid residual table to psf size
        r_table  = rebin(residual_table, (int(2*scale),int(2*scale)))
        
        
        construction_sources = pd.DataFrame.from_dict(sources_dict, orient='index',
                                                      columns=['x_pix','y_pix',
                                                               'H_psf',
                                                               'H_psf_err',
                                                               'fwhm'])
        construction_sources.reset_index(inplace = True)
        
        
        if save_PSF_stars:
            construction_sources.to_csv( os.path.join(write_dir,'PSF_stars_'+base+'.csv'))
            

        # if plot_PSF_model_residual:
        #     # Plot figure of PSF shift to make sure model is made correctly
        #     from autophot.packages.create_plots import plot_PSF_model_steps
        #     plot_PSF_model_steps(sources_dict,autophot_input,image)

        logger.info('\nPSF built using %d sources\n'  % sources_used)

        if save_PSF_models_fits:

            from astropy.io import fits
            import os

            if use_moffat:
                
                

                PSF_model_array = r_table+ moffat_2d((xx,yy),r_table.shape[1]/2,r_table.shape[0]/2,
                                                          0,1,
                                                          dict(alpha=image_params['alpha'],
                                                               beta=image_params['beta'])).reshape(r_table.shape)

            else:
                PSF_model_array = r_table + gauss_2d((xx,yy),r_table.shape[1]/2,r_table.shape[0]/2,
                                                         0,1,
                                                         dict(sigma=image_params['sigma'])).reshape(r_table.shape)

            hdu = fits.PrimaryHDU(PSF_model_array/np.sum(PSF_model_array))
            hdul = fits.HDUList([hdu])
            psf_model_savepath = os.path.join(write_dir,'PSF_model_'+base+'.fits')
            hdul.writeto(psf_model_savepath,
                         overwrite = True)

            print('PSF model saved as: %s' % psf_model_savepath)

    except Exception as e:
        logger.exception('BUILDING PSF: ',e)
        raise Exception

    return r_table,fwhm_fit,construction_sources




# =============================================================================
# Fitting of PSF
# =============================================================================
def PSF_MODEL(xc, yc, sky, H, r_table, fwhm,image_params,use_moffat = True,
              fitting_radius = 1.3,regrid_size =10 ,
              slice_scale = None,pad_shape = None):
    r'''
    
    Function that returns a "fitted-able" PSF model. The PSF model has the general format off:
    
    .. math::
       PSF(x,y,A) = G/M(x,y,A) + R(x,y,A)
       
    
    where *G/M* is our base analytical function of a fixed Full Width Half Maximum and *R* is the residual table, with the same shape as the analytical function 
    
    :param xc: Best fitted X position of the PSF model
    :type xc: float
    :param yc: Best fitted Y position of the PSF model
    :type yc: float
    :param sky: Sky background offset
    :type sky: float
    :param H: Best fitted amplitude of PSF model
    :type H: float
    :param r_table: Resiudal table, normailised to unity that is the same shape as the base analytical function
    :type r_table: 2D array
    :param fwhm: FWHM value used in the analytical function
    :type fwhm: float
    :param image_params: Dictionary containing analytical model params. If a moffat is used, this dictionary should containing *alpha* and *beta* and their respective values, else if a gaussian is used, this dictionary should include *sigma* and its value.
    :type image_params: dict
    :param use_moffat: If True, use a moffat fcuntion as the analytical function, else use a gaussian, defaults to True
    :type use_moffat: bool, optional
    :param fitting_radius: Multiple of FWHM to use a zoomed in closeup of image, defaults to 1.3
    :type fitting_radius: float, optional
    :param regrid_size: Zoom scale of increased pesudo-resoloution grid, defaults to 10
    :type regrid_size: int, optional
    :param slice_scale: If the returne PSF model is required to be of a center shape, set *slice\_scale* be half the height/width of the new shape e.g. PSF shape = (:math:`2 \\times slice\_scale`,:math:`2 \\times slice\_scale`), defaults to None
    :type slice_scale: int, optional
    :param pad_shape: If the output PSF shape needs to be larger than in the input gridsize of the residual table set *pad\_shape* be half the height/width of the new shape e.g. PSF shape = (:math:`2 \\times pad\_shape[0]`,:math:`2 \\times pad\_shape[1]`). Does not have to be a sqaure shape, defaults to None
    :type pad_shape: tuple of ints, optional
    :return: Returns a PSF model of a desirced shape
    :rtype: 2D array

    '''

    import numpy as np
    from autophot.packages.functions import gauss_2d,moffat_2d
    from autophot.packages.functions import scale_roll,rebin
    
    
    fitting_radius = int(np.ceil(fitting_radius* fwhm))

    try:

        psf_shape = r_table.shape

        if not (pad_shape is None) and pad_shape != psf_shape:
            
            #  Need to make PSF fitting bigger
            top =    int((pad_shape[0] - r_table.shape[0])/2)
            bottom = int((pad_shape[0] - r_table.shape[0])/2)
            left =   int((pad_shape[1] - r_table.shape[1])/2)
            right =  int((pad_shape[1] - r_table.shape[1])/2)
            
            # print(top,bottom,left,right)

            psf_shape = pad_shape
            
            # print(psf_shape)

            r_table = np.pad(r_table, [(top, bottom), (left, right)], mode='constant', constant_values=0)

        x_rebin = np.arange(0,psf_shape[1])
        y_rebin = np.arange(0,psf_shape[0])

        xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)

        if use_moffat:
            core = moffat_2d((xx_rebin,yy_rebin),xc,yc,sky,H,image_params).reshape(psf_shape)

        else:
            core = gauss_2d((xx_rebin,yy_rebin),xc,yc,sky,H,image_params).reshape(psf_shape)
        
        # Blow up residual table to larger size
        residual_rebinned = np.repeat(np.repeat(r_table, regrid_size, axis=0), regrid_size, axis=1)

        # scale roll = where you wana go,where you are
        x_roll = scale_roll(xc,r_table.shape[1]/2,regrid_size)
        y_roll = scale_roll(yc,r_table.shape[0]/2,regrid_size)
        
        # Roll in y direction then in x direction
        residual_roll = np.roll(np.roll(residual_rebinned,y_roll,axis=0),x_roll,axis = 1)

        # rebin and scale to high to PSF (fitted by analytical funcrion)
        residual = H * rebin(residual_roll,psf_shape)

        # add it all together
        psf =  sky  + core + residual
        
        if np.isnan(np.min(psf)):
            print('PSF model not fitted - nan in image')
            
            return np.zeros(psf_shape)

        psf[np.isnan(psf)] = 0

        if not (slice_scale is None):
            # retrun part of the PSF model given by the slive scale focused on the image center
            psf = psf[int ( 0.5 * r_table.shape[1] - slice_scale): int(0.5*r_table.shape[1] + slice_scale),
                      int ( 0.5 * r_table.shape[0] - slice_scale): int(0.5*r_table.shape[0] + slice_scale)]

    except Exception as e:
        print('PSF model error: %s' % e)
        # logger.exception(e)
        psf = np.nan

    return psf
    

# =============================================================================
# Fit the PSF model
# =============================================================================
def fit(image, sources, residual_table, fwhm, fpath, fitting_radius = 1.3, 
        regrid_size = 10, bkg_level = 3,
        sat_lvl = 65536, use_moffat = True, image_params = None, 
        fitting_method = 'least_sqaure',  save_plot = False,
        show_plot = False, remove_background_val = True, hold_pos = False,
        return_fwhm = False, return_subtraction_image = False,
        no_print = True, return_closeup = False, remove_bkg_local = True, 
        remove_bkg_surface = False, remove_bkg_poly = False,
        remove_bkg_poly_degree = 1, plot_PSF_residuals = False,
        ):
    r'''
        
    Function to fit a given Point Spread Function (PSF) model to a point source located in an image.
    
    :param image: 2D array containing point sources
    :type image: 2D array
    :param sources: Dataframe containg *x_pix* and *y_pix* columns corrospsondong the the XY pixel location in an image
    :type sources: Dataframe
    :param residual_table: Residual image normalised to unity.
    :type residual_table: 2D array
    :param fwhm: Full Width Half Maximum of image
    :type fwhm: float
    :param fpath: Filepath of image. This is used to save plots and figures.
    :type fpath: str
    :param fitting_radius: Multiple of FWHM to use a zoomed in closeup of image, defaults to 1.3
    :type fitting_radius: float, optional
    :param regrid_size: Zoom scale of increased pesudo-resoloution grid, defaults to 10
    :type regrid_size: int, optional
    :param bkg_level: The number of standard deviations, below which is assumed to be due to the background noise distribution, defaults to 3
    :type bkg_level: float, optional
    :param sat_lvl: Counts level above which any detected source is deemed saturated and discarded, defaults to 65536
    :type sat_lvl: float, optional
    :param use_moffat: If True, use a moffat function for FWHM fitting, else use a Gaussian
    :type use_moffat: bool, optional
    :param image_params: Dictionary containing analytical model params. If a moffat is used, this dictionary should containing *alpha* and *beta* and their respective values, else if a gaussian is used, this dictionary should include *sigma* and its value.
    :type image_params: dict
    :param fitting_method: Fitting method when fitting the PSF model, defaults to 'least_square'
    :type fitting_method: str, optional
    :param save_plot: If True, save a plot showing the source, it's fitted PSF and subtraction, defaults to False
    :type save_plot: bool, optional
    :param hold_pos: If True, don't let the PSF model adjust its position, only it's amplitude. This is equivalent to force photometry, defaults to False
    :type hold_pos: bool, optional
    :param return_fwhm: If True, fit the FWHM of the source using the base analytical model and include it in the output dataframe, defaults to False
    :type return_fwhm: bool, optional
    :param no_print: If True, do not display the progress of the PSF fitting, defaults to True
    :type no_print: bool, optional
    :param return_closeup: If True, retrun the close up image used to fit the PSF model, defaults to False
    :type return_closeup: bool, optional
    :param remove_bkg_local: If True, use the local median of the image and subtract this to produce a background free image, see above, defaults to True
    :type remove_bkg_local: Boolean, optional
    :param remove_bkg_surface: If True, use the background fitted surface of the image and subtract this to produce a background free image, see above, defaults to False
    :type remove_bkg_surface: Boolean, optional
    :type remove_bkg_poly: If True, use the background polynomial surface of the image and subtract this to produce a background free image, see above, optional
    :param remove_bkg_poly_degree: If remove_bkg_poly is True, this is the degree of the polynomial fitted to the image, 1 = flat surface, 2 = 2nd order polynomial etc, defaults to 1
    :param plot_PSF_residuals: If True, plot the residual images from the PSF fitting and subtraction and save them to a directory in *file\_path* called *psf\_subtractions*, defaults to False
    :type plot_PSF_residuals: Bool, optional
    :return: Return a dataframe containing information in the PSF fittings
    :rtype: Dataframe
    '''
    
    
    
    import numpy as np
    import pandas as pd
    import pathlib
    import lmfit
    import logging
    
    
    # Model used to fit PSF
    from lmfit import Model

    import matplotlib.pyplot as plt

    from autophot.packages.functions import gauss_2d,moffat_2d,moffat_fwhm,gauss_sigma2fwhm
    from autophot.packages.functions import set_size,order_shift,border_msg
    from matplotlib.gridspec import  GridSpec
    
    from autophot.packages.background import remove_background
    
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    
    if not no_print:
        border_msg('Fitting PSF to sources in the image')
    
     
    base = os.path.basename(fpath)
    write_dir = os.path.dirname(fpath)
    base = os.path.splitext(base)[0]

    logger = logging.getLogger(__name__)
    
    
    fitting_radius = int(fitting_radius * fwhm)

    psf_params = []
    
    scale = int(residual_table.shape[0]/2)
    # print(scale)
    
    if residual_table.shape[0]%2 != 0:
        scale+=0.5
    # print(scale)

    xx,yy= np.meshgrid(np.arange(0,image.shape[1]),np.arange(0,image.shape[0]))
    
    

    lower_x_bound = scale
    lower_y_bound = scale
    upper_x_bound = scale
    upper_y_bound = scale

    if return_subtraction_image:

        from astropy.visualization import  ZScaleInterval

        vmin,vmax = (ZScaleInterval(nsamples = 300)).get_limits(image)
   
    # if return_subtraction_image:
    #     image_before = image.copy()
        
        
    # How is the PSF model allowed to move around
    if hold_pos:
        
        dx_vary = False
        dy_vary = False
        
        dx = 0.5
        dy = 0.5
        
    else:
        
        dx_vary = True
        dy_vary = True
        
        dx = 4*fwhm
        dy = 4*fwhm
        

    
    slice_scale = fitting_radius
    pad_shape = None
        
                
    def psf_residual(x,x0,y0,A):
        
    
        res =  PSF_MODEL(x0,y0,0,A,
                            residual_table,
                            fwhm,
                            image_params,
                            use_moffat = use_moffat,
                            fitting_radius = fitting_radius,
                            regrid_size = regrid_size,
                            slice_scale = slice_scale,
                            pad_shape = pad_shape)

                                
        return res.flatten()
    
    
    psf_residual_model = Model(psf_residual)
    
    psf_residual_model.set_param_hint('x0',
                                      vary = dx_vary,
                                      value = scale,
                                      min   = scale - dx,
                                      max   = scale + dx)
    
    # 0.5*residual_table.shape[1]+dx

    psf_residual_model.set_param_hint('y0',
                                      vary = dy_vary,
                                      value = scale,
                                      min   = scale - dy,
                                      max   = scale + dy)

    x_slice = np.arange(0,2*fitting_radius)
    
    xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

    
    for n  in range(len(sources.index)):
        
        bkg_median = np.nan
        H = np.nan
        H_psf_err = np.nan
        x_fitted = np.nan
        y_fitted = np.nan
        chi2 = np.nan
        redchi2 = np.nan
        noise = np.nan
        max_pixel = np.nan
        
        # if not return_fwhm and not no_print:
        if not no_print:
            print('\rFitting PSF to source: %d / %d ' % (n+1,len(sources)), end = '')

        try:

            idx = list(sources.index)[n]
         
            xc_global = sources.x_pix[idx]
            yc_global = sources.y_pix[idx]
            
            source_base = image[int(yc_global-lower_y_bound): int(yc_global + upper_y_bound),
                                int(xc_global-lower_x_bound): int(xc_global + upper_x_bound)]
            
            if source_base is None or len(source_base) == 0 :
                continue

            
            xc = source_base.shape[1]/2
            yc = source_base.shape[0]/2
            


            try:

                source_bkg_free, bkg_surface, bkg_median, noise = remove_background(source_base,
                                                                                   remove_bkg_local = remove_bkg_local, 
                                                                                   remove_bkg_surface = remove_bkg_surface,
                                                                                   remove_bkg_poly   = remove_bkg_poly,
                                                                                   remove_bkg_poly_degree = remove_bkg_poly_degree,
                                                                                   bkg_level = bkg_level
                                                                                   )
                                                                                    
                
            except Exception as e:
                
                print('cannot fit background - %s' % e)

                psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H,H_psf_err,max_pixel,chi2,redchi2))

                logger.exception(e)
                
                continue
                
            # if return_fwhm:
            #     print('bkg_median: %6.f' % bkg_median)
            
            

            source = source_bkg_free[int(0.5*source_bkg_free.shape[1] - fitting_radius):int(0.5*source_bkg_free.shape[1] + fitting_radius) ,
                                     int(0.5*source_bkg_free.shape[0] - fitting_radius):int(0.5*source_bkg_free.shape[0] + fitting_radius) ]
            
            
            
            
            if source.shape != (int(2*fitting_radius),int(2*fitting_radius)) or np.sum(np.isnan(source)) == len(source):
            
                psf_params.append((idx,x_fitted,y_fitted,bkg_median,noise,H,H_psf_err,max_pixel,chi2,redchi2))

                continue



            # Go ahead and fit a PSF
            try:

                # Update params with amplitude in cutout
                psf_residual_model.set_param_hint('A',
                                                  value = 0.75 * np.nanmax(source),
                                                  min = 1e-6,
                                                  # max = 2*np.nanmax(source)
                                                  )
                
            
                psf_pars = psf_residual_model.make_params()
   

                import warnings
                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
        
                    
                    result = psf_residual_model.fit(data = source,
                                                    params = psf_pars,
                                                    x = np.ones(source.shape),
                                                    method = fitting_method,
                                                    nan_policy = 'omit',
                                                    # weights = np.sqrt(abs(source))
                                                    )
                    

                xc = result.params['x0'].value

                yc = result.params['y0'].value
                
  
                            
                H_psf = result.params['A'].value
                H_psf_err = result.params['A'].stderr

                if H_psf_err is None:
                    H_psf_err = np.nan
                    
                
                max_pixel = np.nanmax(source)

                chi2 = result.chisqr
                redchi2 = result.redchi

                x_fitted = xc - residual_table.shape[1]/2 + xc_global
                y_fitted = yc - residual_table.shape[0]/2 + yc_global
                
                x_fitted_shape = xc - residual_table.shape[1]/2 + fitting_radius 
                y_fitted_shape = yc - residual_table.shape[0]/2 + fitting_radius 
                
        
                
                if return_fwhm:

                        
                    pars = lmfit.Parameters()
                    pars.add('A',
                             value = H_psf,
                             max = 1.25*H_psf,
                             min = 1e-6)
                
                    pars.add('x0',
                             vary = dx_vary,
                             value = x_fitted_shape,
                             min   = x_fitted_shape-dx,
                             max   = x_fitted_shape+dx)
                    pars.add('y0',
                             vary = dy_vary,
                             value = y_fitted_shape,
                             min   = y_fitted_shape-dy,
                             max   = y_fitted_shape+dy)
                    
                    if use_moffat:
                        pars.add('alpha',value = image_params['alpha'],
                                 min = 0,
                                 vary =  False)
                        
                        pars.add('beta',value = image_params['beta'],
                                 min = 0,
                                 vary = False )

                    else:
                        pars.add('sigma', value = image_params['sigma'],
                       
                                  vary = False)
                               
            
                        
                    if use_moffat:
                        
                        fitting_model_fwhm = moffat_fwhm
                        
                        def residual(p):
                            p = p.valuesdict()
                            return (source  - moffat_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(source.shape)).flatten()
                    else:
                        
                        fitting_model_fwhm = gauss_sigma2fwhm
                        
                        def residual(p):
                            p = p.valuesdict()
                            return (source - gauss_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(sigma=p['sigma'])).reshape(source.shape)).flatten()
    
                                    
                    import warnings
                    with warnings.catch_warnings():
                        
                        warnings.simplefilter("ignore")
                        
                        mini = lmfit.Minimizer(residual,
                                               pars,
                                               nan_policy = 'omit',
                                               scale_covar=True)
                        
           
                        result = mini.minimize(method = fitting_method )
                        

                    # This needs to be in the scale of the closeup image and not the overall image
                    FWHM_fitted_xc = result.params['x0'].value - fitting_radius + lower_x_bound
                    FWHM_fitted_yc = result.params['y0'].value - fitting_radius + lower_y_bound

                    if use_moffat:
                        target_PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                    else:
                        target_PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))
    
                    # if not no_print:
                    #     logger.info('Target FWHM: %.3f [ pixels ]\n' % target_PSF_FWHM)
               
                
                
                if  not return_fwhm:

                    if H_psf+bkg_median >= sat_lvl:
                        # print('sat')

                        bkg_median = np.nan
                        H_psf = np.nan
                        H_psf_err = np.nan
                        chi2 = np.nan
                        redchi2 = np.nan
                        xc= np.nan
                        yc = np.nan
        
                        psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,H_psf,H_psf_err,chi2,redchi2))
                        continue

            except Exception as e:
                print('PSF fitting error: %s\n Excluding this source' % e)
                logger.exception(e)
                

                psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H_psf,H_psf_err,max_pixel,chi2,redchi2))

                continue
            
            
            
            # Add these parameters to the output
            psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H_psf,H_psf_err,max_pixel,chi2,redchi2))
            
            
            if return_subtraction_image:

                try:
                    ncols = 3
                    nrows = 2

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(500,1))

                    grid = GridSpec(nrows, ncols ,
                                    wspace=0.5,
                                    hspace=0.1)

                    ax1   = fig.add_subplot(grid[0:2, 0:2])
                    ax_before = fig.add_subplot(grid[0, 2])
                    ax_after = fig.add_subplot(grid[1, 2])

                    image_section = image[int(yc_global -  lower_y_bound): int(yc_global + upper_y_bound),
                                          int(xc_global -  lower_x_bound): int(xc_global + upper_x_bound)]

                    ax_before.imshow(image_section,
                                     vmin = vmin,
                                     vmax = vmax,
                                     origin = 'lower',
                                     cmap='gray',
                                     interpolation = 'nearest')

                    ax1.imshow(image,
                               vmin = vmin,
                               vmax = vmax,
                               origin = 'lower',
                               cmap='gray',
                               interpolation = 'nearest')

                    ax1.scatter(xc_global,
                                yc_global,
                                marker = 'o',
                                facecolor = 'None',
                                color = 'green',
                                s = 25)

                    image_section_subtraction = image_section - PSF_MODEL(xc , yc, 0, H_psf, residual_table,image_params,use_moffat = use_moffat,fitting_radius = fitting_radius,regrid_size = regrid_size,pad_shape = pad_shape,slice_scale = image_section.shape[0]/2)

                    image[int(yc_global  - lower_y_bound): int(yc_global +  upper_y_bound),
                          int(xc_global  - lower_y_bound): int(xc_global +  upper_y_bound)] =  image_section_subtraction

                    ax1.set_xlim(0,image.shape[0])
                    ax1.set_ylim(0,image.shape[1])

                    ax_after.imshow(image_section_subtraction,
                                    vmin = vmin,
                                    vmax = vmax,
                                    origin = 'lower',
                                    cmap='gray',
                                    interpolation = 'nearest')

                    ax_after.axis('off')
                    ax_before.axis('off')

                    ax1.axis('off')

                    ax_after.set_title('After')
                    ax_before.set_title('Before')

                    save_loc = os.path.join(write_dir,'cleaned_images')

                    os.makedirs(save_loc, exist_ok=True)

                    fig.savefig(os.path.join(save_loc,'subtraction_%d.pdf' % idx),
                                bbox_inches='tight')

                    logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')
                    pass

            if plot_PSF_residuals  or save_plot == True:

                try:
                    from astropy.visualization import  ZScaleInterval
                    
                    fitted_source = PSF_MODEL(xc , yc, 0, H_psf, residual_table,fwhm,image_params,use_moffat = use_moffat,fitting_radius = fitting_radius,regrid_size = regrid_size)

                    subtracted_image = source_bkg_free - fitted_source + bkg_surface

                    order = order_shift(abs(source_base))

                    source_base = source_base/order
                    subtracted_image=subtracted_image / order
                    bkg_surface = bkg_surface/ order
                    fitted_source = fitted_source/ order

                    vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(source_base)

                    h, w = subtracted_image.shape

                    x  = np.arange(0, h)
                    y  = np.arange(0, w)

                    X, Y = np.meshgrid(x, y)
                    
                    

                    ncols = 6
                    nrows = 3

                    heights = [1,1,0.75]
                    widths = [1,1,0.75,1,1,0.75]

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(500,aspect = 0.75))

                    grid = GridSpec(nrows, ncols ,wspace=0.5, hspace=0.5,
                                    height_ratios=heights,
                                    width_ratios = widths)

                    ax1   = fig.add_subplot(grid[0:2, 0:2])
                    ax1_B = fig.add_subplot(grid[2, 0:2])
                    ax1_R = fig.add_subplot(grid[0:2, 2])

                    ax2 = fig.add_subplot(grid[0:2, 3:5])

                    ax2_B = fig.add_subplot(grid[2, 3:5])
                    ax2_R = fig.add_subplot(grid[0:2, 5])

                    ax1_B.set_xlabel('X Pixel')
                    ax2_B.set_xlabel('X Pixel')

                    ax1.set_ylabel('Y Pixel')

                    ax1_R.yaxis.tick_right()
                    ax2_R.yaxis.tick_right()

                    ax1_R.xaxis.tick_top()
                    ax2_R.xaxis.tick_top()

                    ax1.xaxis.tick_top()
                    ax2.xaxis.tick_top()

                    ax2.axes.yaxis.set_ticklabels([])

                    bbox=ax1_R.get_position()
                    offset= -0.03
                    ax1_R.set_position([bbox.x0+ offset, bbox.y0 , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax2_R.get_position()
                    offset= -0.03
                    ax2_R.set_position([bbox.x0+ offset, bbox.y0 , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax1_B.get_position()
                    offset= 0.06
                    ax1_B.set_position([bbox.x0, bbox.y0+ offset , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax2_B.get_position()
                    offset= 0.06
                    ax2_B.set_position([bbox.x0, bbox.y0+ offset , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    ax1.imshow(source_base,
                               origin = 'lower',
                               aspect='auto',
                               vmin = vmin,
                               vmax = vmax)
                               

         
                   
                    ax1_R.step(source_base[:,w//2],Y[:,w//2],color = 'blue',label = '1D projection',where='mid')
                    ax1_B.step(X[h//2,:],source_base[h//2,:],color = 'blue',where='mid')

                    # include surface
                    ax1_R.step(bkg_surface[:,w//2],Y[:,w//2],color = 'red',label = 'Background Fit',where='mid')
                    ax1_B.step(X[h//2,:],bkg_surface[h//2,:],color = 'red',where='mid')

                    # include fitted_source
                    ax1_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],color = 'green',label = 'PSF')
                    ax1_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],color = 'green')

                    ax1_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(order))

                    ax1_R.set_xlabel('Counts [$10^{%d}$]' % np.log10(order))

                    ax2_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(order))

                    ax2_R.set_xlabel('Counts [$10^{%d}$] ' % np.log10(order))

                    ax2.imshow(subtracted_image,
                                vmin = vmin,
                                vmax = vmax,
                                origin = 'lower',
                                aspect='auto')
              
                    ax2_R.step(subtracted_image[:,w//2],Y[:,w//2],color = 'blue',where='mid',)
                    ax2_B.step(X[h//2,:],subtracted_image[h//2,:],color = 'blue',where='mid',)

                    # Show surface
                    ax2_R.step(bkg_surface[:,w//2],Y[:,w//2],color = 'red',where='mid',)
                    ax2_B.step(X[h//2,:],bkg_surface[h//2,:],color = 'red',where='mid',)

                    # include fitted_source
                    ax2_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],color = 'green')
                    ax2_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],color = 'green')

                    ax1_R.tick_params(axis='x', rotation=-90)
                    ax2_R.tick_params(axis='x', rotation=-90)
                    
                    # ax1_B.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
                    ax1_B.set_xlim(ax1.get_xlim()[0],ax1.get_xlim()[1])

                    ax1_R.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
                    
                    ax2_B.set_xlim(ax2.get_xlim()[0],ax2.get_xlim()[1])

                    ax2_R.set_ylim(ax2.get_ylim()[0],ax2.get_ylim()[1])

                    
                    ax1.axvline(xc,ls = '--',color = 'black',alpha = 0.5,label = 'Best fit',)
                    ax1.axhline(yc,ls = '--',color = 'black',alpha = 0.5,label = 'Best fit',)
                    
                    ax2.axvline(xc,ls = '--',color = 'black',alpha = 0.5)
                    ax2.axhline(yc,ls = '--',color = 'black',alpha = 0.5)
                     
                    ax1_B.axvline(xc,ls = '--',color = 'black',alpha = 0.5)
                    ax2_B.axvline(xc,ls = '--',color = 'black',alpha = 0.5)
                    
                    ax1_R.axhline(yc,ls = '--',color = 'black',alpha = 0.5)
                    ax2_R.axhline(yc,ls = '--',color = 'black',alpha = 0.5)

                    for ax in [ax1,ax2]:
    
                        search_circle = plt.Circle(( 0.5*residual_table.shape[1] , 0.5*residual_table.shape[0] ),
                                                    radius =  dx,
                                                    ls = ':',
                                                    color = 'red',
                                                    alpha = 0.5,
                                                    lw = 0.5,
                                                    label = 'Fitting area',
                                                    fill = False)
 
                        ax.add_artist( search_circle )
                        
                    if return_fwhm:
                        try:
                            for ax in [ax1,ax2]:
                                ax.scatter(FWHM_fitted_xc,FWHM_fitted_yc,
                                           marker = 'x',
                                           color = 'purple',
                                           s = 20,
                                           label = 'Fitted FWHM [%.3f pixels]' % target_PSF_FWHM)
                        
                        except:
                            pass
                        

                    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                    handles,labels = [sum(i, []) for i in zip(*lines_labels)]

                    by_label = dict(zip(labels, handles))

                    fig.legend(by_label.values(), by_label.keys(),
                               bbox_to_anchor=(0.5, 0.92), 
                               loc='lower center',
                               ncol = 5,
                               frameon=False)

                    if plot_PSF_residuals:
                    
                        pathlib.Path(write_dir+'/'+'psf_subtractions/').mkdir(parents = True, exist_ok=True)
                        
                        save_name = lambda x: write_dir+'psf_subtractions/'+'psf_subtraction_{}.png'.format(int(x))
                        i = 0
                        while True:
                            if not os.path.exists(save_name(n+i)):
                                break
                            else:
                                i+=1
                        
                        plt.savefig(save_name(n+i),bbox_inches='tight')
                        
                    else:
                        
                        save_loc = os.path.join(write_dir,'target_psf_'+base+'.pdf')
                        
                        fig.savefig(save_loc,bbox_inches='tight')

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')

        except Exception as e:
             logger.exception(e)

             
             psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H_psf,H_psf_err,max_pixel,chi2,redchi2))

             continue


    new_df =  pd.DataFrame(psf_params,
                           columns = ('idx','x_fitted','y_fitted','x_closeup_fitted','y_closeup_fitted','bkg','noise','H_psf','H_psf_err','max_pixel','chi2','redchi2'),
                           index = sources.index)

    if return_fwhm:
        new_df['target_fwhm'] = target_PSF_FWHM
        

    elif not no_print:
        print('  ')
        
    PSF_sources = pd.concat([sources,new_df],axis = 1)
    
    if not return_closeup:
        
        return PSF_sources
        
    else:
        
        return  PSF_sources,source_base


# =============================================================================
# Convert fitted heights to counts under PSF
# =============================================================================

def do(df,residual_image= None, ap_size = 1.7, fwhm = 7, unity_PSF_counts = None,
       unity_residual_counts = None, use_moffat = True, image_params = None):
    '''
    
        We exploit the fact that each point source in an image appears the same, i.e.
    the PSF model matches each each, and the only variable that change between each
    source is it's height. We use this function to get the counts under a PDF model
    with ampltide equal to 1. 
    
    This allows use to scale these counts to any source
    fitted with the same PSF later on.
    
    :param df: Dataframe containing the columns *H_PSF* and *H_PSF_err* which corrospsonds to the height of the best fitted PSF and the error associated with that value
    :type df: Dataframe
    :param residual_image: Residual table, normalised to unity that is the same shape as the base analytical function, defaults to None
    :type residual_image:  2D array, optional
    :param ap_size: Multiple of FWHM to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param fwhm: Full Width Half Maximum of image
    :type fwhm: float
    :param unity_PSF_counts: Number of counts under a PSF model with amplitude equal to 1, if this value is left as None, and the residual table is given, this value is calculated and returned, defaults to None
    :type unity_PSF_counts: float, optional
    :param unity_residual_counts: Number of counts under a residual table normalised to an PSF amplitude equal to 1, defaults to None
    :type unity_residual_counts: float, optional
    :param use_moffat: If True, use a moffat function as the analytical function, else use a gaussian, defaults to True
    :type use_moffat: bool, optional
    :param image_params: Dictionary containing analytical model params. If a moffat is used, this dictionary should containing *alpha* and *beta* and their respective values, else if a gaussian is used, this dictionary should include *sigma* and its value.
    :type image_params: dict
    :return: If *unity_PSF_counts* is None and the residual is given, this function will calculate and return the value of *unity_PSF_counts* as well as the counts under the fitted PSF given in the initial dataframe. If *unity_PSF_counts* is defined, then the counts information is returned only

    '''

    from photutils import CircularAperture
    from photutils import aperture_photometry
    import logging
    import numpy as np
    
    from autophot.packages.functions import gauss_2d,moffat_2d,border_msg
    
    logger = logging.getLogger(__name__)
    
    find_unity_PSF_counts = False

    if unity_PSF_counts is None:
        
        border_msg('Measuring PSF model')
    
        find_unity_PSF_counts = True
        
        xc = residual_image.shape[1]/2
        yc = residual_image.shape[0]/2
        
        # Integration radius
        int_scale = ap_size * fwhm
        # int_scale = 30
                                                               
        x_rebin = np.arange(0,residual_image.shape[1])
        y_rebin = np.arange(0,residual_image.shape[0])
        
        xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)
        
        if use_moffat:
            
            core = moffat_2d((xx_rebin,yy_rebin),xc,yc,0,1,image_params).reshape(residual_image.shape)
            
        else:
            
            core =  gauss_2d((xx_rebin,yy_rebin),xc,yc,0,1,image_params).reshape(residual_image.shape)
        
        unity_PSF_model = core + residual_image
        
        
        # Aperture Photometry over residual
        apertures = CircularAperture((xc,yc), r=int_scale)
        residual_table = aperture_photometry(residual_image, 
                                             apertures,
                                             method='subpixel',
                                             subpixels=5)
        residual_int = residual_table[0]
        

        
        PSF_table = aperture_photometry(unity_PSF_model,apertures,
                                        method='subpixel',
                                        subpixels=5)
        PSF_int =  PSF_table[0]
                    
        # Counts from PSF model
        unity_PSF_counts  = float(PSF_int['aperture_sum'])
        
        
        # Counts from Residual component of PSF
        unity_residual_counts = float(residual_int['aperture_sum'])
        
        logger.info('Unity PSF: %.1f [counts] ' % unity_PSF_counts)
        logger.info('Unity Residual table: %.1f [counts] ' % unity_residual_counts)
    
    df['psf_counts']     = df.H_psf.values * unity_PSF_counts
    

    
    df['psf_counts_err'] = np.nan_to_num(df.H_psf_err.values) * unity_PSF_counts

    
    
    if find_unity_PSF_counts:
        
        return df,unity_PSF_counts
    
    else:

        return df








def compute_multilocation_err(image, fwhm, PSF_model, image_params, exp_time,
                              fpath, scale, unity_PSF_counts, 
                              target_error_compute_multilocation_number = 5,
                              target_error_compute_multilocation_position = 1,
                              use_moffat = True, fitting_method = 'least_sqaure',
                              ap_size = 1.7, fitting_radius = 1.3, 
                              regrid_size = 10, xfit = None, yfit = None, 
                              Hfit = None, r_table = None, remove_bkg_local = True,
                              remove_bkg_surface = False, remove_bkg_poly = False,
                              remove_bkg_poly_degree = 1, bkg_level = 3):
    '''
        Package to employ the same error technique as in the `SNOOPY
    <https://sngroup.oapd.inaf.it/snoopy.html>`_ code. In brief, error
    estimates from the transient measurement are obtained through artificial
    star experiment in which a fake star of magnitude equal to that of the SN,
    is placed in the PSF-fit residual image 
    in a position close to, but not coincident with that of the real source.
    
    The artificially injects source is then recovered in an identical manner to
    the original transient magnitude. This is repeated several times. The
    dispersion of these recovered measurements is then taken as the error on
    the transient measurement and added in quadrature. 

    :param image: Image containing transietnf flux
    :type image: 2D array
    :param fwhm: Full Width Half Maximum of image. This is used to constrain the position of the injected sources
    :type fwhm: float
    :param PSF_model: PSF model that is used to fit transient flux, this will also be used to inject the artifical sources
    :type PSF_model: callable function
    :param image_params: Dictionary containing analytical model params. If a moffat is used, this dictionary should containing *alpha* and *beta* and their respective values, else if a gaussian is used, this dictionary should include *sigma* and its value.
    :type image_params: dict
    :param exp_time: Exposure time ins seconds of an image. This is used to calculate the S/N of a potential source.
    :type exp_time: float
    :param unity_PSF_counts: Number of counts under a PSF model with amplitude equal to 1, defaults to None
    :type unity_PSF_counts: float, optional
    :param target_error_compute_multilocation_number: Number of times the psuedo-transient PSF will be reinjected at locations radially around the original SN site, defaults to 5
    :type target_error_compute_multilocation_number: int, optional
    :param target_error_compute_multilocation_position: Multipl of FWHM with which to place the psuedo-transient PSF away from the original site of best fit. Set to -1 to perform sub pixel injection and recovery, defaults to 1
    :type target_error_compute_multilocation_position: float, optional
    :param use_moffat: If True, use a moffat function for FWHM fitting defaults to True
    :type use_moffat: bool, optional
    :param fitting_method: Fitting method when fitting the PSF model, defaults to 'least_square'
    :type fitting_method: str, optional
    :param ap_size: Multiple of FWHM to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param fitting_radius: zoomed region around location of best fit to focus fitting. This allows for the fitting to be concentrated on high S/N areas and not fit the low S/N wings of the PSF, defaults to 1.3
    :type fitting_radius: float, optional
    :param regriding_size: When expanding to larger pseudo-resolution, what zoom factor to use, defaults to 10
    :type regriding_size: int, optional
    :param xfit: X pixel location of best fit for the transient psf in the image, defaults to None
    :type xfit: float, optional
    :param yfit: y pixel location of best fit for the transient psf in the image, defaults to None
    :type yfit: float, optional
    :param Hfit: PSF amplitude of best fit for the transient psf in the image, defaults to None, defaults to None
    :type Hfit: float, optional
    :param r_table: Resiudal table, normailised to unity that is the same shape as the base analytical function
    :type r_table: 2D array
    :param remove_bkg_local: If True, use the local median of the image and subtract this to produce a background free image, see above, defaults to True
    :type remove_bkg_local: Boolean, optional
    :param remove_bkg_surface: If True, use the background fitted surface of the image and subtract this to produce a background free image, see above, defaults to False
    :type remove_bkg_surface: Boolean, optional
    :type remove_bkg_poly: If True, use the background polynomial surface of the image and subtract this to produce a background free image, see above, optional
    :param remove_bkg_poly_degree: If remove_bkg_poly is True, this is the degree of the polynomial fitted to the image, 1 = flat surface, 2 = 2nd order polynomial etc, defaults to 1
    :param bkg_level: The number of standard deviations, below which is assumed to be due to the background noise distribution, defaults to 3
    :type bkg_level: float, optional
    :return: Returns the standard deviation of the recovered magnitudes of the artifically injection pseudo-transient PSFs
    :rtype: float
    '''

    
    import numpy as np
    import pandas as pd
    from autophot.packages import psf
    from autophot.packages.functions import calc_mag
    

    # Number of times to tagret's PSF is injected and recovered
    N = target_error_compute_multilocation_number
    
    # Multiples of FWHM with which to place this sources randomly e.g. 1 -> +/- FWHM/2
    position_shift = target_error_compute_multilocation_position
    
    ran_dx = xfit + (np.random.uniform(-1,1,N) * position_shift * fwhm/2 )
    ran_dy = yfit + (np.random.uniform(-1,1,N) * position_shift * fwhm/2 )

    injection_df = pd.DataFrame([list(ran_dx ),list(ran_dy)])
    injection_df = injection_df.transpose()
    injection_df.columns = ['x_pix','y_pix']
    injection_df.reset_index(inplace = True,drop = True)
    
    Fitted_PSF = PSF_model(xfit, yfit, 0, Hfit,
                       r_table,
                       fwhm,
                       image_params,
                       use_moffat = use_moffat,
                       fitting_radius = fitting_radius,
                       regrid_size = regrid_size,
                       pad_shape = image.shape)
# xc, yc, sky, H, r_table, fwhm,image_params,use_moffat = True, fitting_radius = 1.3,regrid_size =10 ,slice_scale = None,pad_shape = None
    # Remove PSF from image
    residual_image = image - Fitted_PSF
    
    hold_psf_position = False
    
    magnitudes_recovered = []
    
    
    
    for i in range(len(injection_df)):
        
        test_PSF = PSF_model(injection_df['x_pix'].values[i],injection_df['y_pix'].values[i],
                         0,
                         Hfit, 
                         r_table,
                         fwhm,
                         image_params,
                         use_moffat = use_moffat,
                         fitting_radius = fitting_radius,
                         regrid_size = regrid_size,
                         pad_shape = image.shape)

        psf_fit  = psf.fit(image = residual_image + test_PSF,
                        sources = injection_df.iloc[[i]],
                        residual_table = r_table,
                        fwhm = fwhm,
                        fpath = fpath,
                        fitting_radius = fitting_radius,
                        regrid_size = regrid_size,
                        # scale = scale,
                        use_moffat = use_moffat,
                        image_params = image_params,
                        fitting_method = fitting_method,
                        hold_pos = hold_psf_position,
                        return_fwhm = True,
                        no_print = True,
                        remove_bkg_local = remove_bkg_local, 
                        remove_bkg_surface = remove_bkg_surface,
                        remove_bkg_poly   = remove_bkg_poly,
                        remove_bkg_poly_degree = remove_bkg_poly_degree,
                        bkg_level = bkg_level)
       
        psf_params = psf.do(df = psf_fit,
                    residual_image = r_table,
                    ap_size = ap_size,
                    fwhm = fwhm,
                    unity_PSF_counts =unity_PSF_counts,
                    use_moffat = use_moffat,
                    image_params = image_params)
        
        psf_flux = psf_params['psf_counts'].values/exp_time
        
        magnitudes_recovered.append(calc_mag(psf_flux)[0])
      
    error = np.nanstd(magnitudes_recovered)
    
    
    print('Error from multlocation [%d] recovery: %.3f [mag]' % (N,error))              
    
    return error