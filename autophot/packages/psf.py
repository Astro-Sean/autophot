def sigma_clip_list_of_list(list_of_list,shape = None):
    
    
    import numpy as np
    from astropy.stats import sigma_clip
    
    if shape is None:
        shape = list_of_list[0].shape
        
        
    list_of_pixels =  np.dstack(list_of_list)
    
    
    new_images = np.zeros(shape)
    
    for i in range(new_images.shape[1]):
        for j in range(new_images.shape[0]):
            
            c = sigma_clip(list_of_pixels[i,j],
                           maxiters=None,
                           cenfunc=np.nanmedian,
                           masked=False,
                        
                           # axis = 1
                           )
            c = list_of_pixels[i,j]
            
            new_images[i][j] = np.nanmedian(c)
            
    return new_images


def build_r_table(base_image,selected_sources,autophot_input,fwhm,iterations = 1):
    
    import numpy as np
    

    import pandas as pd
    import lmfit
    import logging

    import os
    import warnings
    import matplotlib.pyplot as plt
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    
    from photutils import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    
    from autophot.packages.functions import scale_roll
    from autophot.packages.functions import rebin
    from autophot.packages.background import remove_background
    from autophot.packages.functions import pix_dist,gauss_sigma2fwhm
    from autophot.packages.uncertain import SNR
    from autophot.packages.aperture  import measure_aperture_photometry
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import moffat_2d,moffat_fwhm

    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    if autophot_input['use_moffat']:
        
        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm
        
    try:

        logger = logging.getLogger(__name__)

        image = base_image.copy()

        # Only fit to a small image with radius ~the fwhm
        fitting_radius = int(np.ceil(autophot_input['fitting_radius'] * fwhm))
        x_slice = np.arange(0,2*fitting_radius)
        xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

        # for matchinf each source residual image with will regrid the image for shifting later
        regriding_size = int(autophot_input['regrid_size'])
        # Residual Table in extended format
        # residual_table = np.zeros((int(2 * autophot_input['scale'] * regriding_size), int(2 * autophot_input['scale']*regriding_size)))
        residual_table = []

        if regriding_size % 2 > 0:
            logger.info('regrid size must be even adding 1')
            regriding_size += 1

        # FWHM/sigma fits
        fwhm_fit = []

        # what sources will be used
        construction_sources = []
        
        # Some test - if available to fine useful sources
        if not autophot_input['use_PSF_starlist']:
            selected_sources = selected_sources[~np.isnan(selected_sources['FWHM'])]
        
        if 'include_fwhm' in selected_sources and not autophot_input['use_PSF_starlist']:
            if np.sum(selected_sources['include_fwhm'])>10:
                selected_sources = selected_sources[selected_sources['include_fwhm']]
            
        if 'include_median' in selected_sources and not autophot_input['use_PSF_starlist']:
            if np.sum(selected_sources['include_median'])>10:
                selected_sources = selected_sources[selected_sources['include_median']]
            


        if autophot_input['use_local_stars_for_PSF'] and not autophot_input['prepare_templates'] and not autophot_input['use_PSF_starlist']:

            # Use local stars given by 'use_acrmin' parameter

            selected_sources_test = selected_sources[selected_sources['dist'] <= autophot_input['local_radius']]

            selected_sources  = selected_sources_test
            
        
            
        if not autophot_input['use_PSF_starlist']:
            
            high_counts_idx = [i for i in selected_sources.counts_ap.sort_values(ascending = False).index]
            max_sources_used  = autophot_input['psf_source_no']
            
        else:
            
            max_sources_used = len(selected_sources.index)
            high_counts_idx = selected_sources.index
            
        sources_used = 0
        n = 0
        failsafe = 0
        psf_mag = []
        image_radius_lst = []
        sources_dict = {}

        
        if  autophot_input['use_PSF_starlist']:
            logger.info('\nBuilding PSF model with User source list (%d)' % len(high_counts_idx))
        else:
            logger.info('\nBuilding PSF model and residual table')
            
        while sources_used <= autophot_input['psf_source_no']-1:

            if failsafe>25:
                logger.info('PSF - Failed to build psf')
                residual_table=None
                fwhm_fit = fwhm

            if n >= len(high_counts_idx):
                
                if sources_used  >= autophot_input['min_psf_source_no']:
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
                psf_image = image[int(selected_sources.y_pix[idx]-autophot_input['scale']): int(selected_sources.y_pix[idx] + autophot_input['scale']),
                                  int(selected_sources.x_pix[idx]-autophot_input['scale']): int(selected_sources.x_pix[idx] + autophot_input['scale'])]
                if not autophot_input['use_PSF_starlist']:
                    
                    psf_fwhm_fitted = selected_sources.FWHM[idx]
                else:
                    psf_fwhm_fitted = np.nan
                    
                
                mean, median, std = sigma_clipped_stats(psf_image,
                                        sigma = autophot_input['bkg_level'],
                                        maxiters = 10)
                
                daofind = DAOStarFinder(fwhm      = autophot_input['fwhm'],
                                        threshold = 5 * std,
                                        sharplo   =  0.2,sharphi = 1.0,
                                        roundlo   = -1.0,roundhi = 1.0
                                        )

                sources = daofind(psf_image - median)
                
                if sources is None:
                    
                    logger.info('Cannot detect any point source - skipping')
                        
                    continue
                    
                
                sources = sources.to_pandas()
                
                
                if len(sources)>1:
                    
                    # More than one source found! Might be detecting the same source - check if all points are withint one fwhm
                    dist = [max(pix_dist(x,sources['xcentroid'].values,y,sources['ycentroid'].values)) for x, y in zip(sources['xcentroid'], sources['ycentroid'])]
                    # print(dist)
                    # TODO: turn this back on
                    if all(x >  2*autophot_input['fwhm'] for x in dist) and not autophot_input['use_PSF_starlist']:

                        logger.info('Found faint soures near PSF star at %d sigma - skipping '%autophot_input['bkg_level'])
                        
                        continue
 
                if len(psf_image) == 0:
                    logger.info('PSF image ERROR')
                    continue
                
                if np.min(psf_image) == np.nan:
                    logger.info('PSF image ERROR (nan in image)')
                    continue

                psf_image = image[int(selected_sources.y_pix[idx]-autophot_input['scale']): int(selected_sources.y_pix[idx]+autophot_input['scale']),
                                  int(selected_sources.x_pix[idx]-autophot_input['scale']): int(selected_sources.x_pix[idx]+autophot_input['scale'])]
                
                psf_image_bkg_free, bkg_surface, background, noise = remove_background(psf_image,
                                                                                        autophot_input,
                                                                                        psf_image.shape[1]/2,
                                                                                        psf_image.shape[0]/2)


                x = np.arange(0,2*autophot_input['scale'])
                xx,yy= np.meshgrid(x,x)

                pars = lmfit.Parameters()
                pars.add('A',value = 0.75*np.nanmax(psf_image_bkg_free),
                         min=0)
                pars.add('x0',value = psf_image_bkg_free.shape[1]/2,
                         min = 0
                         
                         
                         ,
                         max =psf_image_bkg_free.shape[1] )
                pars.add('y0',value = psf_image_bkg_free.shape[0]/2,
                         min = 0,
                         max =psf_image_bkg_free.shape[0])

                if autophot_input['use_moffat']:
                    pars.add('alpha',value = autophot_input['image_params']['alpha'],
                             min = 0,
                             vary =  autophot_input['fit_PSF_FWHM'] )
                    pars.add('beta',value = autophot_input['image_params']['beta'],
                             min = 0,
                             vary = autophot_input['vary_moff_beta'] or autophot_input['fit_PSF_FWHM']  )

                else:
                    pars.add('sigma', value = autophot_input['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(autophot_input['max_fit_fwhm']),
                              vary = autophot_input['vary_moff_beta'] or autophot_input['fit_PSF_FWHM']
                             )

                if autophot_input['use_moffat']:
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
                    
                    result = mini.minimize(method = autophot_input['fitting_method'])
                    
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                # global pixel coorindates base on best gaussian fit
                xc_global = xc - psf_image_bkg_free.shape[1]/2 + int(selected_sources.x_pix[idx])
                yc_global = yc - psf_image_bkg_free.shape[0]/2 + int(selected_sources.y_pix[idx])

                # recenter image absed on location of best fit x and y
                psf_image = image[int(yc_global-autophot_input['scale']): int(yc_global + autophot_input['scale']),
                                  int(xc_global-autophot_input['scale']): int(xc_global + autophot_input['scale'])]

                psf_image_bkg_free,_,bkg_median,noise = remove_background(psf_image,
                                                                          autophot_input,
                                                                          psf_image.shape[1]/2,
                                                                          psf_image.shape[0]/2)

                psf_image_slice = psf_image_bkg_free[int(psf_image_bkg_free.shape[1]/2 - fitting_radius):int(psf_image_bkg_free.shape[1]/2 + fitting_radius) ,
                                                     int(psf_image_bkg_free.shape[0]/2 - fitting_radius):int(psf_image_bkg_free.shape[0]/2 + fitting_radius) ]

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmean(psf_image_slice),
                         min = 1e-6,
                         max = np.nanmax(psf_image_slice)*3 )
                pars.add('x0',value = psf_image_slice.shape[1]/2,
                         min = 1,
                         max = psf_image_slice.shape[1])
                pars.add('y0',value = psf_image_slice.shape[0]/2,
                         min = 1,
                         max = psf_image_slice.shape[0] )

                if autophot_input['use_moffat']:

                    pars.add('alpha',value = autophot_input['image_params']['alpha'],
                             min = 0,
                             vary =  autophot_input['fit_PSF_FWHM'] )

                    pars.add('beta',value = autophot_input['image_params']['beta'],
                             min = 0,
                             vary = autophot_input['vary_moff_beta'] or autophot_input['fit_PSF_FWHM']  )

                else:

                    pars.add('sigma', value = autophot_input['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(autophot_input['max_fit_fwhm']),
                             vary =  autophot_input['fit_PSF_FWHM'])
                             

                if autophot_input['use_moffat']:
                    def residual(p):
                        p = p.valuesdict()
                        return (psf_image_slice  - moffat_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(psf_image_slice .shape)).flatten()
                else:
                    def residual(p):
                        p = p.valuesdict()
                        return (psf_image_slice - gauss_2d((xx_sl,yy_sl),p['x0'],p['y0'],0,p['A'],dict(sigma=p['sigma'])).reshape(psf_image_slice.shape)).flatten()

                               
                import warnings
                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
                    
                    mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)
                    
                    result = mini.minimize(method = autophot_input['fitting_method'])

                positions  = list(zip([xc_global],[yc_global]))

                psf_counts,psf_maxpixels,psf_bkg,psf_bkg_std = measure_aperture_photometry(positions,
                                                                 image,
                                                                 radius = autophot_input['ap_size']    * fwhm,
                                                                 r_in   = autophot_input['r_in_size']  * fwhm,
                                                                 r_out  = autophot_input['r_out_size'] * fwhm)
                
                PSF_flux = psf_counts/autophot_input['exp_time']
                PSF_bkg_flux = psf_bkg/autophot_input['exp_time']
                psf_noise_flux = psf_bkg_std/autophot_input['exp_time']

                if autophot_input['use_moffat']:

                    PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:

                    PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))

                PSF_SNR = SNR(flux_star = PSF_flux ,
                              flux_sky = PSF_bkg_flux ,
                              exp_t = autophot_input['exp_time'],
                              radius = autophot_input['ap_size']*autophot_input['fwhm'] ,
                              G  = autophot_input['GAIN'],
                              RN =  autophot_input['RDNOISE'],
                              DC = 0 )

                
                if np.isnan(PSF_SNR) or np.isnan(PSF_FWHM):
                    logger.debug('PSF Contruction source fitting error')
                    continue

                if PSF_SNR < autophot_input['construction_SNR'] and autophot_input['exp_time'] > 1:
                    logger.info('PSF constuction source too low: %s' % int(PSF_SNR))
                    logger.info('\nRan out of PSF sources above SNR=%d' % autophot_input['construction_SNR'] )
                    break
                else:

                    # print('\rPSF source %d / %d :: SNR: %d' % (int(PSF_SNR)),end = '')
                    pass

                # print(result.params)
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                H = result.params['A'].value
                H_err = result.params['A'].stderr
                
                psf_mag.append(PSF_flux)

                chi2 = result.chisqr

                xc_correction =  xc - fitting_radius + autophot_input['scale']
                yc_correction =  yc - fitting_radius + autophot_input['scale']

                if autophot_input['use_moffat']:

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
                
                residual_counts_before = np.sum(residual)

                residual_regrid = np.repeat(np.repeat(residual, regriding_size, axis=0), regriding_size, axis=1)

                x_roll = scale_roll(fitting_radius,xc,regriding_size)
                y_roll = scale_roll(fitting_radius,yc,regriding_size)

                residual_roll = np.roll(np.roll(residual_regrid,y_roll,axis=0),x_roll,axis = 1)

                # residual_table += residual_roll
                residual_table.append(residual_roll)
                
                residual_counts_after = np.sum(rebin(residual_roll, (int(2*autophot_input['scale']),int(2*autophot_input['scale']))))
                
                
                print('Before: %.3f :: After %.3f' % (residual_counts_before,residual_counts_after))
                
                sources_used +=1
                

                sources_dict['PSF_%d'%sources_used] = {}

                # Save details about each source used to make the PSF
                sources_dict['PSF_%d'%sources_used]['x_pix']  = xc_global
                sources_dict['PSF_%d'%sources_used]['y_pix']  = yc_global
                sources_dict['PSF_%d'%sources_used]['H_psf'] = float(H)

                sources_dict['PSF_%d'%sources_used]['fwhm'] = PSF_FWHM
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

        if sources_used < autophot_input['min_psf_source_no']:
            logger.info('BUILDING PSF: Not enough useable sources found')
            return None,None,construction_sources.append([np.nan]*5),autophot_input
        
        # Get the mean of the residual tavles
        residual_table = np.nanmean(np.dstack(residual_table),axis = -1)

        # regrid residual table to psf size
        r_table  = rebin(residual_table, (int(2*autophot_input['scale']),int(2*autophot_input['scale'])))
        
        
        construction_sources = pd.DataFrame.from_dict(sources_dict, orient='index',
                                                      columns=['x_pix','y_pix',
                                                               'H_psf','H_psf_err',
                                                               'fwhm',
                                                               'x_best','y_best'])
        construction_sources.reset_index(inplace = True)
        
        
        if autophot_input['save_PSF_stars']:
            construction_sources.to_csv( os.path.join(autophot_input['write_dir'],'PSF_stars_'+autophot_input['base']+'.csv'))
            

        if autophot_input['plot_PSF_model_residual']:
            # Plot figure of PSF shift to make sure model is made correctly
            from autophot.packages.create_plots import plot_PSF_model_steps
            plot_PSF_model_steps(sources_dict,autophot_input,image)

        logger.info('\nPSF built using %d sources\n'  % sources_used)

        if autophot_input['save_PSF_models_fits'] or autophot_input['use_zogy']:

            from astropy.io import fits
            import os

            if autophot_input['use_moffat']:
                
                

                PSF_model_array = r_table+ moffat_2d((xx,yy),r_table.shape[1]/2,r_table.shape[0]/2,
                                                          0,1,
                                                          dict(alpha=autophot_input['image_params']['alpha'],
                                                               beta=autophot_input['image_params']['beta'])).reshape(r_table.shape)

            else:
                PSF_model_array = r_table + gauss_2d((xx,yy),r_table.shape[1]/2,r_table.shape[0]/2,
                                                         0,1,
                                                         dict(sigma=autophot_input['image_params']['sigma'])).reshape(r_table.shape)

            hdu = fits.PrimaryHDU(PSF_model_array/np.sum(PSF_model_array))
            hdul = fits.HDUList([hdu])
            psf_model_savepath = os.path.join(autophot_input['write_dir'],'PSF_model_'+autophot_input['base']+'.fits')
            hdul.writeto(psf_model_savepath,
                         overwrite = True)

            print('PSF model saved as: %s' % psf_model_savepath)

    except Exception as e:
        logger.exception('BUILDING PSF: ',e)
        raise Exception

    return r_table,fwhm_fit,construction_sources,autophot_input




# =============================================================================
# Fitting of PSF
# =============================================================================
def PSF_MODEL(xc, yc, sky, H, r_table, autophot_input, slice_scale = None,pad_shape = None):
    '''
     Point Spread Function model for use in Autophot. This function is used in psf.fit and numerous limiting magnitude packages to effectively model point sources in an image.
     
    :param xc: x-pixel coordinate
    :type xc: float
    :param yc: y-pixel coordinate
    :type yc: float
    :param sky: sky background offset
    :type sky: numpy array
    :param H: height/amplitude of PSF model
    :type H: float
    :param r_table: residual table from psf.PSF_MODEL_table
    :type r_table: numpy array
    :param autophot_input: AutoPhot Control dictionary, defaults to autophot_input
    :type autophot_input: dict, optional
    Requires autophot_input keywords:
    
        - **use_moffat** (*boolean*): Use a moffat function as the base analytical function for the PSf model
        - **image_params** (*dict*): dictionary containing values of analytical function. if 'use_moffat' dictionary should contain 'alpha' and 'beta' keys and their values, else dictionary should contain 'sigma' and its value

    :param slice_scale: If defined, focus on center of image of size +/- slice , defaults to None
    :type slice_scale: int, optional
    :param pad_shape: If if PSF needs to be resized to larger image, given shape of larger image via this value , defaults to None
    :type pad_shape: tuple, optional
    :return: PSF model
    :rtype: numpy array

    '''
    
    import numpy as np
    from autophot.packages.functions import gauss_2d,moffat_2d,moffat_fwhm,gauss_sigma2fwhm
    from autophot.packages.functions import gauss_2d,moffat_2d,moffat_fwhm,gauss_sigma2fwhm
    from autophot.packages.functions import scale_roll,rebin,gauss_fwhm2sigma,set_size,order_shift
    
    fwhm = autophot_input['fwhm']
    
    fitting_radius = int(np.ceil(autophot_input['fitting_radius'] * fwhm))
    regriding_size = int(autophot_input['regrid_size'])

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

        if autophot_input['use_moffat']:
            core = moffat_2d((xx_rebin,yy_rebin),xc,yc,sky,H,autophot_input['image_params']).reshape(psf_shape)

        else:
            core = gauss_2d((xx_rebin,yy_rebin),xc,yc,sky,H,autophot_input['image_params']).reshape(psf_shape)
        
        # Blow up residual table to larger size
        
        residual_rebinned = np.repeat(np.repeat(r_table, regriding_size, axis=0), regriding_size, axis=1)

        
        #TODO: Check this - may be where the fitting issue is coming from
        # scale roll = where you wana go,where you are
        x_roll = scale_roll(xc,r_table.shape[1]/2,regriding_size)
        y_roll = scale_roll(yc,r_table.shape[0]/2,regriding_size)
        
        # Roll in y direction then in x direction
        residual_roll = np.roll(np.roll(residual_rebinned,y_roll,axis=0),x_roll,axis = 1)
        
        # print(residual_roll)
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
        # print('PSF model error: %s' % e)
        # logger.exception(e)
        import os,sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        psf = np.nan

    return psf
    

# =============================================================================
# Testing of the PSF mode
# =============================================================================

def test_psf_model(image,PSF_model_sources,residual_table,autophot_input):
    
    from autophot.packages.background import remove_background
    from autophot.packages.functions import set_size
    
    # Model used to fit PSF
    from lmfit import Model
    import numpy as np
    import os
    # test model on same PSF stars that it was built from and return plot 
    fwhm = autophot_input['fwhm']
    
    fitting_radius = int(np.ceil(autophot_input['fitting_radius'] * fwhm))
        
    slice_scale = fitting_radius
    pad_shape = None
    
    regriding_size = int(autophot_input['regrid_size'])
    
    dx = autophot_input['dx']
    dy = autophot_input['dy']
    
    dx_vary = True
    dy_vary = True
    
        
    
    # Define the PSF residual model
    def psf_residual(x,x0,y0,A):

        res =  PSF_MODEL(x0,y0,0,A,
                            residual_table,
                            autophot_input,
                            slice_scale = slice_scale,
                            pad_shape = pad_shape)
                                
        return res.flatten()
    
    
    psf_residual_model = Model(psf_residual)
    
    psf_residual_model.set_param_hint('x0',
                                      vary = dx_vary,
                                      value = 0.5*residual_table.shape[1],
                                      min   = 0.5*residual_table.shape[1]-dx,
                                      max   = 0.5*residual_table.shape[1]+dx)
    
    # 0.5*residual_table.shape[1]+dx

    psf_residual_model.set_param_hint('y0',
                                      vary =dy_vary,
                                      value = 0.5*residual_table.shape[0],
                                      min   = 0.5*residual_table.shape[0]-dy,
                                      max   = 0.5*residual_table.shape[0]+dy)

    x_slice = np.arange(0,2*fitting_radius)
    
    xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)
    

    for n  in range(len(PSF_model_sources.index)):
        
        print('Checking PSF model source stat %d / %d' % (n,len(PSF_model_sources)))
         
        idx = list(PSF_model_sources.index)[n]
        
        xc_global = PSF_model_sources.x_pix[idx]
        yc_global = PSF_model_sources.y_pix[idx]
           
        source_base =   image[int(yc_global-autophot_input['scale']): int(yc_global + autophot_input['scale']),
                              int(xc_global-autophot_input['scale']): int(xc_global + autophot_input['scale'])]
        
        # source_base = np.roll(source_base,1,axis  = 1)
        source_bkg_free, bkg_surface, bkg_median,noise = remove_background(source_base,autophot_input)
        
        source = source_bkg_free[int(0.5*source_bkg_free.shape[1] - fitting_radius):int(0.5*source_bkg_free.shape[1] + fitting_radius) ,
                                 int(0.5*source_bkg_free.shape[0] - fitting_radius):int(0.5*source_bkg_free.shape[0] + fitting_radius) ]
           
        psf_residual_model.set_param_hint('A',
                                        value = 0.5 * np.nanmax(source),
                                        min =1e-6,
                                        max = 1.5*np.nanmax(abs(source)))
             
        psf_pars = psf_residual_model.make_params()

        import warnings
        with warnings.catch_warnings():
                    
            warnings.simplefilter("ignore")
        
                    
            result = psf_residual_model.fit(data = source,
                                                params = psf_pars,
                                                x = np.ones(source.shape),
                                                method = autophot_input['fitting_method'],
                                                nan_policy = 'omit',
                                                # weights = np.log10(source)
                                                )
            
        xc = result.params['x0'].value
    
        yc = result.params['y0'].value
    
        H_psf = result.params['A'].value
        
        fitted_PSF = PSF_MODEL(xc,yc,0,H_psf,residual_table,autophot_input)
        
        subtracted_image = source_bkg_free - fitted_PSF
        
        import matplotlib.pyplot as plt
        
        
        
        fig = plt.figure(figsize = set_size(250,1))
        
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
        image_source = ax1.imshow(source_bkg_free,
                   origin = 'lower')
        
        image_psf = ax2.imshow(fitted_PSF,
                   origin = 'lower')
        
        image_residual = ax3.imshow(subtracted_image,
                   origin = 'lower')
        
        cbar_ax1 = plt.colorbar(image_source,ax = [ax1])
        cbar_ax2 = plt.colorbar(image_psf,ax = [ax2])
        cbar_ax3 = plt.colorbar(image_residual,ax = [ax3])
        
        ax1.set_title('PSF model source')
        ax2.set_title('Fitted PSF')
        ax3.set_title('Residual')
        
        for ax in [ax1,ax2,ax3]:
            ax.scatter(source_bkg_free.shape[1]/2,source_bkg_free.shape[0]/2,
                       marker = '+',
                       color = 'black',
                       label = 'Image Center')
            ax.scatter(xc,yc,
                       marker = 'o',facecolor = 'none',edgecolor = 'red',label = 'Image Center')
        
        
        for cbar in [cbar_ax1,cbar_ax2,cbar_ax3]:
            cbar.set_label('Counts')
        
        save_loc = os.path.join(autophot_input['write_dir'],'TEST_PSF')

        os.makedirs(save_loc, exist_ok=True)
        fig.savefig(os.path.join(save_loc,'TEST_PSF_MODEL_%d.pdf' % idx),
                     bbox_inches='tight')

         # logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))

        plt.close(fig)


    return None 
    


# =============================================================================
# Check to see if update residual table will improve phototmetry
# =============================================================================


def update_residual_table(image,construction_sources,r_table,autophot_input,iterations = 10):
    
    
    '''
    Test to update PSF residual table 
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    

    import pandas as pd
    import lmfit
    import logging

    import os
    import warnings
    import matplotlib.pyplot as plt
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    
    from photutils import DAOStarFinder
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    
    from autophot.packages.functions import scale_roll
    from autophot.packages.functions import rebin
    from autophot.packages.background import remove_background
    from autophot.packages.functions import pix_dist,gauss_sigma2fwhm
    from autophot.packages.uncertain import SNR
    from autophot.packages.aperture  import measure_aperture_photometry
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    
    
    from lmfit import Model
    import numpy as np
    import os
    
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    if autophot_input['use_moffat']:
        
        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm
        
    
    # test model on same PSF stars that it was built from and return plot 
    fwhm = autophot_input['fwhm']
    
    fitting_radius = int(np.ceil(autophot_input['fitting_radius'] * fwhm))
        
    slice_scale = fitting_radius
    pad_shape = None
    
    regriding_size = int(autophot_input['regrid_size'])
    
    dx = autophot_input['dx']
    dy = autophot_input['dy']
    
    dx_vary = True
    dy_vary = True
    
    updated_residual_table =  np.repeat(np.repeat(r_table, regriding_size, axis=0), regriding_size, axis=1)
    
    
    # Define the PSF residual model
    
    x_slice = np.arange(0,2*fitting_radius)
    
    xx_sl,yy_sl = np.meshgrid(x_slice,x_slice)
    
    # x = np.arange(0,2*autophot_input['scale'])
    xx,yy = np.meshgrid(np.arange(0,2*autophot_input['scale']),np.arange(0,2*autophot_input['scale']))
    
    
    
    for iteration in range(iterations):
        
        r_table_iter = []
        
        def psf_residual(x,x0,y0,A):
    
            res =  PSF_MODEL(x0,y0,0,A,
                             r_table,
                             autophot_input,
                             slice_scale = slice_scale,
                             pad_shape = pad_shape)
                                    
            return res.flatten()
    
    
        psf_residual_model = Model(psf_residual)
        
        psf_residual_model.set_param_hint('x0',
                                          vary = dx_vary,
                                          value = 0.5*r_table.shape[1],
                                          min   = 0.5*r_table.shape[1]-dx,
                                          max   = 0.5*r_table.shape[1]+dx)
        
        # 0.5*residual_table.shape[1]+dx
    
        psf_residual_model.set_param_hint('y0',
                                          vary = dy_vary,
                                          value = 0.5*r_table.shape[0],
                                          min   = 0.5*r_table.shape[0]-dy,
                                          max   = 0.5*r_table.shape[0]+dy)
        
        for idx in construction_sources.index:
            

            
            psf_image = image[int(construction_sources.y_pix[idx]-autophot_input['scale']): int(construction_sources.y_pix[idx]+autophot_input['scale']),
                              int(construction_sources.x_pix[idx]-autophot_input['scale']): int(construction_sources.x_pix[idx]+autophot_input['scale'])]
            
            psf_image_bkg_free,bkg_surface,background,noise = remove_background(psf_image,
                                                    autophot_input,
                                                    psf_image.shape[1]/2,
                                                    psf_image.shape[0]/2)
    
            psf_image_slice = psf_image_bkg_free[int(psf_image_bkg_free.shape[1]/2 - fitting_radius):int(psf_image_bkg_free.shape[1]/2 + fitting_radius) ,
                                                 int(psf_image_bkg_free.shape[0]/2 - fitting_radius):int(psf_image_bkg_free.shape[0]/2 + fitting_radius) ]
    
            import warnings
            
            psf_residual_model.set_param_hint('A',
                                               value = 0.75 * np.nanmax(psf_image_slice),
                                               min = 1e-6,
                                               max = 1.5*np.nanmax(abs(psf_image_slice)))
             
            psf_pars = psf_residual_model.make_params()
            
            with warnings.catch_warnings():
                        
                warnings.simplefilter("ignore")
            
                        
                result = psf_residual_model.fit(data = psf_image_slice,
                                                params = psf_pars,
                                                x = np.ones(psf_image_slice.shape),
                                                method = autophot_input['fitting_method'],
                                                nan_policy = 'omit',
                                                    # weights = np.log10(source)
                                                    )
                # print(result.fit_report())
                
            xc = result.params['x0'].value 
            yc = result.params['y0'].value 
        
            H_psf = result.params['A'].value
            
            # - fitting_radius +  autophot_input['scale']
            
            
            xc_correction =  xc - fitting_radius + autophot_input['scale']
            yc_correction =  yc - fitting_radius + autophot_input['scale']
            
            fitted_PSF = PSF_MODEL(xc,yc,0,H_psf,r_table,autophot_input)
            
            residual = psf_image_bkg_free - fitted_PSF
                
            # psf_mag.append(PSF_flux)
            
            residual = residual / H_psf
            
            residual_counts_before = np.sum(residual)
    
            residual_regrid = np.repeat(np.repeat(residual, regriding_size, axis=0), regriding_size, axis=1)
    
            x_roll = scale_roll(autophot_input['scale'],xc,regriding_size)
            y_roll = scale_roll(autophot_input['scale'],yc,regriding_size)
            # print(x_roll,y_roll)
    
            residual_roll = np.roll(np.roll(residual_regrid,y_roll,axis=0),x_roll,axis = 1)

            r_table_iter.append(residual_roll)
            # residual_table.append(residual_roll)
            
            residual_counts_after = np.sum(rebin(residual_roll, (int(2*autophot_input['scale']),int(2*autophot_input['scale']))))
            # end subtraction
            from autophot.packages.functions import set_size
            fig = plt.figure(figsize = set_size(500,1))
        
            ax1 = fig.add_subplot(511)
            ax2 = fig.add_subplot(512)
            ax3 = fig.add_subplot(513)
            ax4 = fig.add_subplot(514)
            ax5 = fig.add_subplot(515)
            
            image_source = ax1.imshow(psf_image_bkg_free,
                       origin = 'lower')
            
            image_psf = ax2.imshow(fitted_PSF,
                       origin = 'lower')
            
            image_residual = ax3.imshow(residual*H_psf,
                       origin = 'lower')
            
            new_residual_regrid = ax4.imshow(residual_regrid,
                       origin = 'lower')
            
            new_residual = ax5.imshow(residual_roll,
                       origin = 'lower')
            
            # cbar_ax1 = plt.colorbar(image_source,ax = [ax1])
            # cbar_ax2 = plt.colorbar(image_psf,ax = [ax2])
            # cbar_ax3 = plt.colorbar(image_residual,ax = [ax3])
            
            # ax1.set_title('PSF model source')
            # ax2.set_title('Fitted PSF')
            # ax3.set_title('Residual')
            
            for ax in [ax1,ax2,ax3]:
                ax.scatter(psf_image_bkg_free.shape[1]/2,psf_image_bkg_free.shape[0]/2,
                            marker = '+',
                            color = 'black',
                            label = 'Image Center')
                ax.scatter(xc,yc,
                            marker = 'o',facecolor = 'none',edgecolor = 'red',label = 'Image Center')
            
            
            # for cbar in [cbar_ax1,cbar_ax2,cbar_ax3]:
            #     cbar.set_label('Counts')
            
            save_loc = os.path.join(autophot_input['write_dir'],'TEST_PSF')
    
            os.makedirs(save_loc, exist_ok=True)
            fig.savefig(os.path.join(save_loc,'TEST_PSF_MODEL_%d_iter_%d.pdf' % (idx, iteration)),
                        bbox_inches='tight')
            plt.close()

         # logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))

        # end iteration
        r_table_iter_median = np.nanmedian(np.dstack(r_table_iter),axis = -1)
    
        
        updated_residual_table += r_table_iter_median
        
        r_table  = rebin(updated_residual_table, (int(2*autophot_input['scale']),int(2*autophot_input['scale'])))
        fig = plt.figure(figsize = set_size(500,1))
        
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        
        ax1.imshow(r_table_iter_median,
                       origin = 'lower')
        
        ax2.imshow(updated_residual_table,
                       origin = 'lower')
        
        ax3.imshow(r_table,
                       origin = 'lower')
            
        
        os.makedirs(save_loc, exist_ok=True)
        fig.savefig(os.path.join(save_loc,'NEW_RTABLE_iter_%d.pdf' % ( iteration)),
                    bbox_inches='tight')
        plt.close()
        

    
    return r_table

# =============================================================================
# Fit the PSF model
# =============================================================================
def fit(image,
        sources,
        residual_table,
        autophot_input,
        return_psf_model = False,
        save_plot = False,
        show_plot = False,
        remove_background_val = True,
        hold_pos = False,
        return_fwhm = False,
        return_subtraction_image = False,
        no_print = False,
        return_closeup = False,
        # cutout_base = True
        ):
    
    
    '''
    PSF fitting packages.
    
    This package uses the PSF model created in psf.build to fit for every source in an image. Can return a plot of the subtracted image for the User 
    :param image: 2D image of containg stars with which the user wishes the measure
    :type image: numpy array
    :param sources: DataFarme of sources containing location of sources
    :type sources: pandas DataFrame
    :param residual_table: PSF residual table from psf.build
    :type residual_table: numpy array
    :param autophot_input: Autophot control dictionary
    :type autophot_input: dict
    
    :Required autophot_input:
        - **fwhm** (*float*): FWHM to use with PSF model. Must be the same FWHM used in building the PSF model
        - **regrid_size** (*int*): rebin size for residual image shift
        - **use_moffat** (*boolean*): Use a moffat function as the base analytical function for the PSf model
        - **image_params** (*dict*): dictionary containing values of analytical function. if 'use_moffat' dictionary should contain 'alpha' and 'beta' keys and their values, else dictionary should contain 'sigma' and its value
        - **scale** (*float*): size of cutout around a specific target position (image shape = [2*scale,2*scale])
        - **dx**/**dx** (*float*/*float*): if hold_pos is False, allow PDF model to move by dx/dy during fitting
        - **remove_sat* (**boolean**): if True, remove any sources brighter than the count level set using *sat_lvl*
        
    :param return_psf_model: If True, return the PSF model, defaults to False
    :type return_psf_model: boolean, optional
    :param save_plot: Save plot of PSF subtraction, defaults to False
    :type save_plot: boolen, optional
    :param show_plot: Show plot of PSF subtraction, defaults to False
    :type show_plot: boolen, optional
    :param remove_background_val: if False, do not perform background subtraction to source proir to fitting PSF model, defaults to True
    :type remove_background_val: boolean, optional
    :param hold_pos: if True, Force PSF to fit source at percise location given in "sources" dataframe. IF false, PSF model is allowed to fit with a pixel tolerence given by 'dx'/'dy' given in autophot_input, defaults to False
    :type hold_pos: boolean, optional
    :param return_fwhm: If true return the FWHM of the source proir to PSF fitting, defaults to False
    :type return_fwhm: boolean, optional
    :param return_subtraction_image:If true, return and save an image of the PSF subtraction, defaults to False
    :type return_subtraction_image: boolean, optional
    :param no_print: If true, do not print any fitting information from this package, defaults to False
    :type no_print: boolean, optional
    :return: 
        - **PSF_sources** (*pandas DataFrame*): Dataframe containing output parameters from PSF fitting. Use with psf.do to compute instrumental magnitudes
        - **PSF_MODEL**  (*function*): PSF model

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
    from autophot.packages.functions import scale_roll,rebin,gauss_fwhm2sigma,set_size,order_shift
    from matplotlib.gridspec import  GridSpec
    
    from autophot.packages.background import remove_background
    
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    logger = logging.getLogger(__name__)
    
    fwhm = autophot_input['fwhm']
    
    fitting_radius = int(np.ceil(autophot_input['fitting_radius'] * fwhm))
    regriding_size = int(autophot_input['regrid_size'])

    # if return_psf_model:

    #     shape = int(2*autophot_input['scale']),int(2*autophot_input['scale'])
    #     x_slice = np.arange(0,shape[0])
    #     xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

    #     if autophot_input['use_moffat']:
    #         PSF_model = moffat_2d((xx_sl,yy_sl),shape[1]/2,shape[1]/2,0,1,dict(alpha=autophot_input['image_params']['alpha'],beta=autophot_input['image_params']['beta'])).reshape(shape)
    #     else:
    #         PSF_model= gauss_2d((xx_sl,yy_sl),shape[1]/2,shape[1]/2,0,1,dict(sigma=autophot_input['image_params']['sigma'])).reshape(shape)

    #     return PSF_model

    psf_params = []

    xx,yy= np.meshgrid(np.arange(0,image.shape[1]),np.arange(0,image.shape[0]))

    dx = autophot_input['dx']
    dy = autophot_input['dy']

    lower_x_bound = autophot_input['scale']
    lower_y_bound = autophot_input['scale']
    upper_x_bound = autophot_input['scale']
    upper_y_bound = autophot_input['scale']

    if return_subtraction_image:

        from astropy.visualization import  ZScaleInterval

        vmin,vmax = (ZScaleInterval(nsamples = 300)).get_limits(image)
   
    if return_subtraction_image:
        image_before = image.copy()
        
        
    # How is the PSF model allowed to move around
    if hold_pos:
        dx_vary = False
        dy_vary = False
        
        dx = 1
        dy = 1
    else:
        dx_vary = True
        dy_vary = True
        
        dx = autophot_input['dx']
        dy = autophot_input['dy']
    
    slice_scale = fitting_radius
    pad_shape = None
        
                
    def psf_residual(x,x0,y0,A):

        res =  PSF_MODEL(x0,y0,0,A,
                            residual_table,
                            autophot_input,
                            slice_scale = slice_scale,
                            pad_shape = pad_shape)
                                
        return res.flatten()
    
    
    psf_residual_model = Model(psf_residual)
    
    psf_residual_model.set_param_hint('x0',
                                      vary = dx_vary,
                                      value = 0.5*residual_table.shape[1],
                                      min   = 0.5*residual_table.shape[1]-dx,
                                      max   = 0.5*residual_table.shape[1]+dx)
    
    # 0.5*residual_table.shape[1]+dx

    psf_residual_model.set_param_hint('y0',
                                      vary = dx_vary,
                                      value = 0.5*residual_table.shape[0],
                                      min   = 0.5*residual_table.shape[0]-dy,
                                      max   = 0.5*residual_table.shape[0]+dy)

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
        
        if not return_fwhm and not no_print:
            
            print('\rFitting PSF to source: %d / %d ' % (n+1,len(sources)), end = '')

        try:

            idx = list(sources.index)[n]
         
            xc_global = sources.x_pix[idx]
            yc_global = sources.y_pix[idx]
            
            source_base =   image[int(yc_global-lower_y_bound): int(yc_global + upper_y_bound),
                                  int(xc_global-lower_x_bound): int(xc_global + upper_x_bound)]
            
            
            source_base_median = np.nanmedian(source_base)
            
            xc = source_base.shape[1]/2
            yc = source_base.shape[0]/2
            
            if not remove_background_val:
                
                print('here')
                
                source_bkg_free = source_base
                bkg_surface = np.ones(source_base.shape)
                bkg_median = np.nanmedian(source_base)
                
                # print('->',np.nanmedian(source_base) )
                
            else:

                try:

                    source_bkg_free, bkg_surface, bkg_median,noise = remove_background(source_base,autophot_input)
                    
                except Exception as e:
                    
                    print('cannot fit background')

                    psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H,H_psf_err,max_pixel,chi2,redchi2))

                    logger.exception(e)
                    
                    continue
                
            # if return_fwhm:
            #     print('bkg_median: %6.f' % bkg_median)

            source = source_bkg_free[int(0.5*source_bkg_free.shape[1] - fitting_radius):int(0.5*source_bkg_free.shape[1] + fitting_radius) ,
                                     int(0.5*source_bkg_free.shape[0] - fitting_radius):int(0.5*source_bkg_free.shape[0] + fitting_radius) ]
            
            
            max_pixel = np.nanmax(source)
            
            if source.shape != (int(2*fitting_radius),int(2*fitting_radius)) or np.sum(np.isnan(source)) == len(source):
            
             

                psf_params.append((idx,x_fitted,y_fitted,bkg_median,noise,H,H_psf_err,max_pixel,chi2,redchi2))

                continue



            # Go ahead and fit a PSF
            try:

                # Update params with amplitude in cutout
                psf_residual_model.set_param_hint('A',
                                                  value = 1.5 * np.nanmedian(source),
                                                  min = 1e-6,
                                                  max = 1.5*np.nanmax(source))
                
                

                psf_pars = psf_residual_model.make_params()
                

                import warnings
                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
        
                    
                    result = psf_residual_model.fit(data = source,
                                                    params = psf_pars,
                                                    x = np.ones(source.shape),
                                                    method = autophot_input['fitting_method'],
                                                    nan_policy = 'omit',
                                                    # weights = np.log10(source)
                                                    )
                    

                xc = result.params['x0'].value

                yc = result.params['y0'].value

                H_psf = result.params['A'].value
                H_psf_err = result.params['A'].stderr
      
                chi2 = result.chisqr
                redchi2 = result.redchi

                x_fitted = xc - residual_table.shape[1]/2 + xc_global
                y_fitted = yc - residual_table.shape[0]/2 + yc_global
                
                x_fitted_shape = xc - residual_table.shape[1]/2 + fitting_radius 
                y_fitted_shape = yc - residual_table.shape[0]/2 + fitting_radius 
                
        
                
                if return_fwhm:
                    if not no_print:
                        logger.info('\nFitting function to source to get FWHM')
                        
                    # print(x_fitted_shape,y_fitted_shape,fitting_radius)
                        
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
                    
                    if autophot_input['use_moffat']:
                        pars.add('alpha',value = autophot_input['image_params']['alpha'],
                                 min = 0,
                                 max = 25)
    
                        pars.add('beta',value = autophot_input['image_params']['beta'],
                                 min = 0,
                                 vary = autophot_input['vary_moff_beta']  )
    
                    else:
                        pars.add('sigma',value = autophot_input['image_params']['sigma'],
                                 min = 0,
                                 max = gauss_fwhm2sigma(autophot_input['max_fit_fwhm']) )
    
                    if autophot_input['use_moffat']:
                        
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
                        
                        from lmfit import fit_report
                        
                        result = mini.minimize(method = autophot_input['fitting_method'] )
                        

                    # This needs to be in the scale of the closeup image and not the overall image
                    FWHM_fitted_xc = result.params['x0'].value - fitting_radius + lower_x_bound
                    FWHM_fitted_yc = result.params['y0'].value - fitting_radius + lower_y_bound

                    if autophot_input['use_moffat']:
                        target_PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                    else:
                        target_PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))
    
                    if not no_print:
                        logger.info('Target FWHM: %.3f [ pixels ]\n' % target_PSF_FWHM)
               
                
                
                if autophot_input['remove_sat'] and not return_fwhm:

                    if H_psf+bkg_median >= autophot_input['sat_lvl']:
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

                    image_section_subtraction = image_section - PSF_MODEL(xc , yc, 0, H, residual_table,autophot_input,slice_scale = image_section.shape[0]/2)

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

                    save_loc = os.path.join(autophot_input['write_dir'],'cleaned_images')

                    os.makedirs(save_loc, exist_ok=True)

                    fig.savefig(os.path.join(save_loc,'subtraction_%d.pdf' % idx),
                                bbox_inches='tight')

                    logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')
                    pass

            if autophot_input['plot_PSF_residuals'] or show_plot == True or save_plot == True:

                try:
                    from astropy.visualization import  ZScaleInterval

                    fitted_source = PSF_MODEL(xc,yc,0,H_psf,residual_table,autophot_input)
                    subtracted_image = source_bkg_free - fitted_source + bkg_surface

                    scale = order_shift(abs(source_base))

                    source_base = source_base/scale
                    subtracted_image=subtracted_image / scale
                    bkg_surface = bkg_surface/ scale
                    fitted_source = fitted_source/ scale

                    vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(source_base)

                    h, w = source_bkg_free.shape

                    x  = np.linspace(0, int(2*autophot_input['scale']), int(2*autophot_input['scale']))
                    y  = np.linspace(0, int(2*autophot_input['scale']), int(2*autophot_input['scale']))

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
                               vmax = vmax
                               )

         

                    ax1_R.step(source_base[:,w//2],Y[:,w//2],color = 'blue',label = '1D projection',where='mid')
                    ax1_B.step(X[h//2,:],source_base[h//2,:],color = 'blue',where='mid')

                    # include surface
                    ax1_R.step(bkg_surface[:,w//2],Y[:,w//2],color = 'red',label = 'Background Fit',where='mid')
                    ax1_B.step(X[h//2,:],bkg_surface[h//2,:],color = 'red',where='mid')

                    # include fitted_source
                    ax1_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],color = 'green',label = 'PSF')
                    ax1_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],color = 'green')

                    ax1_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax1_R.set_xlabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax2_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax2_R.set_xlabel('Counts [$10^{%d}$] ' % np.log10(scale))

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

                    if autophot_input['plot_PSF_residuals']:
                    
                        pathlib.Path(autophot_input['write_dir']+'/'+'psf_subtractions/').mkdir(parents = True, exist_ok=True)
                        
                        save_name = lambda x: autophot_input['write_dir']+'psf_subtractions/'+'psf_subtraction_{}.png'.format(int(x))
                        i = 0
                        while True:
                            if not os.path.exists(save_name(n+i)):
                                break
                            else:
                                i+=1
                        
                        plt.savefig(save_name(n+i),bbox_inches='tight')
                        
                    else:
                        
                        fig.savefig(autophot_input['write_dir']+'target_psf_'+autophot_input['base']+'.pdf',bbox_inches='tight')

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')

        except Exception as e:
             logger.exception(e)

             
             psf_params.append((idx,x_fitted,y_fitted,xc,yc,bkg_median,noise,H_psf,H_psf_err,max_pixel,chi2,redchi2))

             continue
    # print(' ... done')

    if autophot_input['plot_before_after'] and return_subtraction_image:

        print('Saving before and after image')

        plt.ioff()

        fig = plt.figure(figsize = set_size(500,1))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image_before,
                    origin = 'lower',
                    # aspect="auto",
                    vmin = vmin,
                    vmax = vmax,
                    cmap='gray',
                    interpolation = 'nearest'
                    )

        ax2.imshow(image,
                    origin = 'lower',
                    # aspect="auto",
                    vmin = vmin,
                    vmax = vmax,
                    cmap='gray',
                    interpolation = 'nearest'
                    )

        for ax in fig.axes:
            ax.set_xlim(0,image.shape[0])
            ax.set_ylim(0,image.shape[1])
            ax.set_xlabel('X Pixel')
            ax.set_xlabel('Y Pixel')

        ax1.set_title('Original Image')
        ax2.set_title('Point source Free image [%d]' % autophot_input['do_all_phot_sigma'])

        plt.savefig(autophot_input['write_dir']+'PSF_BEFORE_AFTER.pdf',
                    bbox_inches='tight')

        plt.close(fig)

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












def do(df,residual,autophot_input,fwhm):

    """ Function to perform PSF model measurment. converts height of PSF model to counts under the PSF within a defined aperature size.

    :param df: dataframe contain amplitude and amplitude error (in counts) of PSF from *psf.fit* function
    :type df: Pandas.DataFrame

    :param residual: 2D image of residual table
    :type residual: np.array

    :param autophot_input: AutoPhot control dictionary
    :type autophot_input: dict
    
    Required autophot_input keywords:
        
        - **use_moffat** (*boolean*): Use a moffat function as the base analytical function for the PSf model
        - **image_params** (*dict*): dictionary containing values of analytical function. if 'use_moffat' dictionary should contain 'alpha' and 'beta' keys and their values, else dictionary should contain 'sigma' and its value
        - **scale** (*float*): size of cutout around a specific target position (image shape = [2*scale,2*scale])
        - **ap_size** (*float*): Aperture size in pixels
        
    :param FWHM: Full Width Half Maximum of image used to set integration radius
    :type FWHM: float

    :return: 
        - **df** (*pandas DataFrame*) Original DataFrame with *psf_counts* and *psf_counts_err* columns
        - **autophot_input** (*dict*) Updated autophot_input file 
    """

    try:
        from photutils import CircularAperture
        from photutils import aperture_photometry
        from scipy.integrate import dblquad
        import logging
        import numpy as np
        
        from autophot.packages.functions import gauss_2d,moffat_2d

        logger = logging.getLogger(__name__)
        
        # TODO: this is a waste - move it somewhere else
        if 'c_counts' not in autophot_input or 'r_counts' not in autophot_input:

            xc = autophot_input['scale']
            yc = autophot_input['scale']
    
            # Integration radius
            int_scale = autophot_input['ap_size'] * fwhm

            x_rebin = np.arange(0,2*autophot_input['scale'])
            y_rebin = np.arange(0,2*autophot_input['scale'])
            
            xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)

            if autophot_input['use_moffat']:
                
                core = moffat_2d((xx_rebin,yy_rebin),xc,yc,0,1,autophot_input['image_params']).reshape(residual.shape)
                
            else:
                
                core =  gauss_2d((xx_rebin,yy_rebin),xc,yc,0,1,autophot_input['image_params']).reshape(residual.shape)

            unity_PSF_model = core + residual
            
            # Aperture Photometry over residual
            apertures = CircularAperture((xc,yc), r=int_scale)
            
            
            residual_table = aperture_photometry(residual, 
                                                 apertures,
                                                 method='subpixel',
                                                 subpixels=6)
            residual_int = residual_table[0]
            
            PSF_table = aperture_photometry(unity_PSF_model, 
                                             apertures,
                                             method='subpixel',
                                             subpixels=6)
            PSF_int =  PSF_table[0]
                        
            # Counts from PSF model
            autophot_input['c_counts'] = float(PSF_int['aperture_sum'])
    
            # Counts from Residual component of PSF
            autophot_input['r_counts'] = float(residual_int['aperture_sum'])
        
            logger.info('Unity PSF: %.1f [counts] ' % autophot_input['c_counts'])
            logger.info('Unity Residual table: %.1f [counts] ' % autophot_input['r_counts'])

        unity_psf = float(autophot_input['c_counts']) 
        
        # print(df)

        df['psf_counts']     = df.H_psf.values   * unity_psf
        
        try:
            df['psf_counts_err'] = df.H_psf_err.values * unity_psf
        except:
            df['psf_counts_err'] = [np.nan] * len(df.H_psf_err.values)
            

    except Exception as e:
        logger.exception(e)
        df = np.nan

    return df,autophot_input








def compute_multilocation_err(image,autophot_input,
                              xfit = None,
                              yfit = None,
                              Hfit = None,
                              MODEL = None,
                              r_table = None):
    
    import numpy as np
    import pandas as pd
    from autophot.packages import psf
    

    # Number of times to tagret's PSF is injected and recovered
    N = autophot_input['target_error_compute_multilocation_number']
    
    # Multiples of FWHM with which to place this sources randomly e.g. 1 -> +/- FWHM/2
    position_shift = autophot_input['target_error_compute_multilocation_position']
    
    ran_dx = xfit + (np.random.uniform(-1,1,N) * position_shift * autophot_input['fwhm']/2 )
    ran_dy = yfit + (np.random.uniform(-1,1,N) * position_shift * autophot_input['fwhm']/2 )

    injection_df = pd.DataFrame([list(ran_dx ),list(ran_dy)])
    injection_df = injection_df.transpose()
    injection_df.columns = ['x_pix','y_pix']
    injection_df.reset_index(inplace = True,drop = True)
    
    Fitted_PSF = MODEL(xfit, yfit, 0, Hfit, r_table, autophot_input,pad_shape = image.shape)

    # Remove PSF from image
    residual_image = image - Fitted_PSF
    
    hold_psf_position = False
    
    magnitudes_recovered = []
    
    for i in range(len(injection_df)):
        
        print('\rComputing error for target location: %d / %d' % (i+1,len(injection_df) ) ,
              end = '',
              flush = True)
        
        test_PSF = MODEL(injection_df['x_pix'].values[i],injection_df['y_pix'].values[i],0,Hfit, 
                             r_table,
                             autophot_input,
                             pad_shape = image.shape)
        
        

        
        psf_fit = psf.fit(residual_image + test_PSF,
                            injection_df.iloc[[i]], 
                            r_table,
                            autophot_input,
                            # return_fwhm = True,
                            no_print = True,
                            hold_pos = hold_psf_position,
                            save_plot = False,
                            # remove_background_val = True
                            )
        
        psf_params,_ = psf.do(psf_fit,
                              r_table,
                              autophot_input,
                              autophot_input['fwhm'])
        
        psf_counts = psf_params['psf_counts'].values
        
        magnitudes_recovered.append(-2.5*np.log10(psf_counts/autophot_input['exp_time'])[0])
      
    error = np.nanstd(magnitudes_recovered)
    
    
    print('\nError from %d measuresments: %.3f' % (N,error))              
    
    return error