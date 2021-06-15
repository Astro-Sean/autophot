def build_r_table(base_image,selected_sources,syntax,fwhm):
    '''
    Create residual table for use in Point Spread Function photometry model
    
    
    :param base_image: 2D image containing bright isolated stars 
    :type base_image: numpy array
    :param selected_sources: DataFrame containing location and approximate flux of bright isolated stars
    
    :type selected_sources: pandas DataFrame
    
    Dataframe must contain the following information:
        
        - x_pix/y_pix (*float*/*float*): pixel coordinates of selected sources
        - flux_ap** (*float*): approximate flux of source - not needed if given PSF star list
    
    :param syntax: syntax file
    :type syntax: dict
    
    :Required syntax keywords for this package:
        
        - **fwhm** (*float*): Full width half maximum of image
        - **regrid_size** (*int*): rebin size for residual image shift
        - **scale** (*float*): size of cutout around a specific target position (image shape = [2*scale,2*scale])
        - **use_PSF_starlist** (*boolean*): if true, use stars given by "'PSF_starlist' filepath. This list should be read in as a DataFrame and have the headers x_pix/y_pix corresponding to their position in the image
        - **prepapre_templates:** (*boolean*): if true, prepare and return PSF of template images
        - **use_local_stars_for_psf** (*boolean*) : Use source near the target for PSF stars.
        - **local_radius** (*float*): distance in arcmin around target to use for PSF model stars
        - **PSF_source_no** (*int*): Number of sources to build the model with
        - **min_PSF_source_no** (*int*): minimum amount of sources that are acceptable for a PSF model, if there isn't enough sources, aperture photometry is used
        - **construction_SNR** (*float*): Minimum SNR a star can have to be acceptable and used in building the PSF model
        - **bkg_level** (*float*): signal to noise level with which below is considered background noise
        - **use_DAOFIND** (*boolean*): use DAOFIND to search for point sources
        - **use_imageseg** (*boolean*): use image segmentation for point source detection
        - **use_moffat** (*boolean*): Use a moffat function as the base analytical function for the PSf model
        - **image_params** (*dict*): dictionary containing values of analytical function. if 'use_moffat' dictionary should contain 'alpha' and 'beta' keys and their values, else dictionary should contain 'sigma' and its value
        - **vary_moff_beta/fit_PSF_fwhm (boolean/boolean): if True, allow for FWHM fitting during PSF creation
        - **fwhm** (*float*): Full width half maximum of image in pixels
        - **gain** (*float*): Gain of image in e/ADU
        - **ap_size** (*float*): Aperture size in pixels
        - **r_in_size** (*float*): Radius of inner annulus for aperture photometry 
        - **r_out_size** (*float*): Radius of inner annulus for aperture photometry
        
        - **plots_PSF_residuals** (*boolean*): Save image of PSF residuals when building the PSF 
        - **plots_PSF_source** (*boolean*): Save image of grid of PSF stars
        - **save_PSF_models_fits** (*boolean*): Save PSF model as normalizated fits file
        
        - **write_dir** (*str*): Working directory where plots will be saved
        - **base**** (*str*): basename of file, used for saving plots
        
    :param fwhm: Full Width Half Maximum
    :type fwhm: float
    :return: 
        - **residual_table** (*numpy array*): 2D image of PSF residual table to be used in PSF model
        - **fwhm_fit** (*float*): FWHM of PSF sources if fitted, else returns image FWHM
        - **construction_sources** (*pandas dataframe*): DataFrame containing parameters on sources used to make PSF model
        - **syntax** (*dict*): update syntax file

    '''
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder

    import pandas as pd
    import lmfit
    import logging

    from autophot.packages.functions import pix_dist,gauss_sigma2fwhm
    from autophot.packages.uncertain import SNR
    from autophot.packages.aperture  import ap_phot

    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma

    from autophot.packages.functions import moffat_2d,moffat_fwhm
    import os

    import warnings
    
    import matplotlib.pyplot as plt

    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    from autophot.packages.functions import scale_roll
    
    from autophot.packages.functions import rebin
    
    from autophot.packages.rm_bkg import rm_bkg

    
    warnings.simplefilter(action='ignore', category=FutureWarning)


    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))



    if syntax['use_moffat']:
        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm



    try:

        logger = logging.getLogger(__name__)

        image = base_image.copy()

        # Only fit to a small image with radius ~the fwhm
        fitting_radius = int(np.ceil(fwhm))

        # for matchinf each source residual image with will regrid the image for shifting later
        regriding_size = int(syntax['regrid_size'])

        if regriding_size % 2 > 0:
            logger.info('regrid size must be even adding 1')
            regriding_size += 1

        # FWHM/sigma fits
        fwhm_fit = []

        # what sources will be used
        construction_sources = []

        # Residual Table in extended format
        residual_table = np.zeros((int(2 * syntax['scale'] * regriding_size), int(2 * syntax['scale']*regriding_size)))

        if syntax['use_local_stars_for_PSF'] and not syntax['prepare_templates']:

            '''
            Use local stars given by 'use_acrmin' parameter

            '''
            selected_sources_test = selected_sources[selected_sources['dist'] <= syntax['local_radius']]

            selected_sources  = selected_sources_test
            
        if not syntax['use_PSF_starlist']:
            
            flux_idx = [i for i in selected_sources.flux_ap.sort_values(ascending = False).index]
        else:
            
            flux_idx = selected_sources.index

        sources_used = 1
        n = 0
        failsafe = 0
        psf_mag = []
        image_radius_lst = []
        sources_dict = {}


        while sources_used <= syntax['psf_source_no']:

            if failsafe>25:
                logger.info('PSF - Failed to build psf')
                residual_table=None
                fwhm_fit = fwhm


            if n >= len(flux_idx):
                if sources_used  >= syntax['min_psf_source_no']:
                    logger.info('Using worst case scenario number of sources')
                    break
                logger.info('PSF - Ran out of sources')
                residual_table=None
                fwhm_fit = fwhm
                break
            try:

                idx = flux_idx[n]

                n+=1

                # Inital guess at where psf source is is
                psf_image = image[int(selected_sources.y_pix[idx]-syntax['scale']): int(selected_sources.y_pix[idx] + syntax['scale']),
                                  int(selected_sources.x_pix[idx]-syntax['scale']): int(selected_sources.x_pix[idx] + syntax['scale'])]

                if len(psf_image) == 0:
                    logger.info('PSF image ERROR')
                    continue
                try:
                    if np.min(psf_image) == np.nan:
                        continue
                except:
                    continue


                mean, median, std = sigma_clipped_stats(psf_image,
                                                        sigma = syntax['bkg_level'],
                                                        maxiters = 10)


                if syntax['use_daofind']:

                    daofind = DAOStarFinder(fwhm      = np.ceil(fwhm),
                                            threshold = syntax['bkg_level']*std,
                                            sharplo   =  0.2,sharphi = 1.0,
                                            roundlo   = -1.0,roundhi = 1.0
                                            )

                    sources = daofind(psf_image - median)

                    if sources == None:
                        sources = []
                    else:
                        sources = sources.to_pandas()

                elif syntax['use_imageseg']:

                   from photutils import detect_threshold

                   threshold = detect_threshold(psf_image, nsigma=syntax['bkg_level'])

                   from astropy.convolution import Gaussian2DKernel
                   from astropy.stats import gaussian_fwhm_to_sigma
                   from photutils import detect_sources

                   sigma = fwhm * gaussian_fwhm_to_sigma

                   kernel = Gaussian2DKernel(sigma,
                                             # x_size=psf_image.shape[0],
                                             # y_size=psf_image.shape[1]
                                             )
                   kernel.normalize()

                   segm = detect_sources(psf_image,
                                         threshold,
                                         npixels=5,
                                         filter_kernel=kernel
                                         )

                   from photutils import deblend_sources
                   segm_deblend = deblend_sources(psf_image,
                                                  segm,
                                                  npixels=5,
                                                  filter_kernel=kernel
                                                  )

                   from photutils import SourceCatalog
                   props = SourceCatalog(psf_image, segm_deblend )

                   sources = props.to_table().to_pandas()



                if len(sources) > 1:


                    dist = [list(pix_dist(
                            sources['xcentroid'][i],
                            sources['xcentroid'],
                            sources['ycentroid'][i],
                            sources['ycentroid']) for i in range(len(sources)))]

                    dist = np.array(list(set(np.array(dist).flatten())))

                    if all(dist < 2):
                        pass
                    else:
                        continue



                psf_image = image[int(selected_sources.y_pix[idx]-syntax['scale']): int(selected_sources.y_pix[idx]+syntax['scale']),
                                  int(selected_sources.x_pix[idx]-syntax['scale']): int(selected_sources.x_pix[idx]+syntax['scale'])]

                psf_image_bkg_free,bkg_surface = rm_bkg(psf_image,
                                                        syntax,
                                                        psf_image.shape[0]/2,
                                                        psf_image.shape[1]/2)

                background = np.nanmean(bkg_surface)


                x = np.arange(0,2*syntax['scale'])
                xx,yy= np.meshgrid(x,x)

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmax(psf_image_bkg_free),min=0)
                pars.add('x0',value = psf_image_bkg_free.shape[1]/2,min = 0, max =psf_image_bkg_free.shape[1] )
                pars.add('y0',value = psf_image_bkg_free.shape[0]/2,min = 0, max =psf_image_bkg_free.shape[0])
                pars.add('sky',value = np.nanmedian(psf_image_bkg_free))

                if syntax['use_moffat']:
                    pars.add('alpha',value = syntax['image_params']['alpha'],
                             min = 0,
                             vary =  syntax['fit_PSF_FWHM'] )
                    pars.add('beta',value = syntax['image_params']['beta'],
                             min = 0,
                             vary = syntax['vary_moff_beta'] or syntax['fit_PSF_FWHM']  )

                else:
                    pars.add('sigma', value = syntax['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(syntax['max_fit_fwhm']),
                              vary = syntax['vary_moff_beta'] or syntax['fit_PSF_FWHM']
                             )


                if syntax['use_moffat']:
                    def residual(p):
                        p = p.valuesdict()
                        return (psf_image_bkg_free - moffat_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(psf_image_bkg_free.shape)).flatten()
                else:
                    def residual(p):
                      p = p.valuesdict()
                      return (psf_image_bkg_free - gauss_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(sigma=p['sigma'])).reshape(psf_image_bkg_free.shape)).flatten()

                    

                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
                    
                    mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)
                    
                    result = mini.minimize(method = 'least_squares')
                    
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                ap_range = np.arange(0.1,syntax['scale']/fwhm,1/25)
                ap_sum = []

                for nm in ap_range:

                    ap,bkg = ap_phot( [(xc,yc)] ,
                                 psf_image_bkg_free,
                                 radius = nm * fwhm,
                                 r_in = syntax['r_in_size'] * fwhm,
                                 r_out = syntax['r_out_size'] * fwhm)

                    ap_sum.append(ap)

                ap_sum = ap_sum/np.nanmax(ap_sum)

                radius = ap_range[np.argmax(ap_sum>=syntax['norm_count_sum'])]

                image_radius = radius * fwhm
                image_radius_lst.append(image_radius)
                
                '''
                Refit only focusing on highest SNR area given by fitting radius

                '''
                # global pixel coorindates base on bn gaussian fit
                xc_global = xc - syntax['scale'] + int(selected_sources.x_pix[idx])
                yc_global = yc - syntax['scale'] + int(selected_sources.y_pix[idx])

                # recenter image absed on location of best fit x and y
                psf_image = image[int(yc_global-syntax['scale']): int(yc_global + syntax['scale']),
                                  int(xc_global-syntax['scale']): int(xc_global + syntax['scale'])]

                psf_image_bkg_free,bkg_median = rm_bkg(psf_image,syntax,psf_image.shape[0]/2,psf_image.shape[0]/2)

                psf_image_slice = psf_image_bkg_free[int(psf_image_bkg_free.shape[0]/2 - fitting_radius):int(psf_image_bkg_free.shape[0]/2 + fitting_radius) ,
                                                     int(psf_image_bkg_free.shape[0]/2 - fitting_radius):int(psf_image_bkg_free.shape[0]/2 + fitting_radius) ]


                x_slice = np.arange(0,2*fitting_radius)
                xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmean(psf_image_slice),min = 0,max = np.nanmax(psf_image_slice) )
                pars.add('x0',value = psf_image_slice.shape[1]/2,min = 0,max = psf_image_slice.shape[1])
                pars.add('y0',value = psf_image_slice.shape[0]/2,min = 0,max = psf_image_slice.shape[0] )

                if syntax['use_moffat']:

                    pars.add('alpha',value = syntax['image_params']['alpha'],
                             min = 0,
                             vary =  syntax['fit_PSF_FWHM'] )

                    pars.add('beta',value = syntax['image_params']['beta'],
                             min = 0,
                             vary = syntax['vary_moff_beta'] or syntax['fit_PSF_FWHM']  )

                else:

                    pars.add('sigma', value = syntax['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(syntax['max_fit_fwhm']),
                             vary =  syntax['fit_PSF_FWHM']
                             )

                if syntax['use_moffat']:
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
                    
                    result = mini.minimize(method = 'least_squares')

                positions  = list(zip([xc_global],[yc_global]))

                psf_counts,psf_bkg = ap_phot(positions,
                                             image,
                                             radius = syntax['ap_size']    * fwhm,
                                             r_in   = syntax['r_in_size']  * fwhm,
                                             r_out  = syntax['r_out_size'] * fwhm)

                if syntax['use_moffat']:

                    PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:

                    PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))



                PSF_SNR = SNR(psf_counts,psf_bkg,syntax['exp_time'],0,syntax['ap_size']*fwhm,syntax['gain'],0)[0]

                if np.isnan(PSF_SNR) or np.isnan(PSF_FWHM):
                    logger.debug('PSF Contruction source fitting error')
                    continue


                if PSF_SNR < syntax['construction_SNR'] and syntax['exp_time'] > 1:
                    logger.debug('PSF constuction source too low: %s' % int(PSF_SNR))
                    logger.info('\nRan out of PSF sources above SNR=%d' % syntax['construction_SNR'] )
                    break
                else:

                    # print('\rPSF source %d / %d :: SNR: %d' % (int(PSF_SNR)),end = '')
                    pass

                # print(result.params)
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                H = result.params['A'].value
                H_err = result.params['A'].stderr

                chi2 = result.chisqr

                xc_correction =  xc - fitting_radius + syntax['scale']
                yc_correction =  yc - fitting_radius + syntax['scale']

                if syntax['use_moffat']:

                    residual = psf_image_bkg_free - moffat_2d((xx,yy),xc_correction,yc_correction,
                                                              0,H,
                                                              dict(alpha=result.params['alpha'],beta=result.params['beta'])).reshape(psf_image_bkg_free.shape)
                    PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:
                    residual = psf_image_bkg_free - gauss_2d((xx,yy),xc_correction,yc_correction,
                                                             0,H,
                                                             dict(sigma=result.params['sigma'])).reshape(psf_image_bkg_free.shape)
                    PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))

                residual /= H



                psf_mag.append(-2.5*np.log10(H/syntax['exp_time']))

                residual_regrid = np.repeat(np.repeat(residual, regriding_size, axis=0), regriding_size, axis=1)

                x_roll = scale_roll(fitting_radius,xc,regriding_size)
                y_roll = scale_roll(fitting_radius,yc,regriding_size)

                residual_roll = np.roll(np.roll(residual_regrid,y_roll,axis=0),x_roll,axis = 1)

                residual_table += residual_roll

                sources_dict['PSF_%d'%sources_used] = {}

                # print(H_err)

                sources_dict['PSF_%d'%sources_used]['x_pix']  = xc_global
                sources_dict['PSF_%d'%sources_used]['y_pix']  = yc_global
                sources_dict['PSF_%d'%sources_used]['H_psf'] = float(H/syntax['exp_time'])
                # sources_dict['PSF_%d'%sources_used]['H_psf_err'] = float(H_err/syntax['exp_time'])
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

                # logger.debug('Residual table updated: %d / %d ' % (sources_used,syntax['psf_source_no']))

                logger.info('\rResidual table updated: %d / %d ' % (sources_used,syntax['psf_source_no']) )
                logger.info(' - SNR: %d FWHM: %.3f' % (PSF_SNR,PSF_FWHM))

                sources_used +=1

                fwhm_fit.append(PSF_FWHM)


            except Exception as e:
                logger.exception(e)

                logger.error('** Fitting error - trying another source**')
                failsafe+=1
                n+=1

                continue
        print('  ')

        if sources_used < syntax['min_psf_source_no']:
            logger.warning('BUILDING PSF: Not enough useable sources found')
            return None,None,construction_sources.append([np.nan]*5),syntax

        logger.debug('PSF Successful')

        #
        # Get average of residual table
        residual_table /= sources_used

        # regrid residual table to psf size
        residual_table  = rebin(residual_table,( int(2*syntax['scale']),int(2*syntax['scale'])))

        construction_sources = pd.DataFrame.from_dict(sources_dict, orient='index',
                                                      columns=['x_pix','y_pix','H_psf','H_psf_err','fwhm','x_best','y_best'])
        construction_sources.reset_index(inplace = True)

        if syntax['plots_PSF_residual']:

            from autophot.packages.create_plots import plot_PSF_model_steps
            plot_PSF_model_steps(sources_dict,syntax,image)

        if syntax['plots_PSF_sources']:

            from autophot.packages.create_plots import plot_PSF_construction_grid

            plot_PSF_construction_grid(construction_sources,image,syntax)

        if syntax['save_PSF_models fits']:

            from astropy.io import fits
            import os

            if syntax['use_moffat']:
                
                x = np.arange(0,2*syntax['scale'])
                xx,yy= np.meshgrid(x,x)

                PSF_model_array = residual_table + moffat_2d((xx,yy),residual_table.shape[1]/2,residual_table.shape[0]/2,
                                                          0,1,
                                                          dict(alpha=syntax['image_params']['alpha'],
                                                               beta=syntax['image_params']['beta'])).reshape(residual_table.shape)

            else:
                PSF_model_array = residual_table + gauss_2d((xx,yy),residual_table.shape[1]/2,residual_table.shape[0]/2,
                                                         0,1,
                                                         dict(sigma=syntax['image_params']['sigma'])).reshape(residual_table.shape)


            hdu = fits.PrimaryHDU(PSF_model_array/np.sum(PSF_model_array))
            hdul = fits.HDUList([hdu])
            hdul.writeto(os.path.join(syntax['write_dir'],'PSF_model_'+syntax['base']+'.fits'),
                         overwrite = True)



            print('PSF model saved as .fits')


        image_radius_lst = np.array(image_radius_lst)

        syntax['image_radius'] = image_radius_lst.mean()

        logger.info('Image_radius [pix] : %.3f +/- %.3f' % (image_radius_lst.mean(), image_radius_lst.std()))

    except Exception as e:
        logger.exception('BUILDING PSF: ',e)
        raise Exception

    return residual_table,fwhm_fit,construction_sources,syntax



# =============================================================================
# Fitting of PSF
# =============================================================================

def fit(image,sources,
        residual_table,
        syntax,
        return_psf_model = False,
        save_plot = False,
        show_plot = False,
        rm_bkg_val = True,
        hold_pos = False,
        return_fwhm = False,
        return_subtraction_image = False,
        fname = None,
        no_print = False,
        cutout_base = True
        ):
    
    
    '''
    PSF fitting packages.
    
    This package uses the PSF model created in psf.build to fit for every source in an image. Can return a plot of the subtracted image for the User 
    :param image: 2D image of containg stars with which the user wishes the measure
    :type image: numpy array
    :param sources: DataFarme of sources containing location 
    :type sources: pandas DataFrame
    :param residual_table: PSF residual table from psf.build
    :type residual_table: numpy array
    :param syntax: Autophot control dictionary
    :type syntax: dict
    
    :Required syntax keywords for this package:

    :param return_psf_model: DESCRIPTION, defaults to False
    :type return_psf_model: TYPE, optional
    :param save_plot: DESCRIPTION, defaults to False
    :type save_plot: TYPE, optional
    :param show_plot: DESCRIPTION, defaults to False
    :type show_plot: TYPE, optional
    :param rm_bkg_val: DESCRIPTION, defaults to True
    :type rm_bkg_val: TYPE, optional
    :param hold_pos: DESCRIPTION, defaults to False
    :type hold_pos: TYPE, optional
    :param return_fwhm: DESCRIPTION, defaults to False
    :type return_fwhm: TYPE, optional
    :param return_subtraction_image: DESCRIPTION, defaults to False
    :type return_subtraction_image: TYPE, optional
    :param fname: DESCRIPTION, defaults to None
    :type fname: TYPE, optional
    :param no_print: DESCRIPTION, defaults to False
    :type no_print: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    
    import numpy as np
    import pandas as pd
    import pathlib
    import lmfit
    import logging

    import matplotlib.pyplot as plt

    from autophot.packages.functions import gauss_2d,moffat_2d,moffat_fwhm,gauss_sigma2fwhm
    from autophot.packages.functions import scale_roll,rebin,gauss_fwhm2sigma,set_size,order_shift
    from matplotlib.gridspec import  GridSpec
    
    from autophot.packages.rm_bkg import rm_bkg
    
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))


    logger = logging.getLogger(__name__)
    
    fwhm = syntax['fwhm']


    fitting_radius = int(np.ceil(1.3*fwhm))
    regriding_size = int(syntax['regrid_size'])

    sources = sources
    residual_table = residual_table

    def build_psf(xc, yc, sky, H, r_table, slice_scale = None,pad_shape = None):

        '''
        Slice_scale = only return "slice scaled" image
        '''

        try:

            psf_shape = r_table.shape

            if pad_shape != None:
                #  Need to make PSF fitting bigger
                top =    int((pad_shape[0] - r_table.shape[0])/2)
                bottom=  int((pad_shape[0] - r_table.shape[0])/2)
                left =   int((pad_shape[1] - r_table.shape[1])/2)
                right =  int((pad_shape[1] - r_table.shape[1])/2)

                psf_shape = pad_shape

                r_table = np.pad(r_table, [(top, bottom), (left, right)], mode='constant', constant_values=0)

            x_rebin = np.arange(0,psf_shape[0])
            y_rebin = np.arange(0,psf_shape[1])

            xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)

            if syntax['use_moffat']:
                core = moffat_2d((xx_rebin,yy_rebin),xc,yc,sky,H,syntax['image_params']).reshape(psf_shape)

            else:
                core = gauss_2d((xx_rebin,yy_rebin),xc,yc,sky,H,syntax['image_params']).reshape(psf_shape)

            residual_rebinned = np.repeat(np.repeat(r_table, regriding_size, axis=0), regriding_size, axis=1)

            x_roll = scale_roll(xc,int(r_table.shape[1]/2),regriding_size)
            y_roll = scale_roll(yc,int(r_table.shape[0]/2),regriding_size)

            residual_roll = np.roll(np.roll(residual_rebinned,y_roll,axis=0),x_roll,axis = 1)

            residual = rebin(residual_roll,psf_shape)

            psf =  (sky  + (H* residual )) + core

            if np.isnan(np.min(psf)):
                logger.info(sky,H,np.min(residual),np.min(core))

            psf[np.isnan(psf)] = 0

            if not (slice_scale is None):
                # print(slice_scale)
                psf = psf[int ( 0.5 * r_table.shape[1] - slice_scale): int(0.5*r_table.shape[1] + slice_scale),
                          int ( 0.5 * r_table.shape[0] - slice_scale): int(0.5*r_table.shape[0] + slice_scale)]

        except Exception as e:
            logger.exception(e)
            psf = np.nan

        return psf


    if return_psf_model:

        shape = int(2*syntax['scale']),int(2*syntax['scale'])
        x_slice = np.arange(0,shape[0])
        xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

        if syntax['use_moffat']:
            PSF_model = moffat_2d((xx_sl,yy_sl),shape[1]/2,shape[1]/2,0,1,dict(alpha=syntax['image_params']['alpha'],beta=syntax['image_params']['beta'])).reshape(shape)
        else:
            PSF_model= gauss_2d((xx_sl,yy_sl),shape[1]/2,shape[1]/2,0,1,dict(sigma=syntax['image_params']['sigma'])).reshape(shape)

        return PSF_model


    psf_params = []

    x = np.arange(0,2*syntax['scale'])

    xx,yy= np.meshgrid(x,x)

    if hold_pos:
        dx = 1e-6
        dy = 1e-6
    else:
        dx = syntax['dx']
        dy = syntax['dy']


    lower_x_bound = syntax['scale']
    lower_y_bound = syntax['scale']
    upper_x_bound = syntax['scale']
    upper_y_bound = syntax['scale']


    if return_subtraction_image:

        from astropy.visualization import  ZScaleInterval

        vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(image)

    '''
    Known issue - for poor images, some sources may be too close to boundary, remove this
    '''
    if not return_fwhm and not no_print:
        logger.info('Image cutout size: (%.f,%.f) (%.f,%.f)' % ((lower_x_bound,upper_x_bound,lower_y_bound,upper_y_bound)))

    # sources = sources[sources.x_pix < image.shape[1] - upper_x_bound]
    # sources = sources[sources.x_pix > lower_x_bound]
    # sources = sources[sources.y_pix < image.shape[0] - upper_y_bound]
    # sources = sources[sources.y_pix > lower_y_bound]

    if return_subtraction_image:
        image_before = image.copy()
        
    # print(sources)

    for n  in range(len(sources.index)):
        if not return_fwhm and not no_print:
            print('\rFitting PSF to source: %d / %d ' % (n+1,len(sources)), end = '')

        try:

            idx = list(sources.index)[n]
            # print('idx:',idx)
            # if hold_pos:
            #     source_base =   image
            # else:
                
            if cutout_base:
                
                source_base =   image[int(sources.y_pix[idx]-lower_y_bound): int(sources.y_pix[idx] + upper_y_bound),
                                      int(sources.x_pix[idx]-lower_x_bound): int(sources.x_pix[idx] + upper_x_bound)]
            else:
                
                source_base =   image
                # print(source_base.shape)
                # print(source_base.simage)
                
    
                # print(source_base)

            # if source_base.shape != (int(2*syntax['scale']),int(2*syntax['scale'])):
            #     print('not right shape')

            #     bkg_median = np.nan
            #     H = np.nan
            #     H_psf_err = np.nan
            #     x_fitted = np.nan
            #     y_fitted = np.nan
            #     chi2 = np.nan
            #     redchi2 = np.nan

            #     psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))
            #     continue

            xc = source_base.shape[1]/2
            yc = source_base.shape[0]/2

            xc_global = sources.x_pix[idx]
            yc_global = sources.y_pix[idx]

            if not rm_bkg_val:
                
                source_bkg_free = source_base
                bkg_median = 0
                
            else:

                try:
                    # print(source_base.shape)
                    
                    source_bkg_free, bkg_surface = rm_bkg(source_base,
                                                          syntax,
                                                          # source_base.shape[1]/2,
                                                          # source_base.shape[0]/2
                                                          )
                    bkg_median = np.nanmedian(bkg_surface)

                except Exception as e:
                    print('cannot fit background')

                    bkg_median = np.nan
                    H = np.nan
                    H_psf_err = np.nan
                    x_fitted = np.nan
                    y_fitted = np.nan
                    chi2 = np.nan
                    redchi2 = np.nan

                    psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))

                    logger.exception(e)
                    
                    continue

            if cutout_base:
                source = source_bkg_free[int(0.5*source_bkg_free.shape[1] - fitting_radius):int(0.5*source_bkg_free.shape[1] + fitting_radius) ,
                                         int(0.5*source_bkg_free.shape[0] - fitting_radius):int(0.5*source_bkg_free.shape[0] + fitting_radius) ]
                
                if source.shape != (int(2*fitting_radius),int(2*fitting_radius)):
                
                    print('fitting radius size error')
                    bkg_median = np.nan
                    H = np.nan
                    H_psf_err = np.nan
                    x_fitted = np.nan
                    y_fitted = np.nan
                    chi2 = np.nan
                    redchi2 = np.nan
    
                    psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))
                    
                    continue
            else:
                source = source_bkg_free
                
            

            if np.sum(np.isnan(source)) == len(source):
                
                
                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                x_fitted = np.nan
                y_fitted = np.nan
                chi2 = np.nan
                redchi2 = np.nan

                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))
                continue


            if hold_pos:
                dx = 1e-6
                dy = 1e-6
            else:
                dx = syntax['dx']
                dy = syntax['dy']

            if cutout_base:
                x_slice = np.arange(0,2*fitting_radius)
            else:
                x_slice = np.arange(0,source.shape[0])
                
            xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

            if return_fwhm:
                if not no_print:
                    logger.info('Fitting gaussian to source to get FWHM')

                pars = lmfit.Parameters()
                pars.add('A',
                         value = np.nanmax(source),
                         min = 0)
                
                pars.add('x0',
                         value = source.shape[1]/2,
                         min = 0.5*source.shape[1] - dx,
                         max = 0.5*source.shape[1] + dx)
                
                pars.add('y0',
                         value = source.shape[0]/2,
                         min = 0.5*source.shape[0] - dy,
                         max = 0.5*source.shape[0] + dy)

                if syntax['use_moffat']:
                    pars.add('alpha',value = syntax['image_params']['alpha'],
                             min = 0,max = 25)

                    pars.add('beta',value = syntax['image_params']['beta'],
                             min = 0,
                             vary = syntax['vary_moff_beta']  )

                else:
                    pars.add('sigma',value = syntax['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(syntax['max_fit_fwhm']) )

                if syntax['use_moffat']:
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
                    
                    result = mini.minimize(method = 'least_squares')

                xc = result.params['x0'].value
                yc = result.params['y0'].value

                if syntax['use_moffat']:
                    target_PSF_FWHM = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:
                    target_PSF_FWHM = fitting_model_fwhm(dict(sigma=result.params['sigma']))


                if not no_print:
                    logger.info('Target FWHM: %.3f' % target_PSF_FWHM)

                xc_global = xc - 0.5*source.shape[1] + sources.x_pix[idx]
                yc_global = yc - 0.5*source.shape[0] + sources.y_pix[idx]
                
                
                # if cutout_base:

                source_base =  image[int(yc_global-lower_y_bound): int(yc_global + upper_y_bound),
                                     int(xc_global-lower_x_bound): int(xc_global + upper_x_bound)]

                source_bkg_free,bkg_surface = rm_bkg(source_base,syntax,
                                                     # source_bkg_free.shape[0]/2,
                                                     # source_bkg_free.shape[1]/2
                                                     )

                bkg_median = np.nanmean(bkg_surface)

                source = source_bkg_free[int(source_bkg_free.shape[1]/2 - fitting_radius):int(source_bkg_free.shape[1]/2 + fitting_radius) ,
                                         int(source_bkg_free.shape[0]/2 - fitting_radius):int(source_bkg_free.shape[0]/2 + fitting_radius) ]
                # else:
                    
                #     source = source_base

# =============================================================================
#
# =============================================================================

            try:

                '''
                Fit and subtract PSF model
                '''

                if hold_pos:
                    dx = 1e-6
                    dy = 1e-6
                else:
                    dx = syntax['dx']
                    dy = syntax['dy']

                pars = lmfit.Parameters()

                pars.add('A', value = np.nanmax(source)*0.75,min = 0)

                pars.add('x0',
                         value = 0.5*residual_table.shape[1],
                         min   = 0.5*residual_table.shape[1]-dx,
                         max   = 0.5*residual_table.shape[1]+dx)

                pars.add('y0',
                         value = 0.5*residual_table.shape[0],
                         min   = 0.5*residual_table.shape[0]-dy,
                         max   = 0.5*residual_table.shape[0]+dy)
                
                def residual(p):
                    p = p.valuesdict()
                    # print(build_psf(p['x0'],p['y0'],0,p['A'],residual_table,slice_scale = source.shape[0]/2))
                    res = (source - build_psf(p['x0'],p['y0'],0,p['A'],
                                              residual_table,
                                              slice_scale = source.shape[0]/2,
                                              # pad_shape = source.shape
                                              ))
                    return res.flatten()


                
                import warnings
                with warnings.catch_warnings():
                    
                    warnings.simplefilter("ignore")
                    
                    mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)
                    
                    result = mini.minimize(method = 'least_squares')

                xc = result.params['x0'].value

                yc = result.params['y0'].value

                H = result.params['A'].value

                H_psf_err = result.params['A'].stderr

                chi2 = result.chisqr
                redchi2 = result.redchi

                x_fitted = xc - 0.5*residual_table.shape[1] + xc_global
                y_fitted = yc - 0.5*residual_table.shape[1] + yc_global

                if syntax['remove_sat']:

                    if H+bkg_median >= syntax['sat_lvl']:
                        # print('***sat')
                        # print('here')
                        bkg_median = np.nan
                        H = np.nan
                        H_psf_err = np.nan
                        # x_fitted = np.nan
                        # y_fitted = np.nan
                        chi2 = np.nan
                        redchi2 = np.nan
        
                        psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))
                        continue


            except Exception as e:
                print('some error',e)
                logger.exception(e)
                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                x_fitted = np.nan
                y_fitted = np.nan
                chi2 = np.nan
                redchi2 = np.nan

                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))
                continue

            if syntax['use_covarience']:

                H_psf_err = result.params['A'].stderr

            else:
                logger.warning('Error not computed')
                H_psf_err = 0

            psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))

            if return_subtraction_image:



                try:
                    ncols = 3
                    nrows = 2

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(500,0.5))

                    grid = GridSpec(nrows, ncols ,
                                    wspace=0.5,
                                    hspace=0.1)

                    ax1   = fig.add_subplot(grid[0:2, 0:2])
                    ax_before = fig.add_subplot(grid[0, 2])
                    ax_after = fig.add_subplot(grid[1, 2])

                    image_section = image[int(yc_global -  lower_y_bound): int(yc_global + upper_y_bound),
                                          int(xc_global -  lower_y_bound): int(xc_global + upper_y_bound)]

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

                    image_section_subtraction = image_section - build_psf(xc , yc, 0, H, residual_table,slice_scale = image_section.shape[0]/2)

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

                    save_loc = os.path.join(syntax['write_dir'],'cleaned_images')

                    os.makedirs(save_loc, exist_ok=True)

                    fig.savefig(os.path.join(save_loc,'subtraction_%d.pdf' % idx),
                                bbox_inches='tight')

                    logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')
                    pass


            if syntax['show_residuals'] or show_plot == True or save_plot == True:


                try:
                    from astropy.visualization import  ZScaleInterval

                    fitted_source = build_psf(xc,yc,0,H,residual_table)
                    subtracted_image = source_bkg_free - fitted_source + bkg_surface

                    scale = order_shift(abs(source_base))

                    source_base = source_base/scale
                    subtracted_image=subtracted_image / scale
                    bkg_surface = bkg_surface/ scale
                    fitted_source = fitted_source/ scale

                    vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(source_base)

                    h, w = source_bkg_free.shape

                    x  = np.linspace(0, int(2*syntax['scale']), int(2*syntax['scale']))
                    y  = np.linspace(0, int(2*syntax['scale']), int(2*syntax['scale']))

                    X, Y = np.meshgrid(x, y)

                    ncols = 6
                    nrows = 3

                    heights = [1,1,0.75]
                    widths = [1,1,0.75,1,1,0.75]

                    plt.ioff()

                    fig = plt.figure(figsize = set_size(500,aspect = 0.5))

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
                               aspect="auto",
                               vmin = vmin,
                               vmax = vmax,
                               interpolation = 'nearest'
                               )

                    ax1.scatter(xc,yc,
                                label = 'Best fit',
                                marker = '+',
                                color = 'red',
                                s = 20)

                    ax1_R.plot(source_base[:,w//2],Y[:,w//2],marker = 'o',color = 'blue',label = '1D projection')
                    ax1_B.plot(X[h//2,:],source_base[h//2,:],marker = 'o',color = 'blue')

                    # include surface
                    ax1_R.plot(bkg_surface[:,w//2],Y[:,w//2],marker = 's',color = 'red',label = 'Background Fit')
                    ax1_B.plot(X[h//2,:],bkg_surface[h//2,:],marker = 's',color = 'red')

                    # include fitted_source
                    ax1_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],marker = 's',color = 'green',label = 'PSF')
                    ax1_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],marker = 's',color = 'green')


                    ax1_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax1_R.set_xlabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax2_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(scale))

                    ax2_R.set_xlabel('Counts [$10^{%d}$] ' % np.log10(scale))

                    ax2.imshow(subtracted_image,
                               origin = 'lower',
                               aspect="auto",
                               vmin = vmin,
                               vmax = vmax,
                               interpolation = 'nearest'
                               )

                    ax2.scatter(xc,yc,label = 'Best fit',
                                marker = '+',
                                color = 'red',
                                s = 20)

                    ax2_R.plot(subtracted_image[:,w//2],Y[:,w//2],marker = 'o',color = 'blue')
                    ax2_B.plot(X[h//2,:],subtracted_image[h//2,:],marker = 'o',color = 'blue')

                    # Show surface
                    ax2_R.plot(bkg_surface[:,w//2],Y[:,w//2],marker = 's',color = 'red')
                    ax2_B.plot(X[h//2,:],bkg_surface[h//2,:],marker = 's',color = 'red')

                    # include fitted_source
                    ax2_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],marker = 's',color = 'green')
                    ax2_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],marker = 's',color = 'green')

                    ax1_R.tick_params(axis='x', rotation=-90)
                    ax2_R.tick_params(axis='x', rotation=-90)

                    ax2_B.set_ylim(ax1_B.get_ylim()[0],ax1_B.get_ylim()[1])
                    ax2_B.set_xlim(ax1_B.get_xlim()[0],ax1_B.get_xlim()[1])

                    ax2_R.set_ylim(ax1_R.get_ylim()[0],ax1_R.get_ylim()[1])
                    ax2_R.set_xlim(ax1_R.get_xlim()[0],ax1_R.get_xlim()[1])


                    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                    handles,labels = [sum(i, []) for i in zip(*lines_labels)]

                    by_label = dict(zip(labels, handles))

                    fig.legend(by_label.values(), by_label.keys(),
                               bbox_to_anchor=(0.5, 0.95), loc='lower center',
                               ncol = 4,
                               frameon=False)

                    if syntax['show_residuals']:
                    
                        pathlib.Path(syntax['write_dir']+'/'+'psf_subtractions/').mkdir(parents = True, exist_ok=True)
                        
                        save_name = lambda x: syntax['write_dir']+'psf_subtractions/'+'psf_subtraction_{}.png'.format(int(x))
                        i = 0
                        while True:
                            if not os.path.exists(save_name(n+i)):
                                break
                            else:
                                i+=1
                        
                        plt.savefig(save_name(n+i),bbox_inches='tight')
                        
                    else:
                        fig.savefig(syntax['write_dir']+'target_psf_'+fname+'.pdf',bbox_inches='tight')

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')


        except Exception as e:
             logger.exception(e)


             bkg_median = np.nan
             H = np.nan
             H_psf_err = np.nan
             psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err,chi2,redchi2))

             continue
    # print(' ... done')

    if syntax['plot_before_after'] and return_subtraction_image:

        print('Saving before and after image')

        plt.ioff()

        fig = plt.figure(figsize = set_size(500,1))

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image_before,
                    origin = 'lower',
                    aspect="auto",
                    vmin = vmin,
                    vmax = vmax,
                    cmap='gray',
                    interpolation = 'nearest'
                    )

        ax2.imshow(image,
                    origin = 'lower',
                    aspect="auto",
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
        ax2.set_title('Point source Free image [%d$\\sigma$]' % syntax['do_all_phot_sigma'])

        plt.savefig(syntax['write_dir']+'PSF_BEFORE_AFTER.pdf',
                    bbox_inches='tight')

        plt.close(fig)

    new_df =  pd.DataFrame(psf_params,
                           columns = ('idx','x_fitted','y_fitted','bkg','H_psf','H_psf_err','chi2','redchi2'),
                           index = sources.index)

    if return_fwhm:
        new_df['target_fwhm'] = target_PSF_FWHM

    elif not no_print:
        print('  ')
    if not return_psf_model:


        return pd.concat([sources,new_df],axis = 1),build_psf



def do(df,residual,syntax,fwhm):

    """ Function to perform PSF model measuremt


    :param df: dataframe contain amplitude and amplitude error (in counts) of PSF from *psf.fit* function
    :type df: Pandas.DataFrame

    :param residual: 2D image of residual table
    :type residual: np.array

    :param syntax: AutoPhot control dictionary
    :type syntax: dict

    :param FWHM: Full Width Half Maximum of image used to set integration radius
    :type FWHM: float

    :return: original Dataframe with *psf_counts* and *psf_counts_err* columns
    :rtype: Pandas.DataFrame
    """


    try:
        from photutils import CircularAperture
        from photutils import aperture_photometry
        from scipy.integrate import dblquad
        import logging
        import numpy as np
        
        from autophot.packages.functions import gauss_2d,moffat_2d

        logger = logging.getLogger(__name__)

        xc = syntax['scale']
        yc = syntax['scale']

        # Integration radius
        int_scale = syntax['ap_size'] * fwhm

        int_range_x = [xc - int_scale , xc + int_scale]
        int_range_y = [yc - int_scale , yc + int_scale]


        x_rebin = np.arange(0,2*syntax['scale'])
        y_rebin = np.arange(0,2*syntax['scale'])

        xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)

        if syntax['use_moffat']:
            core = moffat_2d((xx_rebin,yy_rebin),syntax['scale'],syntax['scale'],0,1,syntax['image_params']).reshape(residual.shape)
        else:
            core = gauss_2d((xx_rebin,yy_rebin),syntax['scale'],syntax['scale'],0,1,syntax['image_params']).reshape(residual.shape)

        # Core Gaussian component with height 1 and sigma value sigma
        if syntax['use_moffat']:
            core_int= lambda y, x: moffat_2d((x,y),syntax['scale'],syntax['scale'],0,1,syntax['image_params'])
        else:
            core_int= lambda y, x: gauss_2d((x,y),syntax['scale'],syntax['scale'],0,1,syntax['image_params'])

        core_int = dblquad(core_int, int_range_y[0],int_range_y[1],lambda x:int_range_x[0],lambda x:int_range_x[1])[0]

        # Aperture Photometry over residual
        apertures = CircularAperture((syntax['scale'],syntax['scale']), r=int_scale)
        phot_table = aperture_photometry(residual, apertures,method='subpixel',subpixels=4)

        phot_table['aperture_sum'].info.format = '%.8g'
        residual_int = phot_table[0]

        # Counts from core compoent on PSF
        syntax['c_counts'] = float(core_int)

        # Counts from Residual component of PSF
        syntax['r_counts'] = float(residual_int['aperture_sum'])


        unity_psf = aperture_photometry(core+residual, apertures,method='subpixel',subpixels=4)
        # Counts in a PSF with fwhm 2 sqrt(2 ln 2) * sigma and height 1
        sudo_psf = float(unity_psf['aperture_sum'][0])

        psf_int     = df.H_psf     * sudo_psf
        psf_int_err = df.H_psf_err * sudo_psf

        df['psf_counts']     = psf_int.values
        df['psf_counts_err'] = psf_int_err.values

    except Exception as e:
        logger.exception(e)
        df = np.nan

    return df,syntax