#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:13:14 2018

@author: seanbrennan
"""

def create_circular_mask(h, w, center=None, radius=None):
    '''
    
    :param h: DESCRIPTION
    :type h: TYPE
    :param w: DESCRIPTION
    :type w: TYPE
    :param center: DESCRIPTION, defaults to None
    :type center: TYPE, optional
    :param radius: DESCRIPTION, defaults to None
    :type radius: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE
    
    TODO: Mask out boundary pixes while searching for sources

    '''
    import numpy as np

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    
    
    return mask


def fwhm(image,syntax,sigma_lvl = None,fwhm = None):
    '''
    
    :param image: 2D astronoimical image
    :type image: numpy.array
    :param syntax: DESCRIPTION
    :type syntax: TYPE
    :param sigma_lvl: DESCRIPTION, defaults to None
    :type sigma_lvl: TYPE, optional
    :param fwhm: DESCRIPTION, defaults to None
    :type fwhm: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma
    from autophot.packages.functions import pix_dist

    from autophot.packages.functions import moffat_2d,moffat_fwhm
    import numpy as np
    import pandas as pd
    import lmfit
    from astropy.stats import sigma_clip
    import logging
    import os

    logger = logging.getLogger(__name__)



    def combine(dictionaries):
        combined_dict = {}
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                combined_dict.setdefault(key, []).append(value)
        return combined_dict

    if sigma_lvl != None:
        min_source_no  = 0
        max_source_no  = np.inf
    else:
        max_source_no = syntax['max_source_lim']
        min_source_no = syntax['min_source_lim']

    if sigma_lvl == None:
        threshold_value  = syntax['threshold_value']
        int_fwhm = syntax['fwhm_guess']
    else:
        threshold_value = sigma_lvl
        if fwhm != None:
            int_fwhm = fwhm

        else:
            int_fwhm = syntax['fwhm_guess']

    logging.info('\nFinding FWHM')

    if fwhm != None:
        syntax['int_scale'] = syntax['scale']
        int_fwhm = syntax['fwhm']

    if syntax['use_moffat']:
        logging.info('Using Moffat Profile for fitting')

        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:
        logging.info('Using Gaussian Profile for fitting')

        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm

    if syntax['use_daofind']:

        print('Using DAOFIND for source detection')

    elif syntax['use_imageseg']:
        print('Using Image Segmentation for source detection')

    else:
        syntax['use_daofind'] = True
        print('No search method selected - Using DAOFIND for source detection')

    if syntax['use_local_stars_for_FWHM'] and not syntax['prepare_templates']:

        mask = np.zeros(image.shape).astype(int)

        h, w = mask.shape

        mask_circular = create_circular_mask(h, w,
                                             center = (syntax['target_x_pix'],syntax['target_y_pix']),
                                             radius = syntax['local_radius'] )
        mask[~mask_circular] = 1

    else:
        mask = np.zeros(image.shape)

    if syntax['mask_sources'] and not syntax['prepare_templates']:

        h, w = mask.shape



        for X_mask,Y_mask,R_mask in syntax['mask_sources_XY_R']:
            
            mask_circular = create_circular_mask(h, w,
                                             center = (X_mask,Y_mask),
                                             radius = R_mask )
            mask[mask_circular] = 1


    image_params = []
    isolated_sources = []

    img_seg = [0]
    try:
        for idx in range(len(list(img_seg))):

            mean, median, std = sigma_clipped_stats(image,
                                                    sigma=syntax['bkg_level'],
                                                    maxiters=3)
            
            if median > 60000 and syntax['remove_sat']:
                logger.info('High background level [%d counts] - ignoring saturated  stars' % median)

                syntax['remove_sat'] = False

            if sigma_lvl == None:
                logger.debug('Image stats: Mean %.3f :: Median %.3f :: std %.3f' % (mean,median,std))
                syntax['global_mean'] = mean
                syntax['global_median'] = median
                syntax['global_std'] = std

            # decrease
            m = 0

            # increase
            n = 0

            # backstop
            failsafe = 0

            decrease_increment = False

            # How much to drop/increase each iteration each
            fudge_factor = syntax['fudge_factor']

            search_image = image.copy()

            iso_temp = []


            try:
                # Remove target by masking with area with that of median image value - just so it's not picked up
                if syntax['target_name'] != None and fwhm == None and not syntax['prepare_templates']:

                    logger.info('Target location : (x,y) -> (%.3f,%.3f)' % (syntax['target_x_pix'] , syntax['target_y_pix']))

                    if abs(syntax['target_x_pix'])>=search_image.shape[1] or abs(syntax['target_y_pix'])>=search_image.shape[0]:
                        logger.warning('Target not located on images')
                        raise Exception ( ' *** EXITING - Target pixel coordinates outside of image [%s , %s] *** ' % (int(syntax['target_y_pix']), int(syntax['target_y_pix'])))

                    else:
                        search_image[int(syntax['target_y_pix'])-syntax['int_scale']: int(syntax['target_y_pix']) + syntax['int_scale'],
                                     int(syntax['target_x_pix'])-syntax['int_scale']: int(syntax['target_x_pix']) + syntax['int_scale']] =  syntax['global_median'] * np.ones((int(2*syntax['int_scale']),int(2*syntax['int_scale'])))
            except:
                print('Target position not defined - ignoring for now')

            while True:
                    try:
                        # If iterations get to big - terminate
                        if failsafe>syntax['source_max_iter']:
                            logger.info(' Source detection gives up!')
                            break
                        else:
                            failsafe +=1

                        # check if threshold value is still good
                        threshold_value_check = threshold_value + n - m

                        # if <=0 reverse previous drop and and fine_fudge factor
                        if threshold_value_check  <= syntax['lim_SNR']:
                            
                                logger.warning('Threshold value has gone below threshold - increasing by smaller increment ')

                                # revert privious decrease
                                decrease_increment = True
                                n=syntax['fine_fudge_factor']

                                # m = 0 to stop any further decrease
                                threshold_value += m
                                m = 0

                        else:

                            threshold_value = round(threshold_value + n - m,3)

                        # if threshold goes negative use smaller fudge factor
                        if decrease_increment:
                            fudge_factor = syntax['fine_fudge_factor']

                        if syntax['use_daofind']:

                            daofind = DAOStarFinder(fwhm      = np.ceil(int_fwhm),
                                                    threshold = threshold_value*std,
                                                    sharplo   =  0.2,sharphi = 1.0,
                                                    roundlo   = -1.0,roundhi = 1.0,
                                                    )
                            sources = daofind(search_image - median,
                                              mask = mask.astype(bool))


                            if sources == None:
                                logger.warning('Sources == None at %.1f sigma - increasing offse' % threshold_value)
                                m = fudge_factor
                                continue

                            sources = sources.to_pandas()

                        elif syntax['use_imageseg']:

                            from photutils import detect_threshold

                            threshold = detect_threshold(search_image,
                                                         nsigma=threshold_value)

                            from astropy.convolution import Gaussian2DKernel
                            from astropy.stats import gaussian_fwhm_to_sigma
                            from photutils import detect_sources

                            sigma = np.ceil(int_fwhm) * gaussian_fwhm_to_sigma    # FWHM = 2.
                            kernel = Gaussian2DKernel(sigma,
                                                      # x_size=6, y_size=6
                                                      )
                            # kernel.normalize()

                            segm = detect_sources(search_image,
                                                  threshold,
                                                  npixels=3,
                                                  filter_kernel=kernel)

                            from photutils import deblend_sources
                            segm_deblend = deblend_sources(search_image,
                                                           segm,
                                                           npixels=3,
                                                           filter_kernel=kernel)

                            from photutils import SourceCatalog
                            props = SourceCatalog(search_image, segm_deblend )

                            sources = props.to_table().to_pandas()

                            eccentricity_check = sources['eccentricity']<syntax['eccentricity_check']

                            sources = sources[eccentricity_check]



                        logger.info('\nNumber of sources before cleaning [ %.1f sigma ]: %d ' % (threshold_value,len(sources)))




                        if len(sources) == 0:
                            logger.warning('No sources')
                            m = fudge_factor
                            continue


                        try:
                            sources['xcentroid']
                            'Make sure some are detceted, if not try again '
                        except Exception as e:
                            logger.exception(e)
                            break

                        if len(sources) > max_source_no:

                            logger.warning('Too many sources')
                            if n==0:
                                threshold_value *=2

                            elif m != 0 :
                                decrease_increment = True
                                n = syntax['fine_fudge_factor']
                                fudge_factor = syntax['fine_fudge_factor']

                            else:
                                n = fudge_factor

                            continue

                        elif len(sources) > 5000 and m !=0:

                            logger.warning('Picking up noise')
                            fudge_factor = syntax['fine_fudge_factor']
                            n = syntax['fine_fudge_factor']
                            m = 0
                            decrease_increment = True

                            continue

                        elif len(sources) < min_source_no:

                            logger.warning('Too few sources')
                            m = fudge_factor

                            continue

                        elif len(sources) == 0:

                            logger.warning('No sources')
                            m = fudge_factor

                            continue



                        if syntax['remove_boundary_sources']:

                            with_boundary = len(sources)

                            sources = sources[sources['xcentroid'] < image.shape[1] - syntax['pix_bound'] ]
                            sources = sources[sources['xcentroid'] > syntax['pix_bound'] ]
                            sources = sources[sources['ycentroid'] < image.shape[0] - syntax['pix_bound'] ]
                            sources = sources[sources['ycentroid'] > syntax['pix_bound'] ]

                            logger.info('Removed %d sources near boundary' % (with_boundary - len(sources)))

                        #  Interested in these sources
                        x = np.array(sources['xcentroid'])
                        y = np.array(sources['ycentroid'])

                        if len(sources) < min_source_no:

                            logger.warning('Less than min source after boundary removal')
                            m = fudge_factor

                            continue

                        if sigma_lvl != None or len(sources) < 10:

                            isolated_sources = pd.DataFrame({'x_pix':x,'y_pix':y})

                        elif syntax['isolate_sources']:
                        # else:

                            # Make sure sources are isolated by a significant amount

                            not_isolated = 0

                            for idx in range(len(x)):

                                try:

                                    x0 = x[idx]
                                    y0 = y[idx]

                                    dist = np.sqrt((x0-np.array(x))**2+(y0-np.array(y))**2)

                                    dist = dist[np.where(dist>0)]

                                    if len(dist) == 0:
                                        dist = [0]

                                    if min(dist) >= syntax['iso_scale']:

                                        df = np.array((float(x0),float(y0)))

                                        iso_temp.append(df)

                                        # pix_dist.append(dist)

                                    else:
                                        not_isolated+=1

                                except Exception as e:
                                    logger.exception(e)
                                    pass

                            logger.info('Removed %d crowded source' % ( not_isolated))

                            if len(iso_temp) == 0:
                                logger.warning('Less than min source available after isolating sources')
                                m = fudge_factor
                                continue

                            isolated_sources= pd.DataFrame(data = iso_temp)
                            isolated_sources.columns = ['x_pix','y_pix']
                            isolated_sources.reset_index()
                        else:
                            isolated_sources = sources

                        #  x,y recenter lists
                        x_rc = []
                        y_rc = []

                        fwhm_list=[]
                        fwhm_list_err = []
                        medianlst=[]

                        x_pix = np.arange(0,2 * syntax['int_scale'])
                        y_pix = np.arange(0,2 * syntax['int_scale'])

                        image_copy = image.copy()

                        saturated_source=0
                        broken_closeup = 0

                        if syntax['remove_sat']:

                            try:
                                # Look for predefined saturation level
                                saturation_lvl = syntax['sat_lvl']

                            except:

                                saturation_lvl = 2**16
                                syntax['sat_lvl'] = saturation_lvl


                        # print(isolated_sources)

                        for idx in isolated_sources.index:

                            try:

                                x0 = float(isolated_sources['x_pix'].loc[[idx]])
                                y0 = float(isolated_sources['y_pix'].loc[[idx]])

                                close_up = image_copy[int(y0 - syntax['int_scale']): int(y0 + syntax['int_scale']),
                                                      int(x0 - syntax['int_scale']): int(x0 + syntax['int_scale'])]
                                # Incorrect image size
                                if close_up.shape != (int(2*syntax['int_scale']),int(2*syntax['int_scale'])):

                                    fwhm_list.append(np.nan)
                                    fwhm_list_err.append(np.nan)
                                    x_rc.append(np.nan)
                                    y_rc.append(np.nan)
                                    medianlst.append(np.nan)
                                    # logger.warning('wrong close-up size [x: %.3f :: y %.3f]' % (x0,y0))
                                    broken_closeup+=1

                                    continue

                                # Saturdated or nan values in close-up
                                if (np.nanmax(close_up)>= syntax['sat_lvl'] or np.isnan(np.max(close_up))) and syntax['remove_sat']:

                                    saturated_source +=1
                                    # logger.warning('Saturated source [x: %.3f :: y %.3f]' % (x0,y0))
                                    fwhm_list.append(np.nan)
                                    fwhm_list_err.append(np.nan)
                                    medianlst.append(np.nan)
                                    x_rc.append(np.nan)
                                    y_rc.append(np.nan)

                                    continue

                                mean, median_val, std = sigma_clipped_stats(close_up,
                                                                            sigma=syntax['bkg_level'],
                                                                            maxiters=syntax['fwhm_iters'])

                                xx, yy = np.meshgrid(x_pix, y_pix)

                                try:

                                     pars = lmfit.Parameters()
                                     pars.add('A',value = np.nanmax(close_up),min = 0)
                                     pars.add('x0',value = close_up.shape[0]/2,min = 0,max = close_up.shape[1])
                                     pars.add('y0',value = close_up.shape[0]/2,min = 0,max = close_up.shape[0])
                                     pars.add('sky',value = np.nanmedian(close_up))

                                     if syntax['use_moffat']:
                                         pars.add('alpha',value = 3,min = 0,max = 30)
                                         pars.add('beta',value = syntax['default_moff_beta'],
                                                  min = 0,
                                                  vary = syntax['vary_moff_beta']  )

                                     else:
                                         pars.add('sigma',value = 3,
                                                  min = 0,
                                                  max = gauss_fwhm2sigma(syntax['max_fit_fwhm']) )

                                     if syntax['use_moffat']:
                                         def residual(p):
                                             p = p.valuesdict()
                                             return (close_up - moffat_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(close_up.shape)).flatten()
                                     else:
                                         def residual(p):
                                             p = p.valuesdict()
                                             return (close_up - gauss_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(sigma=p['sigma'])).reshape(close_up.shape)).flatten()

                                     mini = lmfit.Minimizer(residual, pars,nan_policy = 'omit')
                                     result = mini.minimize(method = 'least_squares')

                                     if syntax['remove_sat']:

                                        if result.params['A'].value >= syntax['sat_lvl']:

                                            saturated_source +=1

                                            # logger.warning('Saturated source [x: %.3f :: y %.3f]' % (x0,y0))
                                            fwhm_list.append(np.nan)
                                            fwhm_list_err.append(np.nan)
                                            medianlst.append(np.nan)
                                            x_rc.append(np.nan)
                                            y_rc.append(np.nan)

                                            continue

                                     if syntax['use_moffat']:
                                         source_image_params = dict(alpha=result.params['alpha'].value,
                                                                    beta=result.params['beta'].value)

                                         source_image_params_STD = dict(alpha=result.params['alpha'].stderr,
                                                                        beta=result.params['beta'].stderr)
                                         fwhm_fit = fitting_model_fwhm(source_image_params)
                                         fwhm_fit_err = np.nan

                                     else:
                                         '''
                                         TODO account for it fitting fails and fitting returns NONE
                                         '''
                                         source_image_params = dict(sigma=result.params['sigma'].value)
                                         source_image_params_STD = dict(sigma=result.params['sigma'].stderr)
                                         
                                         
                                         fwhm_fit = fitting_model_fwhm(source_image_params)
                                         fwhm_fit_err = fitting_model_fwhm(source_image_params_STD)

                                     bkg_approx = result.params['sky'].value

                                     # Add details to lists
                                     image_params.append(source_image_params)

                                     fwhm_list.append(fwhm_fit)
                                     fwhm_list_err.append(fwhm_fit_err)


                                     medianlst.append(bkg_approx)

                                     x_rc.append(result.params['x0'].value - syntax['int_scale'] + x0)
                                     y_rc.append(result.params['y0'].value - syntax['int_scale'] + y0)


                                except Exception as e:
                                    logger.exception(e)

                                    fwhm_list.append(np.nan)
                                    fwhm_list_err.append(np.nan)
                                    medianlst.append(np.nan)
                                    x_rc.append(np.nan)
                                    y_rc.append(np.nan)

                                    pass

                            except Exception as e:

                                logger.exception(e)

                                continue

                        logging.info('Removed %d saturated sources' %  saturated_source)

                        if broken_closeup != 0:
                            logging.info('Incorrect %d cutouts removed ' %  broken_closeup)

                        # Add these values to the dataframe
                        # isolated_sources['x_pix'] = x_rc
                        # isolated_sources['y_pix'] = y_rc
                        isolated_sources['FWHM'] = fwhm_list
                        isolated_sources['FWHM_err'] = fwhm_list_err
                        isolated_sources['median'] = medianlst

                        if syntax['sigmclip_median']:


                            median_mask = sigma_clip(isolated_sources['median'].values,
                                                      sigma=syntax['sigmaclip_median_sigma'],
                                                      masked = True,
                                                      cenfunc = 'mean').mask

                            isolated_sources = isolated_sources[~median_mask]

                            logger.info('Removed %d high background sources' % (np.sum(median_mask)))


                        nan_idx = np.isnan(isolated_sources['x_pix'].values)

                        isolated_sources = isolated_sources[~nan_idx]

                        isolated_sources.reset_index(inplace = True,drop = True)

                        if sigma_lvl == None:

                            if isolated_sources['FWHM'].values == np.array([]):
                                logger.info('No sigma values taken')
                                continue

                            if len(isolated_sources) < min_source_no:

                                logger.warning('Less than min source after sigma clipping: %d' % len(isolated_sources))
                                threshold_value += m

                                if n ==0:

                                    decrease_increment = True
                                    n = syntax['fine_fudge_factor']
                                    fudge_factor = syntax['fine_fudge_factor']

                                else:

                                    n = fudge_factor

                                m=0

                                continue


                            nan_idx = np.isnan(isolated_sources['FWHM'].values)

                            isolated_sources = isolated_sources[~nan_idx]

                            if syntax['sigmaclip_FWHM']:

                                isolate_mask = sigma_clip(isolated_sources['FWHM'].values,
                                                          sigma=syntax['sigmaclip_FWHM_sigma'],
                                                          masked = True,
                                                          cenfunc = 'mean')
                                
                                if len(isolate_mask) ==0:
                                    pass
                                
                                else:
                                    isolated_sources = isolated_sources[~isolate_mask.mask]
                                    logger.info('Removed %d FWHM outliers' % (np.sum(isolate_mask.mask)))

                            logger.info('Useable sources found [ %d sigma ]: %d' % (threshold_value,len(isolated_sources)))

                            mean_fwhm =  np.nanmean(isolated_sources['FWHM'])

                            if np.isnan(mean_fwhm):

                                m = fudge_factor

                                continue


                        else:

                            isolate_mask = sigma_clip(isolated_sources['FWHM'].values,
                                                      sigma=syntax['sigmaclip_FWHM_sigma'],
                                                      cenfunc = 'mean').mask

                            isolated_sources = isolated_sources[~isolate_mask]


                            mean_fwhm = fwhm

                        break

                    except Exception as e:
                        logger.exception(e)
                        continue

        image_params_combine = combine(image_params)
        image_params_out={}

        for key,val in image_params_combine.items():
            val = np.array(val)

            mask = np.array(sigma_clip(val,
                                       sigma=3,
                                       cenfunc = 'mean').mask)

            image_params_out[key] = np.nanmedian(val[~mask])
            image_params_out[key+'_err'] = np.nanstd(val[~mask])

        syntax['image_params'] = image_params_out

        mean_fwhm = fitting_model_fwhm(image_params_out)


        # Update and set image cutout scale
        syntax['scale'] = int(np.ceil(syntax['scale_multipler'] * mean_fwhm)) + 0.5

        if syntax['Save_FWHM_plot']:

            import numpy as np
            from scipy.stats import norm
            import matplotlib.pyplot as plt
            try:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
            except:
                pass
            from autophot.packages.functions import set_size


            plt.ioff()

            fig = plt.figure(figsize = set_size(250,1))


            ax1 = fig.add_subplot(111)

            # Fit a normal distribution to the data:
            mu, std = norm.fit(isolated_sources['FWHM'].values)

            # Plot the histogram.
            ax1.hist(isolated_sources['FWHM'].values, bins=25, density=True,label = 'FWHM')

            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)

            ax1.plot(x, p, 'k', linewidth=0.5,label = 'PDF',color = 'r')

            ax1.set_xlabel(r'Full Width Half Maximum [pixels]')
            ax1.set_ylabel('Probability Denisty')

            ax1.legend(loc = 'best',
                       frameon = False)

            ax1.axvline(mean_fwhm,color = 'black',ls = '--',label = 'Mean FWHM')




            figname = os.path.join(syntax['write_dir'],'image_analysis_'+syntax['base']+'.pdf')

            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )
            plt.close(fig)


        if syntax['plot_image_analysis']:


            import numpy as np
            from scipy.stats import norm
            import matplotlib.pyplot as plt
            import matplotlib.pyplot as plt
            from autophot.packages.functions import set_size
            import matplotlib as mpl
            try:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
            except:
                pass



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

            if syntax['use_local_stars_for_FWHM'] and not syntax['prepare_templates']:
                local_radius_circle = plt.Circle( ( syntax['target_x_pix'], syntax['target_y_pix'] ), syntax['local_radius'],
                                                     color = 'red',
                                                     ls = '--',
                                                     label = 'Local Radius [%d px]' % syntax['local_radius'],
                                                     fill=False)
                ax1.add_patch( local_radius_circle)
            if syntax['mask_sources']:

                for X_mask,Y_mask,R_mask in syntax['mask_sources_XY_R']:
                    masked_radius_circle = plt.Circle( ( X_mask, Y_mask ), R_mask,
                                                 color = 'green',
                                                 ls = ':',
                                                 label = 'Masked Region',
                                                 fill=False)
                    ax1.add_patch(masked_radius_circle)
                    
            ax1.set_xlim(0,image.shape[1])
            ax1.set_ylim(0,image.shape[0])

            cmap = plt.cm.jet

            import numpy as np

            import matplotlib.pyplot as plt
            import pandas as pd

            ticks=np.linspace(isolated_sources['FWHM'].values.min(),isolated_sources['FWHM'].values.max(),10)

            # cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
            norm = mpl.colors.BoundaryNorm(ticks, cmap.N)

            ax1.scatter(isolated_sources['x_pix'].values,
                         isolated_sources['y_pix'].values,
                         cmap=cmap,
                         norm = norm,
                         marker = "+",
                         s = 25,
                         facecolors = None,
                         c = isolated_sources['FWHM'].values)

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            if  not syntax['prepare_templates']:
                ax1.scatter([syntax['target_x_pix']],[syntax['target_y_pix']],
                           marker = 'D',
                           s = 25,
                           facecolor = 'None',
                           edgecolor = 'gold')


            ax1_R.scatter(isolated_sources['FWHM'].values,isolated_sources['y_pix'].values,
                          cmap=cmap,
                          norm = norm,
                          marker = "o",
                          # s = 25,
                          # facecolors = None,
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
                          # s = 25,
                          # facecolors = None,
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

            # ax1.set_title(syntax['base']+'\n'+syntax['tele']+' -> '+syntax['inst']+' -> '+syntax['filter']+'-band')

            figname = os.path.join(syntax['write_dir'],'image_analysis_'+syntax['base']+'.pdf')
            fig.savefig(figname,
                        format = 'pdf',
                        bbox_inches='tight'
                        )
            plt.close(fig)


        if syntax['save_image_analysis']:


            isolated_sources.round(3).to_csv(os.path.join(syntax['write_dir'],'image_analysis_'+syntax['base']+'.csv'))


        return mean_fwhm,isolated_sources,syntax

    except Exception as e:

        logger.exception(e)

        return np.nan,np.nan,syntax





def phot(image,syntax,df,fwhm,LOG=None,):

    import sys
    import numpy as np
    import os
    import logging

    from autophot.packages.aperture import ap_phot
    from autophot.packages.functions import find_mag

    logger = logging.getLogger(__name__)


    try:

        ap_dict = {}
        ap_dict['inf_ap'] = syntax['inf_ap_size'] * fwhm
        ap_dict['ap']     = syntax['ap_size'] * fwhm

        positions  = list(zip(np.array(df['x_pix']),np.array(df['y_pix'])))



        for key,val in ap_dict.items():

            try:

                ap , bkg = ap_phot(positions,image,
                                   radius = val,
                                   r_in = val + syntax['r_in_size']   * fwhm,
                                   r_out = val + syntax['r_out_size'] * fwhm
                                   )

                df['flux_'+str(key)] = ap

            except Exception as e:

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.info(exc_type, fname, exc_tb.tb_lineno,e)

                pass

        df['mag_inst'] = find_mag(df.flux_ap/syntax['exp_time'],0)

        if syntax['save_dataframe']:
            df.to_csv(str(syntax['fname_dataframe']) + '.csv')
    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.info(exc_type, fname, exc_tb.tb_lineno,e)



    return df


def ap_correction(image,syntax,df):

    import numpy as np
    from astropy.stats import sigma_clip
    import matplotlib.pyplot as plt
    from autophot.packages.functions import find_mag
    from autophot.packages.functions import set_size
    
    import os

    import logging
    logger = logging.getLogger(__name__)

    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    except:
        pass





    ap_diff = find_mag(df['flux_inf_ap'] / df['flux_ap'],0)
    ap_diff = ap_diff[~np.isnan(ap_diff)]

    corr_mask = np.array(~sigma_clip(ap_diff,
                                     sigma = syntax['ap_corr_sigma'],
                                     cenfunc = 'mean').mask)

    ap_corr = ap_diff[corr_mask]

    if syntax['ap_corr_plot']:

        plt.ioff()

        fig = plt.figure(figsize = set_size(250,1))


        ax1 = fig.add_subplot(111)

        import numpy as np
        from scipy.stats import norm
        import matplotlib.pyplot as plt


        # Fit a normal distribution to the data:
        mu, std = norm.fit(ap_corr)

        # Plot the histogram.
        ax1.hist(ap_corr, bins=25, density=True,label = 'Aperture Correction')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        ax1.plot(x, p, 'k', linewidth=0.5,label = 'PDF',color = 'r')

        ax1.set_xlabel(r'Aperature Correction [$-2.5 log_{10}(\frac{\sum Infinite \ Aperture}{\sum Aperture})]$')
        ax1.set_ylabel('Probability Denisty')

        ax1.legend(loc = 'best',
                   frameon = False)

        fig.savefig(syntax['write_dir']+'APCOR.pdf',
                    format = 'pdf',
                    bbox_inches='tight'
                    )
        plt.close()


    logger.info('Aperture correction: %.3f +/- %.3f' % (np.nanmean(ap_corr),np.nanstd(ap_corr)))
    ap_corr = np.nanmean(ap_corr)
    return ap_corr


#
#
#
#