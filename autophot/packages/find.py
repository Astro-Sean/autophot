#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:13:14 2018

@author: seanbrennan
"""

def fwhm(image,syntax,sigma_lvl = None,fwhm = None):


    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    '''
    Find full width half maxiumu of an image via gaussian fitting to isolated bright stars

    Can be very tempromental for bad images
    '''

    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma

    from autophot.packages.functions import moffat_2d,moffat_fwhm
    import numpy as np
    import pandas as pd
    import lmfit
    from astropy.stats import sigma_clip
    import logging

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



    image_params = []
    isolated_sources = []

    img_seg = [0]
    try:
        for idx in range(len(list(img_seg))):

            mean, median, std = sigma_clipped_stats(image,
                                                    sigma=syntax['fwhm_sigma'],
                                                    maxiters=syntax['fwhm_iters'])

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



            # Remove target by masking with area with that of median image value - just so it's not picked up
            if syntax['target_name'] != None and fwhm == None:

                logger.info('Target location : (x,y) -> (%.3f,%.3f)' % (syntax['target_x_pix'] , syntax['target_y_pix']))

                if abs(syntax['target_x_pix'])>=search_image.shape[1] or abs(syntax['target_y_pix'])>=search_image.shape[0]:
                    logger.warning('Target not located on images')
                    raise Exception ( ' *** EXITING - Target pixel coordinates outside of image [%s , %s] *** ' % (int(syntax['target_y_pix']), int(syntax['target_y_pix'])))

                else:
                    search_image[int(syntax['target_y_pix'])-syntax['int_scale']: int(syntax['target_y_pix']) + syntax['int_scale'],
                                 int(syntax['target_x_pix'])-syntax['int_scale']: int(syntax['target_x_pix']) + syntax['int_scale']] =  syntax['global_median'] * np.ones((int(2*syntax['int_scale']),int(2*syntax['int_scale'])))


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

                        daofind = DAOStarFinder(fwhm      = np.ceil(int_fwhm),
                                                threshold = threshold_value*std,
                                                sharplo   =  0.2,sharphi = 1.0,
                                                roundlo   = -1.0,roundhi = 1.0
                                                )

                        sources = daofind(search_image - median)

                        if sources == None:
                            logger.warning('Sources == None')
                            m = fudge_factor
                            continue

                        sources = sources.to_pandas()

                        logger.info('Number of sources before cleaning - [s = %.1f]: %d ' % (threshold_value,len(sources)))

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
                            # print('m: %s :: n %s' % (m,n))
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


                        if len(sources) > 30:

                            if syntax['remove_boundary_sources']:

                                with_boundary = len(sources)

                                sources = sources[sources['xcentroid'] < image.shape[1] - syntax['pix_bound'] ]
                                sources = sources[sources['xcentroid'] > syntax['pix_bound'] ]
                                sources = sources[sources['ycentroid'] < image.shape[0] - syntax['pix_bound'] ]
                                sources = sources[sources['ycentroid'] > syntax['pix_bound'] ]

                                logger.debug('Removed %d sources near boundary' % (with_boundary - len(sources)))

                        x = np.array(sources['xcentroid'])
                        y = np.array(sources['ycentroid'])

                        iso_temp = []
                        pix_dist = []

                        if len(sources) < min_source_no:
                            logger.warning('Less than min source after boundary removal')
                            m = fudge_factor
                            continue

                        if sigma_lvl != None or len(sources) < 10:
                            isolated_sources = pd.DataFrame({'x_pix':x,'y_pix':y})

                        else:

                            for idx in range(len(x)):
                                try:

                                    x0 = x[idx]
                                    y0 = y[idx]

                                    dist = np.sqrt((x0-np.array(x))**2+(y0-np.array(y))**2)

                                    dist = dist[np.where(dist!=0)]

                                    isolated_dist = 0

                                    if syntax['isolate_sources']:

                                        isolated_dist = syntax['iso_scale']

                                    if len(dist) == 0:
                                        dist = [0]

                                    if min(list(dist)) >  isolated_dist:

                                        df = np.array((float(x0),float(y0)))

                                        iso_temp.append(df)

                                        pix_dist.append(dist)
                                except Exception as e:
                                    logger.exception(e)
                                    pass

                            if len(iso_temp) == 0:
                                logger.warning('Less than min source after isolating sources')
                                m = fudge_factor
                                continue

                            isolated_sources= pd.DataFrame(data = iso_temp)
                            isolated_sources.columns = ['x_pix','y_pix']
                            isolated_sources.reset_index()



                        x_rc = []
                        y_rc = []

                        fwhm_list=[]
                        medianlst=[]

                        x_pix = np.arange(0,2 * syntax['int_scale'])
                        y_pix = np.arange(0,2 * syntax['int_scale'])

                        image_copy = image.copy()

                        for idx in isolated_sources.index:

                            try:


                                x0 = isolated_sources['x_pix'].loc[[idx]]
                                y0 = isolated_sources['y_pix'].loc[[idx]]

                                close_up = image_copy[int(y0 - syntax['int_scale']): int(y0 + syntax['int_scale']),
                                                      int(x0 - syntax['int_scale']): int(x0 + syntax['int_scale'])]

                                if close_up.shape != (int(2*syntax['int_scale']),int(2*syntax['int_scale'])):
                                    fwhm_list.append(np.nan)
                                    x_rc.append(np.nan)
                                    y_rc.append(np.nan)
                                    medianlst.append(np.nan)
                                    logger.warning('wrong close-up size')
                                    continue

                                mean, median_val, std = sigma_clipped_stats(close_up,
                                                                            sigma=syntax['fwhm_sigma'],
                                                                            maxiters=syntax['fwhm_iters'])
                                medianlst.append(median_val)
                                xx, yy = np.meshgrid(x_pix, y_pix)


                                if syntax['remove_sat']:
                                    try:
                                        saturation_lvl = syntax['sat_lvl']
                                    except:
                                        saturation_lvl = 2**16

                                    # if np.nanmax(close_up) >= saturation_lvl:
                                    #     print('Saturated image')
                                    #     fwhm_list.append(np.nan)
                                    #     x_rc.append(np.nan)
                                    #     y_rc.append(np.nan)
                                    #     continue


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



                                     if syntax['use_moffat']:
                                         source_image_params = dict(alpha=result.params['alpha'].value,beta=result.params['beta'].value)
                                     else:
                                         source_image_params = dict(sigma=result.params['sigma'])


                                     fwhm_fit = fitting_model_fwhm(source_image_params)

                                     image_params.append(source_image_params)

                                     fwhm_list.append(fwhm_fit)

                                     x_rc.append(result.params['x0'] - syntax['int_scale'] + x0)
                                     y_rc.append(result.params['y0'] - syntax['int_scale'] + y0)


                                     if syntax['remove_sat']:

                                        if result.params['A'].value >= syntax['sat_lvl']:
                                            print('-Saturated source')
                                            fwhm_list.append(np.nan)
                                            x_rc.append(np.nan)
                                            y_rc.append(np.nan)
                                            pass





                                except Exception as e:
                                    logger.exception(e)

                                    fwhm_list.append(np.nan)
                                    x_rc.append(np.nan)
                                    y_rc.append(np.nan)

                                    pass

                            except Exception as e:
                                logger.exception(e)

                                fwhm_list.append(np.nan)
                                x_rc.append(np.nan)
                                y_rc.append(np.nan)
                                medianlst.append(median_val)

                                continue
                        isolated_sources['FWHM'] = pd.Series(fwhm_list)
                        isolated_sources['median'] = pd.Series(medianlst)
                        if sigma_lvl == None:



                            if isolated_sources['FWHM'].values == np.array([]):
                                logger.info('> No sigma values taken <')
                                continue
                            try:

                                if len(isolated_sources) > 5:

                                    isolate_mask = sigma_clip(isolated_sources['FWHM'].values, sigma=3).mask

                                    isolated_sources = isolated_sources[~isolate_mask]

                            except:

                                isolated_sources = []

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

                            logger.info('Isolated sources found [ %d sigma ]: %d' % (threshold_value,len(isolated_sources)))

                            mean_fwhm =  np.nanmean(isolated_sources['FWHM'])



                            if np.isnan(mean_fwhm):
                                m = fudge_factor
                                continue


                        else:
                            isolate_mask = sigma_clip(isolated_sources['FWHM'].values, sigma=3).mask

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


            mask = np.array(sigma_clip(val, sigma=3).mask)
            # print(mask)

            image_params_out[key] = np.nanmean(val[~mask])
            image_params_out[key+'_err'] = np.nanstd(val[~mask])

        syntax['image_params'] =image_params_out

        mean_fwhm = fitting_model_fwhm(image_params_out)

        syntax['scale'] = int(np.ceil(syntax['scale_multipler'] * mean_fwhm)) + 0.5



        return mean_fwhm,isolated_sources,syntax

    except Exception as e:
        logger.exception(e)
        return np.nan



# def image_radius(image,syntax,df,fwhm,LOG=None):

#     from photutils import centroid_2dg
#     import numpy as np
#     from autophot.packages.aperture import ap_phot
#     from astropy.stats import sigma_clip
#     from autophot.packages.rm_bkg import BKG_SUBTRACTION


#     radius = [ ]
#     for x,y in zip(df['x_pix'],df['y_pix']):
#         try:

#             close_up = image[int(y)- syntax['scale']: int(y) + syntax['scale'],
#                              int(x)- syntax['scale']: int(x) + syntax['scale']]

#             bkg = BKG_SUBTRACTION(close_up  = close_up,
#                                   syntax =  syntax,
#                                   center = [syntax['scale'],syntax['scale']],
#                                   radius = fwhm)

#             close_up, bkg ,_ = bkg.find()


#             ap_sum = [ ]

#             xc_guess,yc_guess = centroid_2dg(close_up)

#             ap_range = np.arange(0.1,syntax['scale']/fwhm,1/25)

#             for n in ap_range:

#                 ap,bkg = ap_phot( (xc_guess,yc_guess) ,
#                                  close_up,
#                                  radius = n * fwhm,
#                                  r_in = syntax['r_in_size'] * fwhm),

#                 ap_sum.append(float(ap))

#             ap_sum = np.array(ap_sum)/np.nanmax(ap_sum)


#             radius.append(ap_range[np.argmax(ap_sum>=syntax['norm_count_sum'])])

#         except Exception as e:
#             logger.info('Problem with image_radius:',e)
#             pass

#     radius_mul = sigma_clip(radius).compressed()

#     radius_mul = np.mean(radius_mul)
#     image_radius = radius_mul*fwhm

#     syntax['image_radius'], =float(image_radius)

#     logger.info('image radius updated: ' +str(radius_mul*fwhm))

#     return syntax


def phot(image,syntax,df,fwhm,LOG=None,):

    import sys
    import numpy as np
    import os
    import logging

    from autophot.packages.aperture import ap_phot
    from autophot.packages.functions import mag

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

        df['mag_inst'] = mag(df.flux_ap/syntax['exp_time'],0)

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
    from autophot.packages.functions import mag
    from autophot.packages.functions import set_size

    import logging
    logger = logging.getLogger(__name__)

    plt.ioff()


    ap_diff = mag(df['flux_inf_ap'] / df['flux_ap'],0)
    ap_diff = ap_diff[~np.isnan(ap_diff)]

    corr_mask = np.array(~sigma_clip(ap_diff, sigma = syntax['ap_corr_sigma']).mask)

    ap_corr = ap_diff[corr_mask]

    if syntax['ap_corr_plot']:

        fig = plt.figure(figsize = set_size(240,1))


        ax1 = fig.add_subplot(111)
        ax1.hist([ap_diff,ap_corr],
                 weights = [ np.ones_like(ap_diff)/float(len(ap_diff)),np.ones_like(ap_corr)/float(len(ap_corr))],
                 bins = 30,
                 label = ['w/o Clipping','w/ Clipping [$%d\sigma$]' % syntax['ap_corr_sigma']],
                 density = False)

        # ax1.hist(ap_corr,bins = 'auto',label = ,density = True)

        ax1.set_xlabel(r'$-2.5 log_{10}(\frac{\sum Infinite \ Aperture}{\sum Aperture})$')
        ax1.set_ylabel('Probability')
        ax1.legend(loc = 'best',
                   frameon = False)

        fig.savefig(syntax['write_dir']+'APCOR.pdf',
                                                    format = 'pdf',
                                                    bbox_inches='tight'
                                                    )
        plt.close()

        # plt.show()


    logger.info('Aperture correction: %.3f +/- %.3f' % (np.nanmean(ap_corr),np.nanstd(ap_corr)))
    ap_corr = np.nanmean(ap_corr)
    return ap_corr


#
#
#
#
