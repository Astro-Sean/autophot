#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 29 12:24:07 2019

@author: seanbrennan
"""
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from autophot.packages.functions import gauss_2d,moffat_2d

from autophot.packages.functions import scale_roll

from autophot.packages.functions import rebin

from autophot.packages.functions import gauss_fwhm2sigma,set_size,order_shift

from autophot.packages.rm_bkg import rm_bkg

from astropy.stats import sigma_clip

# =============================================================================
# Point spread function by Me :)
# =============================================================================

def build_r_table(base_image,selected_sources,syntax,fwhm):

    '''
     Build tables of residuals from bright isolated sources given in selected_sources dataframe

     Function will function selected function to these sources and normialise there residaul array to build a residual image
     which will then be used to make a PSF for the image

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
        # m = regriding_size

        if regriding_size % 2 > 0:
            logger.info('regrid size must be even adding 1')
            regriding_size += 1

        # FWHM/sigma fits
        fwhm_fit = []

        # what sources will be used
        construction_sources = []

        # Residual Table in extended format
        residual_table = np.zeros((int(2 * syntax['scale'] * regriding_size), int(2 * syntax['scale']*regriding_size)))


        # if syntax['remove_sat']:
        #     len_with_sat = len(selected_sources)
        #     selected_sources = selected_sources[selected_sources['flux_ap']+selected_sources['median']<= syntax['sat_lvl']]
        #     print('%d saturdated PSF stars removed' % (len_with_sat-len(selected_sources)))

        selected_sources['dist'] =  pix_dist(syntax['target_x_pix'],selected_sources.x_pix,
                                             syntax['target_y_pix'],selected_sources.y_pix)

        selected_sources_mask = sigma_clip(selected_sources['median'], sigma=3, maxiters=5,masked=True)

        selected_sources = selected_sources[~selected_sources_mask.mask]






        if syntax['use_local_stars_for_PSF']:

            '''
            Use local stars given by 'use_acrmin' parameter

            '''
            selected_sources_test = selected_sources[selected_sources['dist'] <= syntax['local_radius']]

            selected_sources  = selected_sources_test


        flux_idx = [i for i in selected_sources.flux_ap.sort_values(ascending = False).index]

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


                mean, median, std = sigma_clipped_stats(psf_image, sigma = syntax['source_sigma_close_up'], maxiters = syntax['iters'])

                daofind = DAOStarFinder(fwhm=np.floor(fwhm),
                                     threshold = syntax['lim_SNR']*std,
                                     roundlo = -1.0, roundhi = 1.0,
                                     sharplo =  0.2, sharphi = 1.0)


                sources = daofind(psf_image - median)

                if sources is None:
                    sources = []

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

                psf_image_bkg_free,bkg_surface = rm_bkg(psf_image,syntax,psf_image.shape[0]/2,psf_image.shape[0]/2)


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


                mini = lmfit.Minimizer(residual, pars,nan_policy = 'omit')
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
                refit only focusing on highest SNR area given by fitting radius

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

                mini = lmfit.Minimizer(residual, pars,nan_policy = 'omit')
                result = mini.minimize(method = 'least_squares')
                # print(result.params)

                positions  = list(zip([xc_global ],[yc_global ]))

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
                    continue
                else:
                    logger.info('SNR: %d FWHM: %.3f' % (PSF_SNR,PSF_FWHM))
                    # print('\rPSF source %d / %d :: SNR: %d' % (int(PSF_SNR)),end = '')
                    pass

                # print(result.params)
                xc = result.params['x0'].value
                yc = result.params['y0'].value

                H = result.params['A'].value
                H_err = result.params['A'].stderr


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



                psf_mag.append(-2.5*np.log10(H))

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
                sources_dict['PSF_%d'%sources_used]['x_best'] = xc_correction
                sources_dict['PSF_%d'%sources_used]['y_best'] = yc_correction


                sources_dict['PSF_%d'%sources_used]['close_up'] = psf_image_bkg_free
                sources_dict['PSF_%d'%sources_used]['residual'] = residual
                sources_dict['PSF_%d'%sources_used]['regrid'] = residual_regrid
                sources_dict['PSF_%d'%sources_used]['roll'] = residual_roll
                sources_dict['PSF_%d'%sources_used]['x_roll'] =x_roll
                sources_dict['PSF_%d'%sources_used]['y_roll'] =y_roll

                logger.debug('Residual table updated: %d / %d ' % (sources_used,syntax['psf_source_no']))

                print('\rResidual table updated: %d / %d ' % (sources_used,syntax['psf_source_no']) ,end = '')
                sources_used +=1

                fwhm_fit.append(PSF_FWHM)


            except Exception as e:
                # logger.exception(e)

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
        residual_table/= sources_used

        # regrid residual table to psf size
        residual_table  = rebin(residual_table,( int(2*syntax['scale']),int(2*syntax['scale'])))

        # construction_sources = pd.DataFrame(construction_sources)
        construction_sources = pd.DataFrame.from_dict(sources_dict, orient='index',
                                                      columns=['x_pix','y_pix','H_psf','H_psf_err','fwhm','x_best','y_best'])
        construction_sources.reset_index(inplace = True)

        if syntax['plots_PSF_residual']:
            from autophot.packages.create_plots import plot_PSF_model_steps
            plot_PSF_model_steps(sources_dict,syntax,image)

        if syntax['plots_PSF_sources']:

            from autophot.packages.create_plots import plot_PSF_construction_grid

            plot_PSF_construction_grid(construction_sources,image,syntax)


        image_radius_lst = np.array(image_radius_lst)

        syntax['image_radius'] = image_radius_lst.mean()

        logger.info('Image_radius [pix] : %.3f +/- %.3f' % (image_radius_lst.mean(), image_radius_lst.std()))

    except Exception as e:
        logger.exception('BUILDING PSF: ',e)
        raise Exception

    return residual_table,fwhm_fit,construction_sources,syntax


def fit(image,sources,residual_table,syntax,fwhm,
        return_psf_model = False,
        save_plot = False,show_plot = False,
        rm_bkg_val = True,hold_pos = False,
        return_fwhm = False,return_subtraction_image = False,
        fname = None,no_print = False

        ):



    '''

    Fitting of PSF model to source

    '''


    import numpy as np
    import pandas as pd
    import pathlib
    import lmfit
    import logging

    import matplotlib.pyplot as plt

    from autophot.packages.functions import gauss_2d,moffat_2d,moffat_fwhm,gauss_sigma2fwhm

    from matplotlib.gridspec import  GridSpec

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))


    logger = logging.getLogger(__name__)


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

                # print((top, bottom), (left, right))

                psf_shape = pad_shape

                r_table = np.pad(r_table, [(top, bottom), (left, right)], mode='constant', constant_values=0)

            x_rebin = np.arange(0,psf_shape[0])
            y_rebin = np.arange(0,psf_shape[1])

            xx_rebin,yy_rebin = np.meshgrid(x_rebin,y_rebin)

            # sigma  = fwhm/(2*np.sqrt(2*np.log(2)))
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

            if slice_scale != None:
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

        from astropy.visualization.mpl_normalize import ImageNormalize

        from astropy.visualization import  ZScaleInterval, SquaredStretch


        norm = ImageNormalize( stretch = SquaredStretch())

        vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(image)

    '''
    Known issue - for poor images, some sources may be too close to boundary, remove this
    '''
    if not return_fwhm and not no_print:
        logger.info('Image cutout size: (%.f,%.f) (%.f,%.f)' % ((lower_x_bound,upper_x_bound,lower_y_bound,upper_y_bound)))

    sources = sources[sources.x_pix < image.shape[1] - upper_x_bound]
    sources = sources[sources.x_pix > lower_x_bound]
    sources = sources[sources.y_pix < image.shape[0] - upper_y_bound]
    sources = sources[sources.y_pix > lower_y_bound]

    if not no_print:
        logger.info('Fitting PSF to %d sources' % len(sources))

    for n  in range(len(sources.index)):
        if not return_fwhm and not no_print:
            print('\rFitting PSF to source: %d / %d' % (n+1,len(sources)), end = '')

        try:

            idx = list(sources.index)[n]


            source_base =   image[int(sources.y_pix[idx]-lower_y_bound): int(sources.y_pix[idx] + upper_y_bound),
                                  int(sources.x_pix[idx]-lower_x_bound): int(sources.x_pix[idx] + upper_x_bound)]

            if source_base.shape != (int(2*syntax['scale']),int(2*syntax['scale'])):
                print('not right shape')

                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                x_fitted = np.nan
                y_fitted = np.nan
                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))
                continue

            xc = syntax['scale']
            yc = syntax['scale']

            xc_global =  sources.x_pix[idx]
            yc_global =  sources.y_pix[idx]

            if not rm_bkg_val:
                source_bkg_free = source_base
                bkg_median = 0
            else:

                try:
                    source_bkg_free,bkg_surface = rm_bkg(source_base,syntax,source_base.shape[1]/2,source_base.shape[0]/2)
                    bkg_median = np.nanmedian(bkg_surface)

                except Exception as e:
                    bkg_median = np.nan
                    H = np.nan
                    H_psf_err = np.nan
                    x_fitted = np.nan
                    y_fitted = np.nan
                    psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))
                    logger.exception(e)
                    continue


            source = source_bkg_free[int(0.5*source_bkg_free.shape[1] - fitting_radius):int(0.5*source_bkg_free.shape[1] + fitting_radius) ,
                                     int(0.5*source_bkg_free.shape[0] - fitting_radius):int(0.5*source_bkg_free.shape[0] + fitting_radius) ]

            if source.shape != (int(2*fitting_radius),int(2*fitting_radius)):
                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                x_fitted = np.nan
                y_fitted = np.nan
                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))
                continue

            if np.sum(np.isnan(source)) == len(source):
                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                x_fitted = np.nan
                y_fitted = np.nan
                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))

                continue


            if hold_pos:
                dx = 1e-6
                dy = 1e-6
            else:
                dx = syntax['dx']
                dy = syntax['dy']
            # if not return_fwhm:
            #     dx = syntax['dx']
            #     dy = syntax['catalogdy']


            x_slice = np.arange(0,2*fitting_radius)
            xx_sl,yy_sl= np.meshgrid(x_slice,x_slice)

            if return_fwhm :
                if not no_print:
                    logger.info('Fitting gaussian to source to get FWHM')

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmax(source),min = 0)
                pars.add('x0',value = source.shape[1]/2,min = 0.5*source.shape[1] - dx,max = 0.5*source.shape[1] + dx)
                pars.add('y0',value = source.shape[0]/2,min = 0.5*source.shape[0] - dy,max = 0.5*source.shape[0] + dy)

                # print(pars)

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

                mini = lmfit.Minimizer(residual, pars,nan_policy = 'omit')
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


                #  shift image and increase shize of image by shift

                # xc_global = sources.x_pix[idx]
                # yc_global = sources.y_pix[idx]

                source_base =   image[int(yc_global-lower_y_bound): int(yc_global + upper_y_bound),
                                      int(xc_global-lower_x_bound): int(xc_global + upper_x_bound)]

                source_bkg_free,bkg_surface = rm_bkg(source_base,syntax,source_bkg_free.shape[0]/2,source_bkg_free.shape[1]/2)

                bkg_median = np.nanmedian(bkg_surface)

                source = source_bkg_free[int(source_bkg_free.shape[0]/2 - fitting_radius):int(source_bkg_free.shape[0]/2 + fitting_radius) ,
                                         int(source_bkg_free.shape[1]/2 - fitting_radius):int(source_bkg_free.shape[1]/2 + fitting_radius) ]

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

                pars.add('x0',value = 0.5*residual_table.shape[1],
                         min = 0.5*residual_table.shape[1]-dx,
                         max = 0.5*residual_table.shape[1]+dx)

                pars.add('y0',value = 0.5*residual_table.shape[0],
                         min = 0.5*residual_table.shape[0]-dy,
                         max = 0.5*residual_table.shape[0]+dy)


                def residual(p):
                    p = p.valuesdict()
                    res = ((source - build_psf(p['x0'],p['y0'],0,p['A'],residual_table,slice_scale = source.shape[0]/2)))
                    return res.flatten()


                mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit',
                                       scale_covar=True)

                result = mini.minimize(method = 'least_squares')

                xc = result.params['x0'].value

                yc = result.params['y0'].value

                H = result.params['A'].value

                H_psf_err = result.params['A'].stderr

                x_fitted = xc -0.5*residual_table.shape[1] + xc_global
                y_fitted = yc -0.5*residual_table.shape[1] + yc_global

                # print(H,bkg_median)

                if syntax['remove_sat']:

                    if H+bkg_median >= syntax['sat_lvl']:
                        # print('here')
                        bkg_median = np.nan
                        H = np.nan
                        H_psf_err = np.nan
                        psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))
                        continue


            except Exception as e:
                logger.exception(e)
                bkg_median = np.nan
                H = np.nan
                H_psf_err = np.nan
                psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))
                continue

            if syntax['use_covarience']:

                H_psf_err = result.params['A'].stderr

            else:
                logger.warning('Error not computed')
                H_psf_err = 0

            psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))

            if return_subtraction_image:

                try:

                    image_section = image[int(yc_global -  syntax['scale']): int(yc_global + syntax['scale']),
                                          int(xc_global -  syntax['scale']): int(xc_global + syntax['scale'])]

                    image[int(yc_global  - syntax['scale']): int(yc_global +  syntax['scale']),
                          int(xc_global  - syntax['scale']): int(xc_global +  syntax['scale'])] =  image_section - build_psf(xc , yc, 0, H, residual_table)

                    image_section_subtraction = image_section - build_psf(xc , yc, 0, H, residual_table)


                    fig, ax1, = plt.subplots()

                    ax_before = ax1.inset_axes([0.95, 0.70, 0.4, 0.25])
                    ax_after  = ax1.inset_axes([0.95, 0.20, 0.4, 0.25])

                    ax1.imshow(image,
                               vmin = vmin,
                               vmax = vmax,
                               norm = norm,
                               origin = 'lower',
                               cmap = 'gist_heat',
                               interpolation = 'nearest')

                    ax1.set_xlim(0,image.shape[0])
                    ax1.set_ylim(0,image.shape[1])

                    ax1.scatter(xc_global,
                                yc_global,
                                marker = 'o',
                                facecolor = 'None',
                                color = 'green',
                                s = 25)


                    ax_before.imshow(image_section,
                                      vmin = vmin,
                                      vmax = vmax,
                                      norm = norm,
                                      origin = 'lower',
                                      cmap = 'gist_heat',
                                     interpolation = 'nearest')

                    ax_after.imshow(image_section_subtraction,
                                     vmin = vmin,
                                     vmax = vmax,
                                     norm = norm,
                                     origin = 'upper',
                                     cmap = 'gist_heat',
                                     interpolation = 'nearest')

                    ax_after.axis('off')
                    ax_before.axis('off')
                    ax1.axis('off')

                    ax_after.set_title('After')
                    ax_before.set_title('Before')

                    logger.info('Image %s / %s saved' % (str(idx),str(len(sources.index))))
                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')
                    pass





            if syntax['show_residuals'] or show_plot == True or save_plot == True:
                fig = plt.figure(figsize = set_size(500,aspect =0.5))
                try:
                    from astropy.visualization import  ZScaleInterval

                    vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(source_base)

                    h, w = source_bkg_free.shape

                    x  = np.linspace(0, int(2*syntax['scale']), int(2*syntax['scale']))
                    y  = np.linspace(0, int(2*syntax['scale']), int(2*syntax['scale']))

                    X, Y = np.meshgrid(x, y)

                    ncols = 6
                    nrows = 3

                    heights = [1,1,0.75]
                    widths = [1,1,0.75,1,1,0.75]

                    grid = GridSpec(nrows, ncols ,wspace=0.5, hspace=0.5,
                                    height_ratios=heights,width_ratios = widths
                                    )
                    # grid = GridSpec(nrows, ncols ,wspace=0.3, hspace=0.5)



                    ax1   = fig.add_subplot(grid[0:2, 0:2])
                    ax1_B = fig.add_subplot(grid[2, 0:2])
                    ax1_R = fig.add_subplot(grid[0:2, 2])

                    ax2 = fig.add_subplot(grid[0:2, 3:5])

                    ax2_B = fig.add_subplot(grid[2, 3:5])
                    ax2_R = fig.add_subplot(grid[0:2, 5])


                    ax1_B.set_xlabel('X Pixel')
                    ax2_B.set_xlabel('X Pixel')

                    ax1.set_ylabel('Y Pixel')

                    # ax1.set_title('H:%d' % (H+bkg_median))
                    # ax2.set_ylabel('Y Pixel')




                    ax1_R.yaxis.tick_right()
                    ax2_R.yaxis.tick_right()


                    ax1.xaxis.tick_top()
                    ax2.xaxis.tick_top()

                    ax2.axes.yaxis.set_ticklabels([])
                    # ax2_R.axes.yaxis.set_ticklabels([])


                    bbox=ax1_R.get_position()
                    offset= -0.04
                    ax1_R.set_position([bbox.x0+ offset, bbox.y0 , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax2_R.get_position()
                    offset= -0.04
                    ax2_R.set_position([bbox.x0+ offset, bbox.y0 , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax1_B.get_position()
                    offset= 0.08
                    ax1_B.set_position([bbox.x0, bbox.y0+ offset , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])

                    bbox=ax2_B.get_position()
                    offset= 0.08
                    ax2_B.set_position([bbox.x0, bbox.y0+ offset , bbox.x1-bbox.x0, bbox.y1 - bbox.y0])


                    ax1.imshow(source_base,
                               origin = 'lower',
                               aspect="auto",
                               vmin = vmin,
                               vmax = vmax,
                               interpolation = 'nearest'
                               )

                    ax1.scatter(xc,yc,label = 'Best fit',
                                marker = '+',
                                color = 'red',
                                s = 20)


                    ax1_R.plot(source_base[:,w//2],Y[:,w//2],marker = 'o',color = 'blue')
                    ax1_B.plot(X[h//2,:],source_base[h//2,:],marker = 'o',color = 'blue')

                    # include surface
                    ax1_R.plot(bkg_surface[:,w//2],Y[:,w//2],marker = 's',color = 'red')
                    ax1_B.plot(X[h//2,:],bkg_surface[h//2,:],marker = 's',color = 'red')

                    fitted_source = build_psf(xc,yc,0,H,residual_table)

                    # include fitted_source
                    ax1_R.plot((bkg_surface+fitted_source)[:,w//2],Y[:,w//2],marker = 's',color = 'green')
                    ax1_B.plot(X[h//2,:],(bkg_surface+fitted_source)[h//2,:],marker = 's',color = 'green')

                    '''
                    Subtracted image
                    '''

                    import matplotlib.ticker as ticker

                    ax1_B_yticks = np.array(ax1_B.get_yticks())
                    scale = order_shift(abs(ax1_B_yticks))
                    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))

                    ax1_B.set_ylabel('$10^{%d}$ counts' % np.log10(order_shift(abs(ax1_B_yticks))))
                    ax1_B.yaxis.set_major_formatter(ticks)

                    ax1_R_yticks = np.array(ax1_R.get_xticks())
                    scale = order_shift(abs(ax1_R_yticks))
                    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))

                    ax1_R.set_xlabel('$10^{%d}$ counts' % np.log10(order_shift(abs(ax1_R_yticks))))
                    ax1_R.xaxis.set_major_formatter(ticks)

                    ax2_B_yticks = np.array(ax2_B.get_yticks())
                    scale = order_shift(abs(ax2_B_yticks))
                    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))

                    ax2_B.set_ylabel('$10^{%d}$ counts' % np.log10(order_shift(abs(ax2_B_yticks))))
                    ax2_B.yaxis.set_major_formatter(ticks)

                    ax2_R_yticks = np.array(ax2_R.get_xticks())
                    scale = order_shift(abs(ax2_R_yticks))
                    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))

                    ax2_R.set_xlabel('$10^{%d}$ counts' % np.log10(order_shift(abs(ax2_R_yticks))))
                    ax2_R.xaxis.set_major_formatter(ticks)

                    subtracted_image = source_bkg_free - fitted_source + bkg_surface

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



                    if save_plot == True:
                        fig.savefig(syntax['write_dir']+'target_psf_'+fname+'.pdf',
                                                    format = 'pdf',
                                                    # bbox_inches='tight'
                                                    )

                        logger.info('Image %s / %s saved' % (str(n+1),str(len(sources.index)) ))
                    else:

                        pathlib.Path(syntax['write_dir']+'/'+'psf_subtractions/').mkdir(parents = True, exist_ok=True)

                        plt.savefig(syntax['write_dir']+'psf_subtractions/'+'psf_subtraction_{}.png'.format(int(n)))

                    plt.close(fig)

                except Exception as e:
                    logger.exception(e)
                    plt.close('all')


        except Exception as e:
             logger.exception(e)


             bkg_median = np.nan
             H = np.nan
             H_psf_err = np.nan
             psf_params.append((idx,x_fitted,y_fitted,bkg_median,H,H_psf_err))

             continue


    new_df =  pd.DataFrame(psf_params,columns = ('idx','x_fitted','y_fitted','bkg','H_psf','H_psf_err'),index = sources.index)

    if return_fwhm:
        new_df['target_fwhm'] = target_PSF_FWHM,
        # new_df['target_fwhm_err'] =source_fwhm_err
    elif not no_print:
        print('  ')
    if not return_psf_model:


        return pd.concat([sources,new_df],axis = 1),build_psf






'''
Perform PSF calculations
'''

def do(df,residual,syntax,fwhm):

    try:
        from photutils import CircularAperture
        from photutils import aperture_photometry
        from scipy.integrate import dblquad
        import logging


        logger = logging.getLogger(__name__)

        xc = syntax['scale']
        yc = syntax['scale']

        # Integration radius
        # int_scale = 2*syntax['image_radius']
        int_scale = syntax['ap_size'] * fwhm

        int_range_x = [xc - int_scale , xc + int_scale]
        int_range_y = [yc - int_scale , yc + int_scale]


        # Core Gaussian component with height 1 and sigma value sigma
        if syntax['use_moffat']:
            core= lambda y, x: moffat_2d((x,y),syntax['scale'],syntax['scale'],0,1,syntax['image_params'])
        else:
            core= lambda y, x: gauss_2d((x,y),syntax['scale'],syntax['scale'],0,1,syntax['image_params'])

        core_int = dblquad(core, int_range_y[0],int_range_y[1],lambda x:int_range_x[0],lambda x:int_range_x[1])[0]

        # core_int = 2*np.pi*sigma**2


        # Aperture Photometry over residual
        apertures = CircularAperture((syntax['scale'],syntax['scale']), r=int_scale)
        phot_table = aperture_photometry(residual, apertures,method='subpixel',subpixels=4)

        phot_table['aperture_sum'].info.format = '%.8g'
        residual_int = phot_table[0]

        # Counts from core compoent on PSF
        syntax['c_counts'] = float(core_int)

        # Counts from Residual component of PSF
        syntax['r_counts'] = float(residual_int['aperture_sum'])

        # Counts in a PSF with fwhm 2 sqrt(2 ln 2) * sigma and height 1
        sudo_psf = core_int+float(residual_int['aperture_sum'])

        psf_int     = df.H_psf     * sudo_psf
        psf_int_err = df.H_psf_err * sudo_psf

        df['psf_counts']     = psf_int.values
        df['psf_counts_err'] = psf_int_err.values

    except Exception as e:
        logger.exception(e)
        df = np.nan

    return df,syntax

