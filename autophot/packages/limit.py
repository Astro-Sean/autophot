#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:10:14 2020

@author: seanbrennan
"""

'''

AutoPHoT Limiting Magnitude Module
'''

import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

from autophot.packages.functions import set_size


def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample_with_minimum_distance(n=[0,40], k=4, d=10):
    import random
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample_x = random.sample(range(int(n[0]),int(n[1])-(k-1)*(d-1)), k)
    sample_y = random.sample(range(int(n[0]),int(n[1])-(k-1)*(d-1)), k)
    return [(x + (d-1)*rx,y + (d-1)*ry) for x,y, rx,ry in zip(sample_x,sample_y,ranks(sample_x),ranks(sample_y))]

def limiting_magnitude_prob(syntax,image,model =None,r_table=None):


    '''
    syntax - dict
        dDtionary of input paramters
    image - np.array
        Image of region of interest with target in center of image
    model - function
        - psf function from autophot
    '''
    try:

        from photutils import CircularAperture
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.gridspec as gridspec
        import random
        from scipy.optimize import curve_fit
        import warnings
        from photutils.datasets import make_noise_image
        # from autophot.packages.functions import mag
        from photutils import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        # from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from autophot.packages.rm_bkg import rm_bkg

        from astropy.visualization import  ZScaleInterval

        import logging

        logger = logging.getLogger(__name__)


        limiting_mag_figure = plt.figure(figsize = set_size(240,aspect = 1.5))

        gs = gridspec.GridSpec(2, 2,hspace = 0.5,wspace=0.2)
        ax0 = limiting_mag_figure.add_subplot(gs[:, :-1])

        ax1 = limiting_mag_figure.add_subplot(gs[-1, -1])
        ax2 = limiting_mag_figure.add_subplot(gs[:-1, -1])

        # level for detection - Rule of thumb ~ 5 is a good detection level
        level = syntax['lim_SNR']


        logger.info('Limiting threshold: %d sigma' % level)



        image_no_surface,surface = rm_bkg(image,syntax,image.shape[0]/2,image.shape[0]/2)


        # =============================================================================
        # find and mask sources in close up
        # =============================================================================

        image_mean, image_median, image_std = sigma_clipped_stats(image,
                                        sigma = syntax['source_sigma_close_up'],
                                        maxiters = syntax['iters'])


        daofind = DAOStarFinder(fwhm = syntax['fwhm'],
                                threshold = syntax['bkg_level']*image_std)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # ignore no sources warning
            sources = daofind(image - image_median)

        if sources != None:
            positions = list(zip(np.array(sources['xcentroid']),np.array(sources['ycentroid'])))

            positions.append((image.shape[0]/2,image.shape[1]/2))

        else:
            positions = [(image.shape[0]/2,image.shape[1]/2)]

        # "size" of source
        source_size =  syntax['image_radius']

        pixel_number = int(np.ceil(np.pi*source_size**2))

        # Mask out target region
        mask_ap  = CircularAperture(positions,r = source_size)

        mask = mask_ap.to_mask(method='center')

        mask_sumed = [i.to_image(image.shape) for i in mask]

        if len(mask_sumed) !=1:
            mask_sumed = sum(mask_sumed)
        else:
            mask_sumed = mask_sumed[0]

        mask_sumed[mask_sumed>0] = 1

        logging.info('Number of pixels in star: %d' % pixel_number)


        # Mask out center region
        mask_image  = (image_no_surface) * (1-mask_sumed)


        vmin,vmax = (ZScaleInterval(nsamples = 1500)).get_limits(mask_image)

        excluded_points = mask_image == 0
        exclud_x = excluded_points[0]
        exclud_y = excluded_points[1]

        exclud_zip = list(zip(exclud_x,exclud_y))

        included_points = np.where(mask_image != 0)


        includ_x = list(included_points[0])
        includ_y = list(included_points[1])

        includ_zip = list(zip(includ_x,includ_y))


        # ax2.scatter(exclud_y,exclud_x,color ='black',marker = 'X',alpha = 0.5  ,label = 'excluded_pixels',zorder = 1)
        ax2.scatter(includ_y,includ_x,
                    color ='red',
                    marker = 'x',
                    alpha = 0.5,
                    label = 'included_pixels',
                    zorder = 2)

        number_of_points = 300

        fake_points = {}


        if len(includ_zip) < pixel_number:
            includ_zip=includ_zip+exclud_zip


        for i in range(number_of_points):
            fake_points[i] = []
            # count = 0
            random_pixels = random.sample(includ_zip,pixel_number)
            xp_ran = [i[0] for i in random_pixels]
            yp_ran = [i[1] for i in random_pixels]

            fake_points[i].append([xp_ran,yp_ran])



        fake_sum = {}
        for i in range(number_of_points):

            fake_sum[i] = []

            for j in fake_points[i]:

                for k in range(len(j[0])):

                    fake_sum[i].append(image_no_surface[j[0][k]][j[1][k]])


        fake_mags = {}

        for f in fake_sum.keys():

            fake_mags[f] = np.sum(fake_sum[f])


# =============================================================================
#     Histogram
# =============================================================================

        hist, bins = np.histogram(list(fake_mags.values()),
                                  bins = len(list(fake_mags.values())),
                                  density = True)

        center = (bins[:-1] + bins[1:]) / 2

        sigma_guess = np.nanstd(list(fake_mags.values()))
        mean_guess = np.nanmean(list(fake_mags.values()))
        A_guess = np.nanmax(hist)

        def gauss(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        popt,pcov = curve_fit(gauss,center,hist,
                              p0=[A_guess,mean_guess,sigma_guess],
                              absolute_sigma=True )

        mean = popt[1]
        std  = abs(popt[2])

        logging.info('Mean: %s - std: %s' % (round(mean,3),round(std,3)))

        if syntax['probable_detection_limit']:

            beta = float(syntax['probable_detection_limit_beta'])

            def detection_probability(n,sigma,beta ):
                from scipy.special import erfinv

                '''

                Probabilistic upper limit computation base on:
                http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf

                Assuming Gassauin nose distribution

                n: commonly used threshold value integer above some background level

                sigma: sigma value from noise distribution found from local area around source

                beta: Detection probability


                '''
                flux_upper_limit = (n + np.sqrt(2)*erfinv(2*beta - 1)) * sigma

                return flux_upper_limit

            logging.info("Using Probable detection limit [b' = %d%% ]" % (100 * beta))

            f_ul = mean+detection_probability(level,std,beta)

            logging.info("Flux Upper limit: %.3f" % f_ul)

        else:
            f_ul = abs(mean + level*std)
            logging.info('Detection at %s std: %.3f' % (level,f_ul))





        # =============================================================================
        # Plot histogram of background values
        # =============================================================================

        line_kwargs = dict(alpha=0.5,color='black',ls = '--')

        # the histogram of the data
        n, bins, patches = ax0.hist(list(fake_mags.values()),
                                    density=True,
                                    bins = 30,
                                    facecolor='blue',
                                    alpha=1,
                                    label = 'Pseudo-Flux\nDistribution')

        ax0.axvline(mean,**line_kwargs)
        ax0.axvline(mean + 1*std,**line_kwargs)
        ax0.text(mean + 1*std,np.max(n),r'$1\sigma$',rotation = -90,va = 'top')
        ax0.axvline(mean + 2*std,**line_kwargs)
        ax0.text(mean + 2*std,np.max(n),r'$2\sigma$',rotation = -90,va = 'top')

        if syntax['probable_detection_limit']:

            ax0.axvline(f_ul,**line_kwargs)
            ax0.text(f_ul,np.max(n),r"$\beta'$ = %d%%" % (100*beta),rotation = -90,va = 'top')

        else:
            ax0.axvline(f_ul,**line_kwargs)
            ax0.text(mean + level*std,np.max(n),r'$'+str(level)+r'\sigma$',rotation = -90,va = 'top')

        x_fit = np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 250)

        ax0.plot(x_fit, gauss(x_fit,*popt),label = 'Gaussian Fit',color = 'red')

        ax0.ticklabel_format(axis='y', style='sci',scilimits = (-2,0))
        ax0.yaxis.major.formatter._useMathText = True

        ax0.set_xlabel('Pseudo-Flux')
        ax0.set_ylabel('Normalised Probability')

        im2 = ax2.imshow(image-surface,origin='lower',
                         aspect = 'auto',
                         interpolation = 'nearest')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = limiting_mag_figure.colorbar(im2, cax=cax)
        cb.ax.set_ylabel('Counts', rotation=270,labelpad = 10)

        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

        ax2.set_title('Image - Surface')


        # =============================================================================
        # Convert counts to magnitudes
        # =============================================================================

        flux  = f_ul / syntax['exp_time']

        mag_level = -2.5*np.log10(flux)


        # =============================================================================
        # We now have an upper and lower estimate of the the limiting magnitude
        # =============================================================================

        '''
        Visual display of limiting case

        if PSF model is available use that

        else

        use a gaussian profile with the same number of counts
        '''
        fake_sources = np.zeros(image.shape)


        try:

            if syntax['c_counts']:
                pass

            model_label = 'PSF'

            def mag2image(m):
                '''
                Convert magnitude to height of PSF
                '''
                Amplitude  = (syntax['exp_time']/(syntax['c_counts']+syntax['r_counts']))*(10**(m/-2.5))

                return Amplitude


            # PSF model that matches close-up shape around target
            def input_model(x,y,flux):
                return model(x, y,0,flux,r_table, pad_shape = image.shape)



        except:

            '''
            if PSF model isn't available - use Gaussian instead

            '''
            logging.info('PSF model not available - Using Gaussian')
            model_label = 'Gaussian'

            sigma = syntax['fwhm'] / 2*np.sqrt(2*np.log(2))

            def mag2image(m):
                '''
                Convert magnitude to height of Gaussian
                '''

                #  Volumne/counts under 2d gaussian for a magnitude m
                volume =  (10**(m/-2.5)) * syntax['exp_time']

                # https://en.wikipedia.org/wiki/Gaussian_function
                Amplitude =  volume/(2*np.pi*sigma**2)

                return Amplitude

            #  Set up grid

            def input_model(x,y,A):

                x = np.arange(0,image.shape[0])
                xx,yy= np.meshgrid(x,x)

                from autophot.packages.functions import gauss_2d,moffat_2d

                if syntax['use_moffat']:
                    model = moffat_2d((xx,yy),x,y,0,A,syntax['image_params']).reshape(image.shape)

                else:
                    model = gauss_2d((xx,yy),x,y,0,A,syntax['image_params']).reshape(image.shape)

                return model


        # =============================================================================
        #  What magnitude do you want this target to be?
        # =============================================================================

        mag2image = mag2image

        inject_source_mag = mag2image(mag_level)

        # =============================================================================
        # Random well-spaced points to plot
        # =============================================================================

        random_sources = sample_with_minimum_distance(n = [int(syntax['fwhm']),int(image.shape[0]-syntax['fwhm'])],
                                                      k = syntax['inject_sources_random_number'],
                                                      d = int(syntax['fwhm']/2)
                                                      )
        import math
        def PointsInCircum(r,n=100):
            return [(math.cos(2*math.pi/n*x)*r + image.shape[1]/2 ,math.sin(2*math.pi/n*x)*r + image.shape[0]/2) for x in range(0,n)]


        random_sources = PointsInCircum(2*syntax['fwhm'],n=3)
        x = [abs(i[0]) for i in random_sources]
        y = [abs(i[1]) for i in random_sources]

        print(x)
        print(y)


        # =============================================================================
        # Inject sources
        # =============================================================================

        try:
            if syntax['inject_source_random']:

                for i in range(0,len(x)):

                    fake_source_i = input_model(x[i], y[i],inject_source_mag)




                    if syntax['inject_source_add_noise']:

                        nan_idx = np.isnan(fake_source_i)
                        fake_source_i[nan_idx] = 0
                        fake_source_i[fake_source_i<0] = 0

                        fake_source_i = make_noise_image(fake_source_i.shape,
                                                        distribution = 'poisson',
                                                        mean = fake_source_i,
                                                        random_state = np.random.randint(0,1e3))
                        # fake_source_i[nan_idx] = np.nan1

                    fake_sources += fake_source_i
                    ax1.scatter(x[i],y[i],marker = 'o',s=150, facecolors='none', edgecolors='r',alpha = 0.5)
                    ax1.annotate(str(i), (x[i], -.5+y[i]),color='r',alpha = 0.5,ha='center')


            if syntax['inject_source_on_target']:

                fake_source_on_target = input_model(image.shape[1]/2,image.shape[0]/2,inject_source_mag)

                if syntax['inject_source_add_noise']:
                    nan_idx = np.isnan(fake_source_on_target)
                    fake_source_on_target[nan_idx] = 1e-6
                    fake_source_on_target[fake_source_on_target<0] = 0

                    fake_source_on_target = make_noise_image(fake_source_on_target.shape,
                                                    distribution = 'poisson',
                                                    mean = fake_source_on_target,
                                                    random_state = np.random.randint(0,1e3))

                fake_sources += fake_source_on_target


                ax1.scatter(image.shape[1]/2,image.shape[0]/2,marker = 'o',s=150, facecolors='none', edgecolors='black',alpha = 0.5)
                ax1.annotate('On\nTarget', (image.shape[1]/2, -1+image.shape[0]/2),color='black',alpha = 0.5,ha='center')


            im1 = ax1.imshow(image - surface + fake_sources,
                              # vmin = vmin,
                              # vmax = vmax,
                              aspect = 'auto',
                              # norm = norm,
                              origin = 'lower',
                             interpolation = 'nearest')
            ax1.set_title(' Fake [%s] Sources ' % model_label)

        except Exception as e:
            logging.exception(e)
            im1=ax1.imshow(image - surface , origin='lower',aspect = 'auto',)
            ax1.set_title('[ERROR] Fake Sources [%s]' % model_label)


        # plt.colorbar(im1,ax=ax1)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = limiting_mag_figure.colorbar(im1, cax=cax)
        cb.ax.set_ylabel('Counts', rotation=270,labelpad = 10)
        # cb = fig.colorbar(im)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()


        ax0.legend(loc = 'lower center',bbox_to_anchor=(0.5, 1.02),ncol = 2,frameon = False)

        limiting_mag_figure.savefig(syntax['write_dir']+'limiting_mag_porb.pdf',
                                        # box_extra_artists=([l]),
                                        bbox_inches='tight',
                                        format = 'pdf')
        plt.close('all')




    # master try/except
    except Exception as e:
        print('limit issue')
        logging.exception(e)


    syntax['maglim_mean'] = mean
    syntax['maglim_std'] = std

    return mag_level,syntax








        # if syntax['inject_source_mag']:

        #     user_mag_level = syntax['inject_source_mag'] - syntax['zp']

        #     mag2image = mag2image

        #     inject_source_mag = mag2image(user_mag_level)

        # if  syntax['inject_source_recover']:

        #     print('\nPerforming full magnitude recovery test')

        #     from autophot.packages import psf
        #     import pandas as pd
        #     from photutils.datasets.make import apply_poisson_noise

        #     inserted_magnitude = []
        #     recovered_magnitude = []
        #     recovered_magnitude_e = []
        #     recovered_fwhm = []
        #     recovered_fwhm_e = []
        #     before_after = []
        #     recovered_counts = []
        #     recovered_counts_e = []



        #     if syntax['inject_source_mag']:
        #         start_mag = user_mag_level
        #     else:
        #         start_mag = mag_level-2

        #     nsteps = syntax['inject_source_recover_nsteps']
        #     dmag   = syntax['inject_source_recover_dmag']
        #     redo   = syntax['inject_source_recover_dmag_redo']

        #     print('Starting Magnitude: %.3f' % start_mag)
        #     for i in np.arange(0,nsteps*dmag,dmag):

        #         dx = random.uniform(-1, 1)
        #         dy = random.uniform(-1, 1)

        #         injection_df = pd.DataFrame([[ image.shape[1]/2,image.shape[0]/2]],
        #                                 columns = ['x_pix','y_pix'])

        #         # print(i)


        #         if start_mag+i >= 2:
        #             break

        #         for j in range(redo):

        #             print('\rdmag step: %d / %d :: [%d / %d]' % (1+i/dmag,nsteps,j+1,redo),end = '',flush = True)

        #             fake_source_on_target = input_model(injection_df['x_pix'].values[0],injection_df['y_pix'].values[0],
        #                                                 mag2image(start_mag+i))



        #             if syntax['inject_source_add_noise']:

        #                 nan_idx = np.isnan(fake_source_on_target)
        #                 neg_idx = fake_source_on_target<0

        #                 fake_source_on_target[nan_idx] = 0
        #                 fake_source_on_target[neg_idx] = 0

        #                 fake_source_on_target = make_noise_image(fake_source_on_target.shape,
        #                                                 distribution = 'poisson',
        #                                                 mean = fake_source_on_target,
        #                                                 random_state = np.random.randint(0,1e6))

        #                 fake_source_on_target = apply_poisson_noise(fake_source_on_target)


        #             psf_fit,_ = psf.fit(image + fake_source_on_target,
        #                                 injection_df,
        #                                 r_table,
        #                                 syntax,
        #                                 syntax['fwhm'],
        #                                 return_fwhm = True,
        #                                 no_print = True,
        #                                 hold_pos = False,
        #                                 )

        #             psf_params,_ = psf.do(psf_fit,
        #                                   r_table,
        #                                   syntax,
        #                                   syntax['fwhm'])

        #             recovered_counts.append(psf_params['psf_counts'].values[0])
        #             recovered_counts_e.append(psf_params['psf_counts_err'].values[0])

        #             psf_counts = psf_params['psf_counts'].values/syntax['exp_time']
        #             psf_counts_err = psf_params['psf_counts_err'].values/syntax['exp_time']

        #             mag_recovered =  mag(psf_counts,0)
        #             mag_recovered_error = mag(psf_counts,0) - mag(psf_counts+psf_counts_err,0)


        #             inserted_magnitude.append(start_mag+i)
        #             recovered_magnitude.append(mag_recovered[0])
        #             recovered_magnitude_e.append(mag_recovered_error[0])
        #             recovered_fwhm.append(psf_params['target_fwhm'].values[0])
        #             recovered_fwhm_e.append(psf_params['target_fwhm_err'].values[0])



        #     print (' ... done')

        #     recover_data = np.array([inserted_magnitude,recovered_magnitude,recovered_magnitude_e,recovered_fwhm,recovered_fwhm_e])
        #     recover_df = pd.DataFrame(recover_data.T)
        #     recover_df.columns = ['inject_mag','recover_mag','recover_mag_e','recover_fwhm','recover_fwhm_e']

        #     recover_df.to_csv(syntax['write_dir']+'limiting_mag_explore.csv')

        #     # print(inserted_magnitude)
        #     fig = plt.figure('limiting_mag_search')

        #     ax1 = fig.add_subplot(311)

        #     markers, caps, bars = ax1.errorbar(inserted_magnitude+syntax['zp'],recovered_magnitude+syntax['zp'],
        #                  yerr = recovered_magnitude_e,
        #                  ls = '',
        #                  marker = 'o',
        #                  ecolor = 'black',
        #                  color = 'red',
        #                  label = 'Recovered Magnitude')

        #     ax1.plot(inserted_magnitude+syntax['zp'],inserted_magnitude+syntax['zp'],
        #              ls = '--',
        #              color = 'red',
        #              alpha = 0.5,
        #              label = 'True Magnitude')

        #     [bar.set_alpha(0.25) for bar in bars]
        #     [cap.set_alpha(0.25) for cap in caps]


        #     plt.rc('grid', linestyle=":", color='black',alpha = 0.1)

        #     ax1.grid(True)


        #     y_values   = [str(i*2)+r'$\sigma$' for i in range(10)]
        #     y_axis = [std*i*2 for i in range(10)]




        #     ax2 = fig.add_subplot(312,sharex = ax1)
        #     # np.array(recovered_magnitude)-np.array(inserted_magnitude)
        #     markers, caps, bars = ax2.errorbar(inserted_magnitude+syntax['zp'],recovered_counts,
        #                  yerr = recovered_counts_e,
        #                  ls = '',
        #                  marker = 'o',
        #                  ecolor = 'black',
        #                  color = 'blue',
        #                  label = 'PSF Counts')

        #     # ax2.set_yticks(y_axis, y_values)
        #     ax2.set_yticks(y_axis)
        #     ax2.set_yticklabels(y_values)

        #     [bar.set_alpha(0.25) for bar in bars]
        #     [cap.set_alpha(0.25) for cap in caps]

        #     ax3 = fig.add_subplot(313,sharex = ax1)

        #     markers, caps, bars = ax3.errorbar(inserted_magnitude+syntax['zp'],recovered_fwhm,
        #                  yerr = recovered_fwhm_e,
        #                  ls = '',
        #                  marker = 'o',
        #                  ecolor = 'black',
        #                  color = 'green',
        #                  label = r'Recovered FWHM')

        #     [bar.set_alpha(0.25) for bar in bars]
        #     [cap.set_alpha(0.25) for cap in caps]


        #     ax3.set_ylim(syntax['fwhm'] - 2,syntax['fwhm'] + 2)
        #     # ax2.set_ylim(0,10)


        #     # ax2.axhline(0,color = 'blue',ls = '--',label = 'Full Recovery')
        #     ax3.axhline(syntax['fwhm'],color = 'green',ls = '--',label = 'Image FWHM')


        #     ax1.xaxis.set_major_locator(MultipleLocator(1))
        #     ax1.xaxis.set_minor_locator(MultipleLocator(0.25))

        #     ax1.yaxis.set_major_locator(MultipleLocator(2))
        #     ax1.yaxis.set_minor_locator(MultipleLocator(0.5))


        #     # ax2.yaxis.set_major_locator(MultipleLocator(1))
        #     # ax2.yaxis.set_minor_locator(MultipleLocator(0.25))

        #     ax3.yaxis.set_major_locator(MultipleLocator(1))
        #     ax3.yaxis.set_minor_locator(MultipleLocator(0.25))


        #     ax1.set_ylabel(r'$M_{Recovered}$ + ZP')
        #     # ax2.set_ylabel(r'$M_{Recovered}$ - $M_{Inserted}$')
        #     ax2.set_ylabel('Recovered PSF counts')
        #     ax3.set_ylabel(r'$FWHN_{Gaussian}$ Recovered')
        #     ax3.set_xlabel(r'$M_{inserted}$ + ZP')


        #     for ax in [ax1,ax2,ax3]:
        #         ax.yaxis.set_label_coords(-0.06,0.5)
        #         ax.axvline(mag_level+syntax['zp'],alpha=0.5,
        #                    label = r'%d$\sigma$ detection limit' % syntax['lim_SNR'],
        #                    color = 'black',
        #                    ls = ':')



        #     for ax in [ax1,ax2]:
        #         ax.label_outer()


        #     fig.savefig(syntax['write_dir']+'limiting_mag_explore.pdf',
        #                                     format = 'pdf')
