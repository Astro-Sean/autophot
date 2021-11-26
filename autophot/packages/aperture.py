#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def find_aperture_correction(dataframe,
                             image,
                             write_dir = None,
                             base = None,
                             ap_corr_plot = False):
    '''
    
    :param dataframe: DESCRIPTION
    :type dataframe: TYPE
    :param image: DESCRIPTION
    :type image: TYPE
    :param write_dir: DESCRIPTION, defaults to None
    :type write_dir: TYPE, optional
    :param base: DESCRIPTION, defaults to None
    :type base: TYPE, optional
    :param ap_corr_plot: DESCRIPTION, defaults to False
    :type ap_corr_plot: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    

    import os

    import logging

    import numpy as np
    from astropy.stats import sigma_clip
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    from autophot.packages.functions import calc_mag
    from autophot.packages.functions import set_size
    
    import logging
    logger = logging.getLogger(__name__)

    
    # Get magnitude different between aperatrure and "infinite" aperture
    aperture_difference = calc_mag(dataframe['counts_inf_ap'] / dataframe['counts_ap'],1,0)
    
    aperture_difference = aperture_difference[~np.isnan(aperture_difference)]
    
    # sigma clip outliers 
    mask = np.array(~sigma_clip(aperture_difference,
                                sigma = 3,
                                cenfunc = np.nanmedian).mask)

    aperture_correction_cleaned = aperture_difference[mask]
    
    aperture_correction = np.nanmean(aperture_correction_cleaned)
    aperture_correction_err = np.nanstd(aperture_correction_cleaned)
    
    logger.info('Aperture correction: %.3f +/- %.3f [ mag ]' % (aperture_correction,aperture_correction_err))
    
    if ap_corr_plot:
        
        # histogram of aperture corrections
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
        
        # Fit a normal distribution to the data:
        mu, std = norm.fit(aperture_correction_cleaned)
        
        plt.ioff()
        fig = plt.figure(figsize = set_size(250,1))
        
        ax1 = fig.add_subplot(111)

        # Plot the histogram.
        ax1.hist(aperture_correction_cleaned, 
                 bins='auto', 
                 density=True,
                 color = 'blue',
                 label = 'Aperture Correction')
        
                # Plot the Pdataframe.
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        ax1.plot(x, p, label = 'PDF',color = 'r')

        ax1.set_xlabel(r'Correction [ mag ]')
        ax1.set_ylabel('Probability Density')
        
        ax1.axvline(aperture_correction,color = 'black' )

        ax1.legend(loc = 'best',frameon = False) 

        fig.savefig(os.path.join(write_dir,'aperture_correction_'+base+'.pdf'),
                    format = 'pdf',
                    bbox_inches='tight')
                    
    
        plt.close(fig)

    
    return aperture_correction,aperture_correction_err



def do_aperture_photometry(image,
                           dataframe,
                           fwhm,
                           ap_size = 1.7,
                           inf_ap_size = 2.5,
                           r_in_size = 1.9,
                           r_out_size = 2.2):
    '''
    
    :param image: DESCRIPTION
    :type image: TYPE
    :param dataframe: DESCRIPTION
    :type dataframe: TYPE
    :param fwhm: DESCRIPTION
    :type fwhm: TYPE
    :param ap_size: DESCRIPTION, defaults to 1.7
    :type ap_size: TYPE, optional
    :param inf_ap_size: DESCRIPTION, defaults to 2.5
    :type inf_ap_size: TYPE, optional
    :param r_in_size: DESCRIPTION, defaults to 1.9
    :type r_in_size: TYPE, optional
    :param r_out_size: DESCRIPTION, defaults to 2.2
    :type r_out_size: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    

    import sys
    import numpy as np
    import os
    import logging

    from autophot.packages.aperture import measure_aperture_photometry
    from autophot.packages.functions import calc_mag
    
    logger = logging.getLogger(__name__)


    try:

        ap_dict = {}
        ap_dict['inf_ap'] = inf_ap_size * fwhm
        ap_dict['ap']     = ap_size * fwhm

        positions  = list(zip(np.array(dataframe['x_pix']),np.array(dataframe['y_pix'])))

        for key,val in ap_dict.items():

            try:

                aperture_counts,_,_,_ = measure_aperture_photometry(positions,image,
                                                                           radius = val,
                                                                           r_in = val + r_in_size   * fwhm,
                                                                           r_out = val + r_out_size * fwhm)
                                                                           

                dataframe['counts_'+str(key)] = aperture_counts

            except Exception as e:

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.info(exc_type, fname, exc_tb.tb_lineno,e)

                break
 
    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.info(exc_type, fname, exc_tb.tb_lineno,e)



    return dataframe

def measure_aperture_photometry(positions, data, radius ,r_in = None,r_out= None):

    """This a robust aperture photometry packages for use in autophot


    :param positions: list of tuples containing (x,y) positions of object
    :type positions: tuple

    :param r_in: inner radius of background annulus
    :type r_in: float

    :param r_out: outer radius of background annulus
    :type r_out: float

    :return: Returns lost of aperture measurements
    :rtype: list
    
    """

    try:

        from astropy.stats import sigma_clipped_stats
        from photutils import aperture_photometry
        from photutils import CircularAperture, CircularAnnulus
        import numpy as np
        import os,sys

        if r_in == None or r_out == None:
            print('Warning - ap_phot -  inner and outer annulus not set ')
            print('Setting to r-in = 10 r_out = 20')
            r_in = 10
            r_out = 20

        if not isinstance(positions,list):
            positions = list(positions)
        
        # List of aperture for use on the image
        apertures = CircularAperture(positions, r=radius)
        
        # Area of aperture
        area = np.pi * radius ** 2
        
        # Create annulus to get background value with inner/outer radii
        annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        
        # create masks
        annulus_masks = annulus_apertures.to_mask(method='center')
        aperture_masks = apertures.to_mask(method='center')
        
        # Warning for development - shouldn't pop up
        if r_out >= data.shape[0] or r_out > data.shape[1]:
            print('Error - Apphot - Annulus size greater than image size')

        # list of bkg median for each aperture/source
        bkg_median = []
        bkg_std = []
        max_pixel = [] 

        if not isinstance(annulus_masks,list):
            annulus_masks = list(annulus_masks)
            
        if not isinstance(aperture_masks,list):
            aperture_masks = list(aperture_masks)

        # get background for each source
        for annulus_mask,aperture_mask in zip(annulus_masks,aperture_masks):
            median_sigclip = np.nan
            std_sigclip = np.nan
            max_pixel_value = np.nan
            try:
            
                annulus_data = annulus_mask.multiply(data)
                
                annulus_data_1d = annulus_data[annulus_mask.data > 0]
                annulus_data_1d_nonan = annulus_data_1d[~np.isnan(annulus_data_1d)]
                
                aperture_data = aperture_mask.multiply(data)
                
                aperture_data_1d = aperture_data[aperture_mask.data > 0]
                aperture_data_1d_nonan = aperture_data_1d[~np.isnan(aperture_data_1d)]
    
                mean_sigclip, median_sigclip,std_sigclip = sigma_clipped_stats(annulus_data_1d_nonan,
                                                           cenfunc = np.nanmean,
                                                           stdfunc = np.nanstd,
                                                           sigma= 3)
    
                std_sigclip = np.nanstd(annulus_data_1d_nonan)
                max_pixel_value = np.nanmax(aperture_data_1d_nonan)
                
                
            except:
                pass
            
            bkg_median.append(median_sigclip)
            bkg_std.append(std_sigclip)
            max_pixel.append(max_pixel_value)

        bkg_median = np.array(bkg_median)
        bkg_std = np.array(bkg_std)
        
        # Max value within aperture minus background median
        max_pixel = np.array(max_pixel) - bkg_median
        
        # perform aperure photometry on image using list of apertures
        phot = aperture_photometry(data, apertures)
        phot = phot.to_pandas()

        phot['annulus_median'] = bkg_median
        phot['annulus_std'] = bkg_std
        phot['aperture_bkg'] = bkg_median * area

        # print(phot['aperture_sum'], phot['aperture_bkg'])
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aperture_bkg']

        aperture_sum = phot['aper_sum_bkgsub'].values
        
        bkg_median = phot['annulus_median'].values

        # aperture_sum[aperture_sum<=0] = 0

    except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)



    return aperture_sum, max_pixel, bkg_median, bkg_std





def plot_aperture(close_up,
                  target_x_pix_corr,
                  target_y_pix_corr,
                  fwhm,
                  ap_size,
                  r_in_size,
                  r_out_size,
                  write_dir,
                  base,
                  background_value = None):
    
    # Aperture photometry plot
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import  GridSpec
    from autophot.packages.functions import order_shift,set_size
    from matplotlib.pyplot import Circle
    
    import os
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
    
    
    scale = order_shift(abs(close_up))
    
    close_up_plt = close_up/scale
    
    plt.ioff()
    
    fig_target = plt.figure(figsize = set_size(250,aspect = 1.5))
    
    ncols = 2
    nrows = 2
    
    heights = [1,0.3]
    widths =  [1,0.3]
    
    grid = GridSpec(nrows, ncols,
                    wspace=0.05,
                    hspace=0.05,
                    height_ratios=heights,
                    width_ratios = widths
                    )
    
    ax1   = fig_target.add_subplot(grid[0:1, 0:1])
    ax1_B = fig_target.add_subplot(grid[1, 0:1])
    ax1_R = fig_target.add_subplot(grid[0:1, 1])
    
    im = ax1.imshow(close_up_plt,
                   interpolation = 'nearest',
                   origin = 'lower',
                    aspect = 'auto',
                   )

    circ1 = Circle((target_x_pix_corr,target_y_pix_corr),
                   ap_size * fwhm,
                   label = 'Aperture',
                   fill=False,
                   color = 'red',
                   alpha = 0.5)
    
    r0_x = target_x_pix_corr
    r0_y = target_y_pix_corr
    
    inner =  r_in_size * fwhm
    outer =  r_out_size * fwhm
    
    n, radii = 50, [inner , outer]
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1,:] = xs[1,::-1]
    ys[1,:] = ys[1,::-1]
    
    ax1.fill(r0_x+np.ravel(xs),
             r0_y+np.ravel(ys),
             color = 'red',
             edgecolor = 'none',
             alpha = 0.3)
    
    ax1.add_patch(circ1)

    ax1_B.axvline(r0_x - (ap_size * fwhm),ls = '-',color = 'red')
    ax1_B.axvline(r0_x + (ap_size * fwhm),ls = '-',color = 'red')
    
    ax1_R.axhline(r0_y - (ap_size * fwhm),ls = '-',color = 'red')
    ax1_R.axhline(r0_y + (ap_size * fwhm),ls = '-',color = 'red')
    
    ax1_B.axvspan(r0_x - (r_in_size * fwhm),
                  r0_x - (r_out_size * fwhm),
                  color = 'red',
                  alpha = 0.3,
                  label = 'Background Annulus'
                  )
    
    ax1_B.axvspan(r0_x + (r_in_size * fwhm),
                  r0_x + (r_out_size * fwhm),
                  color = 'red',
                  alpha = 0.3)
    
    ax1_R.axhspan(r0_y - (r_in_size * fwhm),
                  r0_y - (r_out_size * fwhm),
                  color = 'red',
                  alpha = 0.3
                  )
    
    ax1_R.axhspan(r0_y + (r_in_size * fwhm),
                  r0_y + (r_out_size * fwhm),
                  color = 'red',
                  alpha = 0.3
                  )
    
    h, w = close_up_plt.shape
    
    x  = np.arange(0,h)
    y  = np.arange(0,w)
    
    X, Y = np.meshgrid(x, y)
    
    ax1_R.step(close_up_plt[:,w//2],Y[:,w//2],color = 'blue',where='mid')
    ax1_B.step(X[h//2,:],close_up_plt[h//2,:],color = 'blue',where='mid')
    
    if not (background_value is None):
        ax1_B.axhline(background_value/scale,color = 'green',label = 'Background')
        ax1_R.axvline(background_value/scale,color = 'green')
    
    # ax1_R.xaxis.tick_top()
    ax1_R.yaxis.set_label_position("right")
    ax1_R.yaxis.tick_right()
    
    
    # ax1.xaxis.tick_top()
    ax1.axis('off')
    
    ax1_B.set_ylabel('Counts [$10^{%d}$]' % np.log10(scale))
    
    ax1_R.set_xlabel('Counts [$10^{%d}$]' % np.log10(scale))
    
    ax1_R.set_ylabel('Y [ Pixel ]')
    ax1_B.set_xlabel('X [ Pixel ]')
    
    ax1_R.set_ylim(ax1.get_ylim()[0]-0.5,ax1.get_ylim()[1]+0.5)
    ax1_B.set_xlim(ax1.get_xlim()[0]-0.5,ax1.get_xlim()[1]+0.5)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig_target.axes]
    handles,labels = [sum(i, []) for i in zip(*lines_labels)]
    
    by_label = dict(zip(labels, handles))
    
    ax1.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(0.5, 1),
               loc='lower center',
               ncol = 3,
               frameon=False)
    
    save_loc = os.path.join(write_dir,'target_ap_'+base+'.pdf')
    
    fig_target.savefig(save_loc ,
                       bbox_inches='tight')
    
    plt.close(fig_target)

   


                 
def find_optimum_aperture_size(dataframe,
                               image,
                               exp_time,
                               fwhm,
                               write_dir,
                               base,
                               ap_size = 1.7,inf_ap_size = 2.5,
                               r_in_size = 1.9,r_out_size = 2.2,
                               GAIN = 1, RDNOISE = 0,
                               plot_optimum_radius = False):
    '''
    
    :param dataframe: DESCRIPTION
    :type dataframe: TYPE
    :param image: DESCRIPTION
    :type image: TYPE
    :param exp_time: DESCRIPTION
    :type exp_time: TYPE
    :param fwhm: DESCRIPTION
    :type fwhm: TYPE
    :param write_dir: DESCRIPTION
    :type write_dir: TYPE
    :param base: DESCRIPTION
    :type base: TYPE
    :param ap_size: DESCRIPTION, defaults to 1.7
    :type ap_size: TYPE, optional
    :param inf_ap_size: DESCRIPTION, defaults to 2.5
    :type inf_ap_size: TYPE, optional
    :param r_in_size: DESCRIPTION, defaults to 1.9
    :type r_in_size: TYPE, optional
    :param r_out_size: DESCRIPTION, defaults to 2.2
    :type r_out_size: TYPE, optional
    :param GAIN: DESCRIPTION, defaults to 1
    :type GAIN: TYPE, optional
    :param RDNOISE: DESCRIPTION, defaults to 0
    :type RDNOISE: TYPE, optional
    :param plot_optimum_radius: DESCRIPTION, defaults to False
    :type plot_optimum_radius: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    
    import numpy as np
    import pandas as pd
    import os
    
    from random import uniform
    
    from autophot.packages.functions import SNR,SNR_err
    import matplotlib.pyplot as plt
    from autophot.packages.functions import set_size
    
    import logging
    logger = logging.getLogger(__name__)

    
    
    optimum_size = 0 
    
    step_size = 0.1
    
    search_size = np.arange(0.1,5,step_size)
    
    default_ap_size = ap_size
    default_infinite_ap_size = inf_ap_size
    
    idx = dataframe['include_fwhm']

    dataframe = dataframe[idx].head(25)
    
    positions  = list(zip(np.array(dataframe.x_pix),np.array(dataframe.y_pix)))
    
    output = []
        
    for s in search_size:

        s_in = r_in_size
        s_out = r_out_size
        
        aperture_counts ,_, aperture_bkg_counts,_ = measure_aperture_photometry(positions,
                                                                                                                         image,
                                                                                                                         radius =  s     * fwhm,
                                                                                                                         r_in =    s_in  * fwhm,
                                                                                                                         r_out =   s_out * fwhm)
        # Background flux from annulus
        aperture_bkg_flux = aperture_bkg_counts/exp_time
    
        # Source flux
        aperture_flux = aperture_counts/exp_time

    
        # SNR from ccd equation
        SNR_val = SNR(flux_star = aperture_flux ,
                        flux_sky = aperture_bkg_flux ,
                        exp_t = exp_time,
                        radius = s * fwhm,
                        G  = GAIN,
                        RN = RDNOISE,
                        DC = 0 )
        
        SNR_val_err = SNR_err(SNR_val)
        
        output.append([[s]*len(positions),list(SNR_val),list(SNR_val_err)])
    
    # # find radius where each source has maximum SNR
    optimum_radii = []
    for i in range(len(SNR_val)):
        # a = [j[1][i] for j in output]
        # b = a[::-1]
        # last_max_idx = len(b) - np.argmax(b) - 1
        last_max_idx =  np.argmax([j[1][i] for j in output])
        
        optimum_radii.append(search_size[last_max_idx])
        
        
    sum_distribution = []
    for i in range(len(SNR_val)):
        sum_distribution.append(np.array([j[1][i] for j in output]))
        
    sum_distribution = np.nanmean(sum_distribution,axis=0)
    sum_distribution_max_idx = np.argmax(sum_distribution)
    optimum_radius = search_size[sum_distribution_max_idx]  * 1.5
    
    # optimum_radius = np.nanmedian(optimum_radii)
    
    if optimum_radius>3 and not plot_optimum_radius:
        logger.info('\nOptimum radius seems high [%.1f x FWHM] - setting to %.1f x FWHM' % (optimum_radius,ap_size))
        optimum_radius = ap_size
        return optimum_radius
            
        
    # =============================================================================
    # Return plot of SNR distribution
    # =============================================================================
    if plot_optimum_radius:
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
        
        fig = plt.figure(figsize = set_size(250,1))
        
        ax1 = fig.add_subplot(111)
        
        for i in range(len(SNR_val)):
            
            max_SNR = np.nanmax([j[1][i] for j in output])
            
            # sum_distribution.append(np.array([j[1][i] for j in output]))
            
            
            ax1.plot([j[0][i] for j in output],
                     [j[1][i] for j in output] / max_SNR,
                     # yerr = [j[2][i] for j in output] / max_SNR,
                     # marker = 'o',
                      ls = '-',
                     # capsize = 1,
                     # ecolor = 'grey',
                     lw = 0.5,
                     alpha  = 0.5,
                     label = 'COG',
                     color = 'grey',
                     zorder = 0)
            

        ax1.plot(search_size,
                 sum_distribution / np.nanmax(sum_distribution),
                 # marker = 's',
                 lw = 1,
                 color = 'blue',
                 label = 'Mean COG',
                 zorder = 1)

        
        if optimum_radius>=3:
            logger.info('\nOptimum radius seems high [%.1f x FWHM] - setting to %.1f x FWHM' % (optimum_radius,ap_size))
            optimum_radius = ap_size
            ax1.set_title('Optimum radius not set')
            
        else:
            logger.info('Optimum Aperture: %.1f x FWHM [ pixels ]' % optimum_radius)
            
            ax1.arrow(optimum_radius, 0.15, 0, -0.1,
                  head_width=0.025, head_length=0.025, 
                  lw = 0.5,
                  fc='blue',
                  ec='none',
                  # label = 'Optimum Radius'
                  
                  )
        
        ax1.scatter( [] ,[], c='blue',marker=r'$\leftarrow$',s=25, label='Optimum Radius x 1.5' )
        # 
        ax1.scatter( [] ,[], c='red',marker=r'$\leftarrow$',s=25,  )
        
            
        ax1.set_xlabel('Aperture Size [ 1/FWHM ]')
        
        ax1.set_ylabel('SNR (normalised) ')
        
        ax1.set_ylim(-0.05,1.05)
        ax1.set_xlim(0,search_size.max()+0.1)
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(),
                   frameon = False,
                   loc = 'lower right')
    
            
        fig.savefig(os.path.join(write_dir,'optimum_aperture_'+base+'.pdf'),
                        format = 'pdf',
                        bbox_inches='tight'
                        )
        
        plt.close(fig)
    
    
    
    
    return round(optimum_radius,1)




