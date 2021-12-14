#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def compute_phot_error(flux_variance,sky_std,sky_annulus_area,ap_area,gain=1.0):
    '''
    
    Computes the flux errors using the DAOPHOT style computation. This code has been adapted from 
    `here <https://github.com/spacetelescope/wfc3_photometry/blob/master/photometry_tools/photometry_with_errors.py>`_
    
    This function is used in combination with aperture photometry packages. The flux error is given by:
    
    .. math::
       \delta flux = \\frac{flux~variance / gain}{area_{ap} \\times \\sigma_{bkg} ^2 \\times (1 + \\frac{area_{ap}}{area_{sky,annulus}})}
     
    :param flux_variance: flux variance of target
    :type flux_variance: float
    :param sky_std: standard deviation of background
    :type sky_std: float
    :param sky_annulus_area: Area of annulus used to find standard deviation of background
    :type sky_annulus_area: float
    :param ap_area: Area under aperture where flux variance is measured
    :type ap_area: float
    :param gain: Gain of image in :math:`e^{-}` per ADU, defaults to 1.0
    :type gain: floar, optional
    :return: Flux error of flux measurement
    :rtype: float

    '''
    """"""
    
    bg_variance_terms = (ap_area * sky_std ** 2. ) * (1. + ap_area/sky_annulus_area)
                        
    variance = flux_variance / gain + bg_variance_terms
    
    flux_error = variance ** .5
    
    return flux_error


def find_aperture_correction(dataframe,
                             write_dir = None,
                             base = None,
                             ap_corr_plot = False):
    '''
    Package used to find aperture correction for use in aperture photometry.
    This correction accounts for the fact that we use an aperture size of finite
    size when a point source ay have flux extending out to much larger radii. This function performs aperture photometry on
    several bright isolated sources using a normal aperture size and a larger aperture size. An aperture correction can then be
    found using the following formula:
    
    .. math::
    
  
       apcorr = -2.5 \\times  Log_{ 10 }( \\frac{ F_{inf} }{ F_{ap} } )
     
    
    where :math:`F_{inf}` is the flux measured under a small aperture size and :math:`F_{ap}` is the flux found under the normal aperture size used.
    
    :param dataframe: Dataframe containing columns with :math:`\mathit{counts\_inf\_ap}`  and :math:`\mathit{counts\_ap}` representing   :math:`F_{inf}` and  :math:`F_{ap}` respectively.
    :type dataframe: Dataframe
    :param write_dir: If ap_corr_plot is True, this param is the write directory of the histogram plot of the aperture corrections. , defaults to None
    :type write_dir: str, optional
    :param base: If ap_corr_plot is True, base is the filename appended onto the word "aperture_correction_" when naming the histogram plot of the aperture corrections. , defaults to None
    :type base: str , optional
    :param ap_corr_plot: If True, save a plot of the distribution of aperture corrections , defaults to False
    :type ap_corr_plot: bool, optional
    :return: Aperture corrections and error on aperture correction given by the standard deviation. 
    :rtype: Tuple
'''

    

    import os
    import logging
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip
    
    from autophot.packages.functions import calc_mag
    from autophot.packages.functions import set_size
    

    logger = logging.getLogger(__name__)

    
    # Get magnitude different between aperatrure and "infinite" aperture
    aperture_difference = calc_mag(dataframe['counts_inf_ap'] / dataframe['counts_ap'],1,0)
    
    aperture_difference = aperture_difference[~np.isnan(aperture_difference)]
    
    # sigma clip outliers 
    mask = np.array(~sigma_clip(aperture_difference,
                                sigma = 3,
                                cenfunc = np.nanmedian).mask)

    aperture_correction_cleaned = aperture_difference[mask]
    
    aperture_correction = np.nanmedian(aperture_correction_cleaned)
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

    Perform aperture photometry using two aperture sizes on a series of targets.
    
    :param image: 2D image containing sources to be measured using aperture photometry
    :type image: 2D array
    :param dataframe: Dataframe containing :math:`\mathit{x\_pix}` and :math:`\mathit{y\_pix}` columns representing the X, Y pixel locations of a source.
    :type dataframe: Dataframe
    :param fwhm: Full Width Half Maximum (FWHM) of image. This is used to calibrated aperture and annulus size.
    :type fwhm: Float
    :param ap_size: Multiple of FWHM to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param inf_ap_size: Multiple of FWHM to be used as larger, :math:`\mathit{infinite}` aperture size, defaults to 2.5
    :type inf_ap_size: float, optional
    :param r_in_size: Multiple of FWHM to be used as inner radius of background annulus, defaults to 1.9
    :type r_in_size: float, optional
    :param r_out_size: Multiple of FWHM to be used as outer radius of background annulus, defaults to 2.2
    :type r_out_size: float, optional
    :return: Returns a dataframe containing original :math:`\mathit{x\_pix}` and :math:`\mathit{y\_pix}` columns as well as columns  :math:`\mathit{counts\_inf\_ap}`  and :math:`\mathit{counts\_ap}` containing counts under the apertures given by :math:`\mathit{ap\_size}` and :math:`\mathit{inf\_ap\_size}`
    :rtype: Dataframe

    '''
    

    import sys
    import numpy as np
    import os
    import logging

    from autophot.packages.aperture import measure_aperture_photometry
    
    logger = logging.getLogger(__name__)


    try:

        ap_dict = {}
        ap_dict['inf_ap'] = inf_ap_size * fwhm
        ap_dict['ap']     = ap_size * fwhm

        positions  = list(zip(np.array(dataframe['x_pix']),np.array(dataframe['y_pix'])))

        for key,val in ap_dict.items():

            try:

                aperture_counts,aperture_counts_error,_,_,_ = measure_aperture_photometry(positions,
                                                                                          image,
                                                                                          ap_size = val,
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


# =============================================================================
# Measure Aperture photometry
# =============================================================================
def measure_aperture_photometry(positions,
                                image,
                                gain = 1, 
                                bkg_level = 3,
                                ap_size = 1.7,
                                r_in = 1.9,
                                r_out = 2.2):
    '''
        
    Main package to perform aperture photometry. The counts under an aperture are found using:

    .. math::
    
      
       counts = F_{ap} \\times T_{exp} = \\sum_{ap}(counts) - \\langle counts_{sky} \\rangle \\times n
       

    where :math:`F_{ap}` is the flux under an aperture, :math:`T_{exp}` is the exposure time of the observations in seconds. :math:`\sum_{ap}(counts)` defines the counts summed up under an aperture, :math:`\\langle counts_{sky} \\rangle` is the average background level assumed to be under the aperture (and the flux we want to measure) and n is the number of pixels in the aperture (:math:`n=\pi r_{ap} ^2`, where :math:`r_{ap}` is the radius of the aperture.)

    :param positions: List of tuples containing x,y positions. For example:  :math:`positions = [(1,2),(3,4)]`
    :type positions: List of Tuples
    :param image: 2D image containing sources which we want to measure with aperture photometry
    :type image: 2D array
    :param gain: Gain of observation in :math:`e^{-}\ per\ ADU`, defaults to 1
    :type gain: float, optional
    :param bkg_level: Number of standard deviations about the mean background below which was assume is due background fluctuations rather than any source flux , defaults to 3
    :type bkg_level: float, optional
    :param ap_size: Size of aperture to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param inf_ap_size:  Size of aperture to be used as larger, :math:`\mathit{infinite}` aperture size, defaults to 2.5
    :type inf_ap_size: float, optional
    :param r_in_size:  Size of aperture to be used as inner radius of background annulus, defaults to 1.9
    :type r_in_size: float, optional
    :param r_out_size:  Size of aperture to be used as outer radius of background annulus, defaults to 2.2
    :type r_out_size: float, optional
    :return: Returns the aperture sum, the error on the aperture sum, the max_pixel found within the aperture, the median value of the background and the standard deviation of the background.
    :rtype: list
    

    '''
    

    try:

        from astropy.stats import sigma_clipped_stats
        from photutils import aperture_photometry
        from photutils import CircularAperture, CircularAnnulus
        import numpy as np
        from photutils.datasets import make_noise_image
        import os,sys

        if r_in == None or r_out == None:
            print('Warning - ap_phot -  inner and outer annulus not set ')
            print('Setting to r-in = 10 r_out = 20')
            r_in = 10
            r_out = 20

        if not isinstance(positions,list):
            positions = list(positions)
        
        # List of aperture for use on the image
        apertures = CircularAperture(positions, r=ap_size)
        
        # Area of aperture
        area = np.pi * ap_size ** 2
        
        # Create annulus to get background value with inner/outer radii
        annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        
        # create masks
        annulus_masks = annulus_apertures.to_mask(method='center')
        aperture_masks = apertures.to_mask(method='center')
        
        # Warning for development - shouldn't pop up
        if r_out >= image.shape[0] or r_out > image.shape[1]:
            print('Error - Apphot - Annulus size greater than image size')

        # list of bkg median for each aperture/source
        bkg_median = []
        bkg_std = []
        max_pixel = [] 
        
        area_sky_annulus = np.pi*(r_out)**2 - np.pi*(r_in)**2
        
        # data_possion_ready = data
        
        if np.any(image < 0):
            
            error_array = None
            
        else:
        
            possion_noise = make_noise_image(image.shape,
                                            distribution = 'poisson',
                                            mean = image)
      
            error_array = possion_noise
            
        if not isinstance(annulus_masks,list):
            annulus_masks = list(annulus_masks)
            
        if not isinstance(aperture_masks,list):
            aperture_masks = list(aperture_masks)

        # get background for each source
        for annulus_mask,aperture_mask in zip(annulus_masks,aperture_masks):
            
            median_sigclip = np.nan
            std_sigclip = np.nan
            max_pixel_value = np.nan
            
            annulus_data = annulus_mask.multiply(image)
            annulus_data_1d = annulus_data[annulus_mask.data > 0]
            annulus_data_1d_nonan = annulus_data_1d[~np.isnan(annulus_data_1d)]
            
            aperture_data = aperture_mask.multiply(image)
            aperture_data_1d = aperture_data[aperture_mask.data > 0]
            aperture_data_1d_nonan = aperture_data_1d[~np.isnan(aperture_data_1d)]

            mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d_nonan,
                                                                            cenfunc = np.nanmedian,
                                                                            stdfunc = np.nanstd,
                                                                            sigma= 3)
            
            # std_sigclip = np.nanstd(annulus_data_1d_nonan)
            max_pixel_value = np.nanmax(aperture_data_1d_nonan) - median_sigclip
     
            bkg_median.append(median_sigclip)
            bkg_std.append(std_sigclip)
            max_pixel.append(max_pixel_value)

        # Background median and standard deviation of annulus
        bkg_median = np.array(bkg_median)
        bkg_std = np.array(bkg_std)
        
        # Max value within aperture
        max_pixel = np.array(max_pixel) 
        
        # perform aperure photometry on image using list of apertures
        phot = aperture_photometry(image, apertures, error = error_array)
        phot = phot.to_pandas()
       

        phot['annulus_median'] = bkg_median
        phot['annulus_std'] = bkg_std
        phot['aperture_bkg'] = bkg_median * area

        phot['aperture_sum_bkgsub'] = phot['aperture_sum'] - phot['aperture_bkg']
        
        phot.at[phot['aperture_sum_bkgsub']<=0,'aperture_sum_bkgsub'] = 0 
        
        aperture_sum = phot['aperture_sum_bkgsub'].values
        
        if error_array is None:
            aperture_sum_error = compute_phot_error(flux_variance = aperture_sum,
                                                    sky_std = bkg_std,
                                                    sky_annulus_area=area_sky_annulus,
                                                    ap_area=area,
                                                    gain=gain)
            
        else:
            aperture_sum_error = compute_phot_error(flux_variance = phot['aperture_sum_err'].values**2,
                                                    sky_std = bkg_std,
                                                    sky_annulus_area=area_sky_annulus,
                                                    ap_area=area,
                                                    gain=gain)
       


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)



    return aperture_sum, aperture_sum_error, max_pixel, bkg_median, bkg_std






# =============================================================================
# PLOT APERTURE CLOSEUP
# =============================================================================
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
    '''

    Package used for plotting close up of aperture photometry on point source. Package produces a cutout showing a closeup of a given target, the aperture placement and size,and the annulus used for the sky background. Additionally, projections along the X and Y plane are given to aid the User in evaluating the aperture placement


    :param close_up: 2D image containing target. Image does not need to be background subtracted
    :type close_up: 2D Array
    :param target_x_pix_corr: X Pixel coordinate of target within :math:`close\_up` image. 
    :type target_x_pix_corr: float
    :param target_y_pix_corr: y Pixel coordinate of target within :math:`close\_up` image. 
    :type target_y_pix_corr: float
    :param fwhm: Full Width Half Maximum of image. This is used to set the size for the apertures and annulli. 
    :type fwhm: float
    :param ap_size: Size of aperture to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param r_in_size:  Size of aperture to be used as inner radius of background annulus, defaults to 1.9
    :type r_in_size: float, optional
    :param r_out_size:  Size of aperture to be used as outer radius of background annulus, defaults to 2.2
    :type r_out_size: float, optional
    :param write_dir: Directory where to place the image.
    :type write_dir: str
    :param base: Name of file which is appended onto ':math:`\mathit{target\_ap\_}`'.
    :type base: str
    :param background_value: If given, plot the background value assumed for the image, defaults to None
    :type background_value: float, optional
    :return: Produces a pdf plot of the target with an aperture and annuli. The file is saved to ':math:`\mathit{write\_dir}`' with the name ':math:`\mathit{target\_ap\_}`'. + :math:`\mathit{base}`.
    :rtype: PDF plot


    '''
    
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
    
    ax1.imshow(close_up_plt,
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
                               ap_size = 1.7,
                               r_in_size = 1.9,
                               r_out_size = 2.2,
                               GAIN = 1, 
                               RDNOISE = 0,
                               plot_optimum_radius = False):
    '''

    Find the optimum aperture radius for a given image. Although the initial guess of :math:`1.7 \\times FWHM` is a suitable guess for the aperture size of an image. Irregular / symmetric point spread functions (PSF) may require a slightly larger or smaller aperture size. This function uses several well isolated sources and finds their Signal to noise ratio (S/N) using the following equation

    .. math::
        
       S/N_{max} = \\frac{ F_{ap,optimum} }{ F_{ap,optimum} + F_{sky,ap,optimum,n} + (RN ^2 + \\frac{G^2}{4} \\times n_{pix}) + (D \\times n_{pix} \\times t_exp) } ^{0.5}
   

     where :math:`F_{ap,optimum}` is the flux under an aperture of an optimum radius, and likewise :math:`F_{sky,ap,optimum,n}` is the flux due to the sky background under the same aperture. By varying the size of the aperture, we produce a curve of growth model for how the S/N ratio behaves for different radii. This package iterations through a small radii towards a very large radii and notes how the S/N changes for a sample of sources. Where these sources reach a maximum (which :math:`\mathit{should}` be the same for all point sources) is considered the optimum aperture size, where we obtain the option ratio of source flux and background noise.
     
    :param dataframe: Dataframe containing :math:`\mathit{x\_pix}` and :math:`\mathit{y\_pix}` columns representing the X, Y pixel locations of a source. Dataframe source also include :math:`\mathit{include\_fwhm}`. This column source be a boolean list where :math:`\mathit{True}` dictates that a source source be included in the optimum radius investigation and :math:`\mathit{False}` meaning it is excluded.
    :type dataframe: Dataframe
    :param image: 2D image containing sources to be measured using aperture photometry
    :type image: 2D array
    :param exp_time: Exposure time in seconds of image.
    :type exp_time: float
    :param fwhm: Full Width Half Maximum (FWHM) of image. This is used to calibrated aperture and annulus size.
    :type fwhm: Float
    :param write_dir: Directory where to place the image.
    :type write_dir: str
    :param base: Name of file which is appended onto ':math:`\mathit{target\_ap\_}`'.
    :type base: str
    :param ap_size: Multiple of FWHM to be used as standard aperture size, defaults to 1.7
    :type ap_size: float, optional
    :param inf_ap_size: Multiple of FWHM to be used as larger, :math:`\mathit{infinite}` aperture size, defaults to 2.5
    :type inf_ap_size: float, optional
    :param r_in_size: Multiple of FWHM to be used as inner radius of background annulus, defaults to 1.9
    :type r_in_size: float, optional
    :param r_out_size: Multiple of FWHM to be used as outer radius of background annulus, defaults to 2.2
    :type r_out_size: float, optional
    :param GAIN: Gain of image in :math:`e^{-}$ per ADU`, defaults to 1.0
    :type GAIN: float, optional
    :param RDNOISE: Read Noise of image  of image in :math:`e^{-}$ per pixel`, defaults to 0
    :type RDNOISE: float, optional
    :param plot_optimum_radius: If true, saves of plot of the curve of growths for a sample of sources  saved to :math:`\mathit{write\_dir}` with the name ":math:`\mathit{optimum\_aperture\_}`". + :math:`\mathit{base}`, defaults to False
    :type plot_optimum_radius: boolean, optional
    :return: Gives the optim radius in units of FWHM.
    :rtype: Float
    '''

    
    import numpy as np
    import os

    
    from autophot.packages.functions import SNR,SNR_err
    import matplotlib.pyplot as plt
    from autophot.packages.functions import set_size
    
    import logging
    logger = logging.getLogger(__name__)

    
    step_size = 0.1
    
    search_size = np.arange(0.1,5,step_size)
    

    idx = dataframe['include_fwhm']

    dataframe = dataframe[idx].head(25)
    
    positions  = list(zip(np.array(dataframe.x_pix),np.array(dataframe.y_pix)))
    
    output = []
        
    for s in search_size:

        s_in = r_in_size
        s_out = r_out_size
    
        aperture_counts,aperture_counts_error ,_, aperture_bkg_counts,_ = measure_aperture_photometry(positions,
                                                                                        image,
                                                                                        ap_size =  s     * fwhm,
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
        
    sum_distribution = np.nanmedian(sum_distribution,axis=0)
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




