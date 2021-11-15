def ranks(sample):

    """Return the ranks of each element in an integer sample.

    :param sample: sample of intergers
    :type sample: array-like

    :return: Return the ranks of each element in an integer sample
    :rtype: sorted list
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample_with_minimum_distance(n=[0,40], k=4, d=10):

    """Sample of k elements from range(n), with a minimum distance d.

    :param n: range of  values
    :type n: tuple

    :param k: number of points
    :type k: int

    :param d: distance between each points
    :type d: float

    :return: list of tuples containing (x,y) location of points seperated by distance(d)
    :rtype: list of tuples
    """
    import random

    sample_x = random.sample(range(int(n[0]),int(n[1])-(k-1)*(d-1)), k)

    sample_y = random.sample(range(int(n[0]),int(n[1])-(k-1)*(d-1)), k)

    return [(x + (d-1)*rx,y + (d-1)*ry) for x,y, rx,ry in zip(sample_x,sample_y,ranks(sample_x),ranks(sample_y))]

def PointsInCircum(r,shape,n=8):
    
    '''
    Generate series of x,y coordinates a distance r from the center of the image
    
    :param r: Distance from center of image 
    :type r: Float

    :param shape: Image shape
    :type shape: Tuple
    
    :param n: Number of points, defaults to 8
    :type n: Integer, optional
    
    :return: List of x,y coorindates placed around the center of the image  at angles 2*pi/n * i for each ith sources in the n sample size
    :rtype: List of tuples

    '''
    import numpy as np
    
    return [(np.cos(2*np.pi/n*x)*r + shape[1]/2 ,np.sin(2*np.pi/n*x)*r + shape[0]/2) for x in range(0,n)]
    

def flatten_dict(d):
    '''
    
    Flatten nested dictionary
    
    :param d: DESCRIPTION
    :type d: TYPE
    
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    def expand(key, value):
        if isinstance(value, dict):
            return [ (key + '.' + k, v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (key, value) ]

    items = [ item for k, v in d.items() for item in expand(k, v) ]

    return dict(items)


def limiting_magnitude_prob(autophot_input,image,model = None,r_table = None,print_progress = True):

    """Find the probable limiting magnitude arond a given tagrte location.
    Methodlology discussed in  'F. Masci 2011 <https://www.semanticscholar.org/paper/Computing-flux-upper-limits-for-non-detections-Masci/6c14bb440637fb6c6c3bad6ce42c2bca7383c735>'__.

    :param autophot_input: AutoPhOT control dictionary
    :type autophot_input: dict
    :Required autophot_input keywords for this package:
        
        - c_counts/r_counts (float): Check to make sure PSF successfully created. These param gives the number of counts under a normalised PSF model as well as within the residual table
        - exp_time (float): Exposure time in seconds
        - fwhm (float): Full width half maximum of image in pixels
        - use_moffat (boolean): Boolean on whether to use moffat (True) or Gaussian (False) in PSF model, default to False.
        - image_params (dict): this dictionary describes the analytical model usinged in the PSF model. If a gaussian function is used this dictionary should contain the key "sigma" and the corresponding value. If a moffat function is function this dictionary should contained 'alpha' and 'beta' values'. These values are found and updated the autophot_input in find.fwhm. 
        - zp (float): Zeropoint of the image
        - gain (float): Gain of image in e/ADU
        - ap_size (float): Aperture size in pixels
        - lim_SNR (float): Signal to Noise detection criteria, default to 3 (sigma)
        - bkg_level (float): Sigma level of background noise, default to 3 (sigma)
        
        - probable_detection_limit (boolean): Use beta detection criteria rather than sigma limit
        - inject_source_random (boolean): Inject sources around the target location with the probable limiting magnitiude
        - inject_source_cutoff_sources (int): Number of random sources to add around the target
        - inject_source_on_target (boolean): Inject a source on target
        - inject_source_add_noise (boolean): Add some Possion noise to the artifically inject sources
        
        - write_dir (str): Working directory where plots will be saved.
        
        
    :param image: 2D image array of target location, typically given as an expanding cutout of the target lcoation
    :type image: 2D array

    :param model: PSF model ,if available, if not a gaussian is used. Default = None
    :type model: Model function

    :param r_table: residual lookup table for build the PSF Default = None
    :type r_table: 2D array

    returns:
    (tuple): tuple containing:

        mag_level: derived value for limting magnitude
        autophot_input: updated AutoPhOT control dictionary
        
    """
    try:
        import matplotlib.pyplot as plt
        import os

        from photutils import CircularAperture
        import numpy as np

        from matplotlib.gridspec import  GridSpec
        import random

        from scipy.optimize import curve_fit
        import warnings

        from photutils.datasets import make_noise_image
        from photutils import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from autophot.packages.background import remove_background
        from autophot.packages.functions import set_size

        from astropy.visualization import  ZScaleInterval
        
        from autophot.packages.functions import gauss_1d

        import logging
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

        logger = logging.getLogger(__name__)

        # level for detection - Rule of thumb ~ 5 is a good detection level
        level = autophot_input['lim_SNR']

        logger.info('Limiting threshold: %d sigma' % level)

        image_no_surface,surface,surface_media,noise = remove_background(image,
                                                                         remove_bkg_local = autophot_input['remove_bkg_local'], 
                                                                            remove_bkg_surface = autophot_input['remove_bkg_surface'],
                                                                            remove_bkg_poly   = autophot_input['remove_bkg_poly'],
                                                                            remove_bkg_poly_degree = autophot_input['remove_bkg_poly_degree'],
                                                                            bkg_level = autophot_input['bkg_level']
                                                                            )


              
        image_mean, image_median, image_std = sigma_clipped_stats(image,
                                        sigma = autophot_input['bkg_level'],
                                        maxiters = 10)
        
        if not autophot_input['subtraction_ready']:
            

            daofind = DAOStarFinder(fwhm    = autophot_input['fwhm'],
                                    threshold = autophot_input['bkg_level']*image_std,
                                    sharplo   =  0.2,sharphi = 1.0,
                                    roundlo   = -1.0,roundhi = 1.0)
    
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ignore no sources warning
                sources = daofind(image - image_median)
    
            if sources != None:

                positions = list(zip(np.array(sources['xcentroid']),np.array(sources['ycentroid'])))
    
                # Add center of image
                positions.append((image.shape[0]/2,image.shape[1]/2))
    
            else:
    
                positions = [(image.shape[0]/2,image.shape[1]/2)]
        else:
                autophot_input['injected_sources_use_beta'] = True
                positions = [(image.shape[0]/2,image.shape[1]/2)]

        # "size" of source, set to the aperture size
        source_size =  autophot_input['ap_size']*autophot_input['fwhm']
        # source_size =  1.3*autophot_input['fwhm']
        
        pixel_number = int(np.pi*source_size**2)

        # Mask out target region
        mask_ap  = CircularAperture(positions,r = source_size)
        mask = mask_ap.to_mask(method='center')
        mask_sumed = [i.to_image(image.shape) for i in mask]
        

        if len(mask_sumed) !=1:
            mask_sumed = sum(mask_sumed)
        else:
            mask_sumed = mask_sumed[0]

        mask_sumed[mask_sumed>0] = 1
        
        if print_progress:
            logging.info('Number of pixels in source: %d [ pixels ]' % pixel_number)

        # Mask out center region
        mask_image  = (image_no_surface) * (1-mask_sumed)
        
        #Used for plotting, get vmin\vmin from aperture are to highlight source
        vmin,vmax = (ZScaleInterval(nsamples = 500)).get_limits(mask_image)

    
        # get pixels that are excluded and included
        excluded_points = np.where(mask_image == 0)

        exclud_x = excluded_points[1]
        exclud_y = excluded_points[0]

        exclud_zip = list(zip(exclud_x,exclud_y))

        included_points = np.where(mask_image != 0)

        includ_x = list(included_points[1])
        includ_y = list(included_points[0])

        includ_zip = list(zip(includ_x,includ_y))

        number_of_points = 150

        fake_points = {}

        # Failsafe - if there isn't enough pixels just use everything
        if len(includ_zip) < pixel_number:

            includ_zip=includ_zip+exclud_zip

        # get random sample of pixels and sum them up
        for i in range(number_of_points):

            fake_points[i] = []

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

        # Fit histogram and get mean and std of distribution
        hist, bins = np.histogram(list(fake_mags.values()),
                                  bins = len(list(fake_mags.values())),
                                  density = True)

        center = (bins[:-1] + bins[1:]) / 2

        sigma_guess = np.nanstd(list(fake_mags.values()))
        mean_guess = np.nanmean(list(fake_mags.values()))
        A_guess = np.nanmax(hist)

        popt,pcov = curve_fit(gauss_1d,center,hist,
                              p0=[A_guess,mean_guess,sigma_guess],
                              absolute_sigma = True )

        mean = popt[1]
        std  = abs(popt[2])
        count_ul = abs(level*std)
        
        if print_progress:
            logging.info('Mean: %s - std: %s' % (round(mean,3),round(std,3)))
            logging.info('Detection at %s std: %.3f' % (level,count_ul))
        
        

        # =============================================================================
        # Plot histogram of background values
        # =============================================================================
        plt.ioff()
        
        limiting_mag_figure = plt.figure(figsize = set_size(250,aspect = 1.5))

        ncols = 2
        nrows = 2

        heights = [0.75,1]

        gs = GridSpec(nrows, ncols ,
                      wspace=0.3,
                      hspace=0.5,
                      height_ratios=heights,
                       # width_ratios = widths
                       )
        
        ax0 = limiting_mag_figure.add_subplot(gs[0, :])
        ax1 = limiting_mag_figure.add_subplot(gs[1, 0])
        ax2 = limiting_mag_figure.add_subplot(gs[1, 1])
        
        line_kwargs = dict(alpha=0.5,color='black',ls = '--')

        
        ax1.scatter(exclud_x,exclud_y,
                    color ='red',
                    marker = 'x',
                    alpha = 0.1,
                    label = 'Encluded areas',
                    zorder = 2)

        # the histogram of the data
        n, bins, patches = ax0.hist(list(fake_mags.values()),
                                    density=True,
                                    bins = 'auto',
                                    facecolor='blue',
                                    alpha=1,
                                    label = 'Pseudo-Counts\nDistribution')

        ax0.axvline(mean,**line_kwargs)
        ax0.axvline(mean + 1*std,**line_kwargs)
        ax0.text(mean + 1*std,np.max(n),r'$1\sigma_{bkg}$',rotation = -90,va = 'top')
        ax0.axvline(mean + 2*std,**line_kwargs)
        ax0.text(mean + 2*std,np.max(n),r'$2\sigma_{bkg}$',rotation = -90,va = 'top')

        ax0.axvline(mean + level*std,**line_kwargs)
        ax0.text(mean + level*std,np.max(n),r'$'+str(level)+r'\sigma_{bkg}$',rotation = -90,va = 'top')

        x_fit = np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 250)

        ax0.plot(x_fit, gauss_1d(x_fit,*popt),
                 label = 'Gaussian Fit',
                 color = 'red')

        # ax0.ticklabel_format(axis='y', style='sci',scilimits = (-2,0))
        # ax0.yaxis.major.formatter._useMathText = True

        ax0.set_xlabel('Pseudo-Counts')
        ax0.set_ylabel('Probability Distribution')

        im2 = ax1.imshow(image-surface,origin='lower',
                         aspect = 'auto',
                         interpolation = 'nearest')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = limiting_mag_figure.colorbar(im2, cax=cax)
        cb.formatter.set_powerlimits((0, 0))

        cb.ax.set_ylabel('Counts', rotation=270,labelpad = 5)

        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

        ax1.set_title('Image - Surface')
        
        # =============================================================================
        # Convert counts to magnitudes
        # =============================================================================

        flux  = count_ul / autophot_input['exp_time']

        mag_level = -2.5*np.log10(flux)

        # =============================================================================
        # We now have an upper and lower estimate of the the limiting magnitude
        # =============================================================================

        fake_sources = np.zeros(image.shape)
        
        try:
            
            if autophot_input['c_counts']:
                pass

            model_label = 'PSF'

            def mag2image(m):
                '''
                Convert magnitude to height of PSF
                '''
                Amplitude  = (autophot_input['exp_time']/(autophot_input['c_counts']+autophot_input['r_counts']))*(10**(m/-2.5))

                return Amplitude
            
            # PSF model that matches close-up shape around target
            def input_model(x,y,flux):
                return model(x, y,0,flux,r_table, autophot_input, pad_shape = image.shape)

        except:

            '''
            if PSF model isn't available - use Gaussian instead

            '''
            logging.info('PSF model not available\n- Using Gaussian for probable limiting magnitude')
            
            model_label = 'Gaussian'

            sigma = autophot_input['fwhm'] / 2*np.sqrt(2*np.log(2))
            
   

            def mag2image(m):
                '''
                Convert magnitude to height of Gaussian
                '''

                #  Volumne/counts under 2d gaussian for a magnitude m
                volume =  (10**(m/-2.5)) * autophot_input['exp_time']

                # https://en.wikipedia.org/wiki/Gaussian_function
                Amplitude =  volume/(2*np.pi*sigma**2)

                return Amplitude

            def input_model(x,y,A):

                # x = np.arange(0,image.shape[0])
                xx,yy= np.meshgrid(np.arange(0,image.shape[1]),np.arange(0,image.shape[0]))

                from autophot.packages.functions import gauss_2d,moffat_2d

                if autophot_input['use_moffat']:
                    model = moffat_2d((xx,yy),x,y,0,A,autophot_input['image_params'])

                else:
                    model = gauss_2d((xx,yy),x,y,0,A,autophot_input['image_params'])
                    
                return model.reshape(image.shape)
            
        # =============================================================================
        #  What magnitude do you want this target to be?
        # =============================================================================

        inject_source_mag = mag2image(mag_level)

        # Number of sources
        source_no = autophot_input['inject_source_cutoff_sources']
        
        random_sources = PointsInCircum(autophot_input['inject_source_location']*autophot_input['fwhm'],
                                        image.shape,n=source_no)
        
        xran = [abs(i[0]) for i in random_sources]
        yran = [abs(i[1]) for i in random_sources]

        # =============================================================================
        # Inject sources
        # =============================================================================
        
        try:
            
            if autophot_input['inject_source_random']:

                for i in range(0,len(random_sources)):
                    

                    fake_source_i = input_model(xran[i], yran[i],inject_source_mag)

                    if autophot_input['inject_source_add_noise']:

                        nan_idx = np.isnan(fake_source_i)
                        fake_source_i[nan_idx] = 0

                        fake_source_i[fake_source_i<0] = 0

                        fake_source_i = make_noise_image(fake_source_i.shape,
                                                        distribution = 'poisson',
                                                        mean = fake_source_i,
                                                        seed = np.random.randint(0,1e3))
                        
                    fake_sources += fake_source_i
                    
                    ax2.scatter(xran[i],yran[i],
                                marker = 'o',
                                s=150,
                                facecolors='none',
                                edgecolors='r',
                                # label = 'Masked area',
                                alpha = 0.25
                                )
                    ax2.scatter([],[],
                                marker = 'o',
                                facecolors='none',
                                edgecolors='r',
                                alpha = 0.1,
                                label = 'Injected Source')

            if autophot_input['inject_source_on_target']:

                fake_source_on_target = input_model(image.shape[1]/2,image.shape[0]/2,inject_source_mag)

                if autophot_input['inject_source_add_noise']:
                    
                    nan_idx = np.isnan(fake_source_on_target)
                    fake_source_on_target[nan_idx] = 1e-6
                    fake_source_on_target[fake_source_on_target<0] = 0

                    fake_source_on_target = make_noise_image(fake_source_on_target.shape,
                                                    distribution = 'poisson',
                                                    mean = fake_source_on_target,
                                                    seed = np.random.randint(0,1e3))

                fake_sources += fake_source_on_target
                
                ax2.scatter(image.shape[1]/2,image.shape[0]/2,
                            marker = 'o',s=150,
                            facecolors='none',
                            edgecolors='black',
                            alpha = 0.5)
                
                ax2.annotate('On\nTarget', (image.shape[1]/2, -1+image.shape[0]/2),
                             color='black',
                             alpha = 0.5,
                             ha='center')


            im1 = ax2.imshow(image - surface + fake_sources,
                              aspect = 'auto',
                              origin = 'lower',
                             interpolation = 'nearest')
            ax2.set_title(' Injected %s Sources ' % model_label)
            
         

        except Exception as e:
            
            logging.exception(e)
            im1=ax2.imshow(image - surface , origin='lower',aspect = 'auto',)
            ax2.set_title('[ERROR] Fake Sources [%s]' % model_label)


        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = limiting_mag_figure.colorbar(im1, cax=cax)
        cb.ax.set_ylabel('Counts', rotation=270,labelpad = 5)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()

        ax1.axis('off')
        ax2.axis('off')


        lines_labels = [ax.get_legend_handles_labels() for ax in limiting_mag_figure.axes]
        handles,labels = [sum(i, []) for i in zip(*lines_labels)]

        by_label = dict(zip(labels, handles))

        leg = limiting_mag_figure.legend(by_label.values(), by_label.keys(),
                                         bbox_to_anchor=(0.5, 0.9),
                                         loc='lower center',
                                         ncol = 4,
                                         frameon=False)
        
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

        limiting_mag_figure.savefig(autophot_input['write_dir']+'limiting_mag_prob_'+str(autophot_input['base'].split('.')[0])+'.pdf',
                                        bbox_inches='tight',
                                        format = 'pdf')
        plt.close(limiting_mag_figure)

    # master try/except
    except Exception as e:
        print('limit issue')
        logging.exception(e)
        mean = np.nan
        std = np.nan
        mag_level = np.nan


    autophot_input['maglim_mean'] = mean
    autophot_input['maglim_std'] = std

    return mag_level,autophot_input




def fractional_change(i,i_minus_1):
    import numpy as np
    
    delta = abs((i-i_minus_1)/i)
    
    if delta > 100:
        return np.nan
    
    return delta



def inject_sources(autophot_input,
                   image,
                   model = None,
                   r_table = None,
                   plot_steps = False,
                   save_cutouts = False,
                   lmag_guess= None,
                   print_progress = True,
                   save_plot = True,
                   save_plot_to_folder = False):
    '''
    
    Artificial source injection within a given image to determine the maximum magnitude a source may have and not be detected to within a given signal to noise ratio (SNR)
    
    :param autophot_input: Option Dictionary for autophot
    :type autophot_input: dict
    :Required autophot_input keywords for this package:
        
        - c_counts/r_counts (float): Check to make sure PSF successfully created. These param gives the number of counts under a normalised PSF model as well as within the residual table
        - exp_time (float): Exposure time in seconds
        - fwhm (float): Full width half maximum of image in pixels
        - use_moffat (boolean): Boolean on whether to use moffat (True) or Gaussian (False) in PSF model, default to False.
        - image_params (dict): this dictionary describes the analytical model usinged in the PSF model. If a gaussian function is used this dictionary should contain the key "sigma" and the corresponding value. If a moffat function is function this dictionary should contained 'alpha' and 'beta' values'. These values are found and updated the autophot_input in find.fwhm. 
        - zp (float): Zeropoint of the image
        - gain (float): Gain of image in e/ADU
        - ap_size (float): Aperture size in pixels
        - lim_SNR (float): Signal to Noise detection criteria, default to 3
        
        - inject_source_recover_nsteps (int): Number of steps to take, default is 100
        - inject_source_recover_dmag (float): Large magnitude step, default is 0.25 mag
        - inject_source_recover_fine_dmag (float): fine magnitude step, default is 0.1 mag
        - inject_source_add_noise (boolean): Inject random noise with artificial source, default is True
        - inject_source_recover_dmag_redo (int): If poisson noise is added, how many times to re-inject the same source with random noise added, default is 3
        - inject_source_cutoff_sources (float, must be <= 1): Percentage of sources needed to be 'lost'/found' to define the limiting magnitude, default is 0.8
        - write_dir (str): Working directory where plots will be saved
        - base (str): basename of file, used for saving plots
        
    :param image: Cutout around target position. Must be large enough to enclose PSF model.
    :type image: 2D numpy array
    :param model: Model to use for modeling sources. This will accept the PSF model created using psf.build,if None using a 2D analytical function, either Gaussian of Moffat, defaults to None
    :type model: Function, optional
    :param r_table: Residual table from PSF.build. Required to perform artificial source injection with PSF model, defaults to None
    :type r_table: 2D numpy array, optional
    :param plot_steps: Return images of each step during source injection. Will save to folder named "limiting_gif" in output folder, defaults to False
    :type plot_steps: Boolean, optional
    :param save_cutouts: Plot an image of final source injected at limiting magnitude, defaults to False
    :type save_cutouts: Boolean, optional
    :param lmag_guess: Initial guess at limiting magnitude. If None use value from 'inject_source_mag' in autophot_input dictionary, defaults to None
    :type lmag_guess: Float, optional
    :raises Exception: DESCRIPTION
    :return: Returns Value for injected limiting magnitude and updated autophot_input file
    :rtype: Tuple of limiting magnitude (float) and autophot_input file (dictionary)

    '''

    import os
    import logging
    import numpy as np
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    from autophot.packages import psf
    from autophot.packages.functions import calc_mag
    from autophot.packages.functions import SNR
    from autophot.packages.functions import set_size
    from autophot.packages.functions import SNR_err
    
    from photutils.datasets import make_noise_image
    from photutils.datasets.make import apply_poisson_noise
    from autophot.packages.aperture import measure_aperture_photometry
    
    # Get location of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    logger = logging.getLogger(__name__)
    if autophot_input['turn_off_warnings']:
        import warnings
        warnings.filterwarnings("ignore")
    
    
    
    try:
        if r_table is  None:
            # Check if the PSF model is available
            raise Exception('PSF MODEL not available - using Gaussian function')
            
        
        if autophot_input['c_counts']:
            pass
        
            # model_label = 'PSF'
        
        def mag2image(m):
            '''
            Convert magnitude to height of PSF
            '''
            Amplitude  = (autophot_input['exp_time']/(autophot_input['c_counts']+autophot_input['r_counts']))*(10**(m/-2.5))
    
            return Amplitude


        # PSF model that matches close-up shape around target
        def input_model(x,y,flux):
            return model(x, y,0,flux,r_table,autophot_input, pad_shape = image.shape)
            
    except Exception as e:
        
        #if PSF model isn't available - use Gaussian instead

        logger.info('- %s' % e)
        # model_label = 'Gaussian'
    
        sigma = autophot_input['fwhm'] / 2*np.sqrt(2*np.log(2))
        
        r_table = np.zeros((int(2*autophot_input['scale']),int(2*autophot_input['scale'])))
            
            
        def mag2image(m):
            
            '''
            Convert instrumental magnitude of to the amplitude of analytical model
            
            :param m: Instrumental maghitude
            :type m: float
            :return: Amplitude of source in counts
            :rtype: float

            '''
            #  Volumne/counts under 2d gaussian for a magnitude m
            volume =  (10**(m/-2.5)) * autophot_input['exp_time']
    
            # https://en.wikipedia.org/wiki/Gaussian_function
            Amplitude =  volume/(2*np.pi*sigma**2)
    
            return Amplitude
    
        #  Set up grid
    
        def input_model(x,y,A):
            '''
            
            :param x: DESCRIPTION
            :type x: TYPE
            :param y: DESCRIPTION
            :type y: TYPE
            :param A: DESCRIPTION
            :type A: TYPE
            :param image: DESCRIPTION, defaults to image
            :type image: TYPE, optional
            :return: DESCRIPTION
            :rtype: TYPE

            '''
    
            xx,yy= np.meshgrid(np.arange(0,image.shape[1]),np.arange(0,image.shape[0]))
    
            from autophot.packages.functions import gauss_2d,moffat_2d
    
            if autophot_input['use_moffat']:
                
                model = moffat_2d((xx,yy),x,y,0,A,autophot_input['image_params'])
    
            else:
                
                model = gauss_2d((xx,yy),x,y,0,A,autophot_input['image_params'])

    
            return model.reshape(image.shape)
    if print_progress:
        print('\nPerforming full magnitude recovery test using injected sources...')
    
    # Dictionaries to track sources 
    inserted_magnitude = {}
    recovered_magnitude = {}
    recovered_magnitude_e = {}
    recovered_fwhm = {}
    recovered_fwhm_e = {}
    recovered_sigma_detection = {}
    recovered_max_flux = {}
    recovered_sources_list  = {}
    recovered_pos = {}
    recovered_SNR={}
    delta_recovered_SNR={}
    SNR_gradient_change = {}
    beta_gradient_change = {}
    beta_probability = {}
    
    if (lmag_guess is None):
        # Start with this magnitude as an initiall guess
        user_mag_level = autophot_input['inject_source_mag'] - autophot_input['zp']
    else:
        user_mag_level = lmag_guess - autophot_input['zp']
            
    # user_mag_level = autophot_input['inject_source_mag'] - autophot_input['zp']
    
    start_mag = user_mag_level
    
    # List containing detections criteria
    recovered_criteria = {}
    
    # User defined detection limit for magnitude fofset - not using this anymore
    lim_SNR_err =  SNR_err(autophot_input['lim_SNR'])
    
    lim_SNR = autophot_input['lim_SNR']
    
    # Number of steps - this should be quite large to avoid timeing out
    nsteps = autophot_input['inject_source_recover_nsteps']
    
    # Large magnitude step
    dmag   = autophot_input['inject_source_recover_dmag']
    
    # Finer magnitude magnitude step
    fine_dmag = autophot_input['inject_source_recover_fine_dmag']
    
    # If random possion noise is added, how many time is this repeated
    redo   = autophot_input['inject_source_recover_dmag_redo']
    
    # Number of sources
    source_no = autophot_input['inject_source_cutoff_sources']
    
    # Percentage of sources needed to be 'lost' to define the limiting magnitude
    detection_cutout = autophot_input['inject_source_cutoff_limit']
    
    # area under an aperture
    aperture_area = np.pi * (autophot_input['ap_size']*autophot_input['fwhm'])**2
    
    # beta probability limit
    beta_limit = autophot_input['beta_limit']
    
    # make sure correct version is selected
    if detection_cutout > 1:
        detection_cutout = 0.8
        print('Detection limit cannot be > 1 - setting to 0.8')
        
    # No need to inject sources multiple time if not adding poission noise
    if not autophot_input['inject_source_add_noise']:
        redo = 1
    
    # Backstop criteria - this will check that the limiting magnitude is consistent beyond "iter_stop_limit" number of steps
    iter_stop_limit = 5
    iter_stop = 0
    
    # 
    image_median = np.nanmedian(image)
    
    # Choose locations to inject sources - default is around source loaction
    random_sources = PointsInCircum(autophot_input['inject_source_location']*autophot_input['fwhm'],image.shape,n=source_no)
    
    xran = [abs(i[0]) for i in random_sources]
    yran = [abs(i[1]) for i in random_sources]
    
    if autophot_input['injected_sources_additional_sources']:
        
        if autophot_input['injected_sources_additional_sources_position'] == -1:
                    # Move around by one pixel
                    autophot_input['injected_sources_additional_sources_position'] = 1/autophot_input['fwhm']/2
        elif autophot_input['injected_sources_additional_sources_position']<=0:
            print('Soiurce possition offset not set correctly [%1.f], setting to -1' % autophot_input['injected_sources_additional_sources_position'])
            autophot_input['injected_sources_additional_sources_position'] = -1
        
        from random import uniform
        
        for k in range(len(xran)):
            
            for j in range(int(autophot_input['injected_sources_additional_sources_number'])):
                           
                
                    
                dx = autophot_input['injected_sources_additional_sources_position'] * autophot_input['fwhm']/2 * uniform(-1,1)
                dy = autophot_input['injected_sources_additional_sources_position'] * autophot_input['fwhm']/2 * uniform(-1,1)
                
                x_dx = np.array(xran)[k] + dx
                y_dy = np.array(yran)[k] + dy
                
                xran.append(x_dx)
                yran.append(y_dy)
            
    
    injection_df = pd.DataFrame([list(xran),list(yran)])
    injection_df = injection_df.transpose()
    injection_df.columns = ['x_pix','y_pix']
    injection_df.reset_index(inplace = True,drop = True)
    
    
    if print_progress:
        print('Starting Magnitude: %.3f [ mag ]' % (start_mag + autophot_input['zp']))
    
    hold_psf_position =  False
    
    # Measure flux at each point proir than adding any fake sources - to account for an irregular SNR at a specific location
    # autophot_input['plot_PSF_residuals'] = True
    if not autophot_input['inject_lamg_use_ap_phot']:
        
        psf_fit = psf.fit(image ,
                            injection_df,
                            r_table,
                            autophot_input,
                            return_fwhm = True,
                            no_print = True,
                            hold_pos = hold_psf_position,
                            save_plot = False,
                                # remove_background_val = True
                            )
        
        
        psf_params,_ = psf.do(psf_fit,
                              r_table,
                              autophot_input,
                              autophot_input['fwhm'])
        
        psf_flux = psf_params['psf_counts'].values/autophot_input['exp_time']
        psf_flux_err = psf_params['psf_counts_err'].values/autophot_input['exp_time']
        psf_bkg_flux = psf_params['bkg'].values/autophot_input['exp_time']
        psf_bkg_std_flux = psf_params['noise'].values/autophot_input['exp_time']
        psf_heights_flux = psf_params['max_pixel'].values/autophot_input['exp_time']
        
    else:
                   
        positions  = list(zip(injection_df.x_pix.values,injection_df.y_pix.values))
        
        # print(np.nanmedian(image))
    
        psf_counts, psf_heights, psf_bkg_counts, psf_bkg_std = measure_aperture_photometry(positions,
                                                      image ,
                                                      radius = autophot_input['ap_size']    * autophot_input['fwhm'],
                                                      r_in   = autophot_input['r_in_size']  * autophot_input['fwhm'],
                                                      r_out  = autophot_input['r_out_size'] * autophot_input['fwhm'])
        psf_counts_err = 0
            
        psf_flux = psf_counts/autophot_input['exp_time']
        psf_flux_err = psf_counts_err/autophot_input['exp_time']
        psf_bkg_flux = psf_bkg_counts/autophot_input['exp_time']
        psf_bkg_std_flux = psf_bkg_std/autophot_input['exp_time']
        psf_heights_flux = psf_heights/autophot_input['exp_time']


    SNR_source = SNR(flux_star = psf_flux ,
                     flux_sky = psf_bkg_flux,
                     exp_t = autophot_input['exp_time'],
                     radius = autophot_input['ap_size']*autophot_input['fwhm'] ,
                     G  = autophot_input['GAIN'],
                     RN =  autophot_input['RDNOISE'],
                     DC = 0 )
    
    from autophot.packages.functions import beta_value,f_ul

    fake_sources_beta = beta_value(n=3,
                                  sigma = psf_bkg_std_flux,
                                  f_ul = psf_heights_flux)
    
    SNR_source = np.array([round(num, 1) for num in SNR_source])
    # if SNR at a position is greater than detection limit - set the SNR limit at this position to this (higher) limit.
    SNR_source[(SNR_source < autophot_input['lim_SNR']) | (np.isnan(SNR_source))] = autophot_input['lim_SNR']
    
    injection_df['limit_SNR'] = SNR_source
    injection_df['initial_beta'] = fake_sources_beta
    injection_df['initial_noise'] = psf_bkg_std_flux
    injection_df['initial_peak_flux'] = psf_heights_flux
    injection_df['f_ul'] = f_ul(3,beta_limit,psf_bkg_std_flux)
    
    if not autophot_input['inject_lamg_use_ap_phot']:
        # update to new positions
        injection_df.rename(columns={"x_pix": "x_pix_OLD",
                                      "y_pix": "y_pix_OLD"})
        injection_df['x_pix'] = psf_params['x_pix'].values
        injection_df['y_pix'] = psf_params['y_pix'].values
    
    # check locations with high SNR (due to random noise spikes) and ignoring them
    
    if not autophot_input['subtraction_ready'] or not autophot_input['injected_sources_use_beta'] :
        good_pos = injection_df.limit_SNR <= autophot_input['lim_SNR']
    else:
        print('Checking for good beta locations')
        good_pos = injection_df.initial_beta < 0.5
    
    
    # print(injection_df.initial_beta)
    if np.sum(~good_pos)>0:
        print('Ignoring %d / %d sources with high SNR' % (np.sum(~good_pos),len(good_pos)))
        injection_df = injection_df[good_pos]
    # print(injection_df)
    
    if np.sum(~good_pos) == len(good_pos):
        print('Could not find any suitable area to testing artifical source injection ')
        return np.nan,autophot_input
        
    from autophot.packages.functions import get_distinct_colors
    cols = get_distinct_colors(len(injection_df))

    # Begin each list for each source - used in plotting
    for k in range(len(injection_df)):
        
        inserted_magnitude [k] = {}
        recovered_magnitude[k] = {}
        recovered_magnitude_e[k] = {}
        recovered_fwhm[k] = {}
        recovered_fwhm_e[k] = {}
        recovered_sigma_detection[k] = {}
        recovered_max_flux[k] = {}
        recovered_sources_list[k] = []
        recovered_SNR[k]={}
        delta_recovered_SNR[k]={}
        
        recovered_pos[k] = {}
        SNR_gradient_change[k]={}
        beta_gradient_change[k]={}
        
        beta_probability[k]={}
        
    # magnitude increments - inital set to make sources fainter
    dmag_range = np.linspace(0,dmag*nsteps,int(nsteps+1))
    
    # USed for saving images
    image_number = 0

    # Are sources getting brighter or fainter - start off with fainter - swap to negative gradient if sources are initial not detected
    gradient = 1

    # criteria to decide it sources are to be detected or not to find limiting magnitude
    citeria = False

    # initialise limiting magnitude
    inject_lmag = None
    lmag_found = False

    # dmag fine scale - inital parameters
    use_dmag_fine = False
    fine_dmag_range = None
    
    # Inital counter
    ith = 0 
    
    # Initial detection for display purposes
    detect_percentage = 100

    while True:
        try:
        
            if use_dmag_fine:
                # Use small magnitude step size
                dmag_step = gradient*fine_dmag_range[ith]
                 
                dmag_step_minus_1 = gradient*fine_dmag_range[ith-1]
                
            else:
                # Use larger step size
                dmag_step =  gradient*dmag_range[ith]
                
                dmag_step_minus_1 = gradient*dmag_range[ith-1]
                
            # step labels to keep track of everything
            step_name = round(start_mag+dmag_step+autophot_input['zp'],3)
            previous_step_name = round(start_mag+dmag_step_minus_1+autophot_input['zp'],3)
            
            # Sources are put in one at a time to avoid contamination
            for k in range(len(injection_df)):
                
                SNR_no_source = injection_df['limit_SNR'].values[k]
                
                inserted_magnitude[k][step_name] = []
                recovered_magnitude[k][step_name] = []
                recovered_magnitude_e[k][step_name] = []
                recovered_fwhm[k][step_name] = []
                recovered_fwhm_e[k][step_name] = []
                recovered_sigma_detection[k][step_name] = []
                recovered_max_flux[k][step_name] = []
                # recovered_sources_list[k][step_name]  = []
                recovered_SNR[k][step_name] = []
                recovered_pos[k][step_name] = []
                SNR_gradient_change[k][step_name] = []
                beta_gradient_change[k][step_name] = []
                beta_probability[k][step_name] = []
                
                for j in range(redo):
                    
                    if print_progress:
                        print('\rStep: %d / %d :: Source %d / %d :: Iteration  %d / %d :: Mag %.3f :: Sources detected: %d%%' % (ith+1,nsteps,k+1,len(injection_df),j+1,redo,start_mag+dmag_step+autophot_input['zp'],detect_percentage),
                               end = '',
                               flush = True)
                
                    fake_source_on_target = input_model(injection_df['x_pix'].values[k],
                                                        injection_df['y_pix'].values[k],
                                                        mag2image(start_mag+dmag_step))
                                                        
                    if autophot_input['inject_source_add_noise']:
                        
                        # add random possion noise to artifical star
                        nan_idx = np.isnan(fake_source_on_target)
                        neg_idx = fake_source_on_target < 0
            
                        fake_source_on_target[nan_idx] = 0
                        fake_source_on_target[neg_idx] = 0
            
                        fake_source_on_target = make_noise_image(fake_source_on_target.shape,
                                                                 distribution = 'poisson',
                                                                 mean = fake_source_on_target,
                                                                 seed = np.random.randint(0,1e3))
            
                        fake_source_on_target = apply_poisson_noise(fake_source_on_target)
                    
                    if not autophot_input['inject_lamg_use_ap_phot']:
                        
                        psf_fit = psf.fit(image + fake_source_on_target,
                                            injection_df.iloc[[k]], 
                                            r_table,
                                            autophot_input,
                                            return_fwhm = True,
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
                        psf_counts_err = psf_params['psf_counts_err'].values
                        psf_bkg_counts = psf_params['bkg'].values
                        psf_bkg_std = psf_params['noise'].values
                        psf_height =  psf_params['max_pixel'].values
                        
                    else:
                    
                        positions  = list(zip(injection_df.iloc[[k]].x_pix.values,injection_df.iloc[[k]].y_pix.values))
                        
                        # print(np.nanmedian(image))
    
                        psf_counts,psf_height,psf_bkg_counts,psf_bkg_std = measure_aperture_photometry(positions,
                                                                      image + fake_source_on_target,
                                                                      radius = autophot_input['ap_size']    * autophot_input['fwhm'],
                                                                      r_in   = autophot_input['r_in_size']  * autophot_input['fwhm'],
                                                                      r_out  = autophot_input['r_out_size'] * autophot_input['fwhm'])
                        psf_counts_err = 0
                        
                        
                        
                    psf_flux = psf_counts/autophot_input['exp_time']
                    psf_flux_err = psf_counts_err/autophot_input['exp_time']
                    psf_bkg_flux = psf_bkg_counts/autophot_input['exp_time']
                    psf_bkg_std_flux = psf_bkg_std/autophot_input['exp_time']
                    psf_height_flux = psf_height/autophot_input['exp_time']
                    
                    # from autophot.packages.functions import beta_value,f_ul

                    fake_target_beta = beta_value(n=3,
                                                  sigma = psf_bkg_std_flux,
                                                  f_ul = psf_height_flux,
                                                  # noise = injection_df['initial_peak_flux'].values[k]
                                                  )
                
                    beta_probability[k][step_name].append(1-fake_target_beta)
                

                    SNR_source_i = SNR(flux_star = psf_flux ,
                                        flux_sky = psf_bkg_flux  ,
                                        exp_t = autophot_input['exp_time'],
                                        radius = autophot_input['ap_size']*autophot_input['fwhm'] ,
                                        G  = autophot_input['GAIN'],
                                        RN =  autophot_input['RDNOISE'],
                                        DC = 0 )


                    recovered_sigma_detection[k][step_name].append( psf_height_flux / injection_df['f_ul'].values[k])
                    # recovered_sigma_detection[k][step_name].append((psf_height_flux) / psf_bkg_std_flux)
                    recovered_max_flux[k][step_name].append(psf_height_flux)
                    
                    mag_recovered =  calc_mag(psf_flux,0)
                    mag_recovered_error = calc_mag(psf_flux,0) - calc_mag(psf_flux+psf_flux_err,0)
                    

                    inserted_magnitude[k][step_name].append(start_mag+dmag_step)
                    recovered_magnitude[k][step_name].append(mag_recovered[0])
                    recovered_magnitude_e[k][step_name].append(mag_recovered_error[0])                  

                    # recovered SNR for each source (i.e.at each position) at this mag step
                    recovered_SNR[k][step_name].append(SNR_source_i[0])
                    
                    recovered_SNR_k = recovered_SNR[k][step_name]

                 
                    try:

                        beta_gradient = (float(beta_probability[k][step_name][j]) - float(beta_probability[k][previous_step_name][j])) / (  dmag_step - dmag_step_minus_1 )
                        next_step = True
                        
                    except Exception as e:
                        # print('Previous step not defined: setting to inf: error->%s ' %e)
                        
                        SNR_gradient = np.inf

                        beta_gradient = np.inf
                        next_step = False
    

                    SNR_gradient_change[k][step_name].append(SNR_gradient)
                    
                    beta_gradient_change[k][step_name].append(beta_gradient)

                    # plot_steps = True
                    if plot_steps:
                        
                        plt.ioff()
                        
                        fig = plt.figure(figsize = set_size(250,1))
                        
                        ax1 = fig.add_subplot(111)
    
                        ax1.imshow(image+fake_source_on_target,origin = 'lower')
                        ax1.set_xlabel('X Pixel')
                        ax1.set_ylabel('Y Pixel')
                        
                        if psf_height_flux < 3*psf_bkg_std_flux:
                            text = r'Source lost'
                            color = 'red' 
                        else:
                            text = r'Source recovered'
                            color = 'green'
                        
                        ax1.annotate(text,
                            xy=(0.9, 0.9),
                            xycoords='axes fraction',
                            va = 'top',
                            ha = 'right',
                            color = color,
                            bbox=dict(
                                      fc="white",
                                      ec="black",
                                      lw=2)
                                )
                        
                        ax1.set_title(r' Detection: %.1f$\sigma_{bkg}$' % (psf_height_flux/psf_bkg_std_flux))
                        
                        ax1.scatter(injection_df['x_pix'].values[k],injection_df['y_pix'].values[k],
                                    marker = 'o',
                                    facecolors='none',
                                    edgecolors='r',
                                    s= 100)
                        
                        
                        gif_images_path = os.path.join(autophot_input['write_dir'],'limiting_gif')
                        
                        os.makedirs(gif_images_path, exist_ok=True)
                        
                        
                        plt.savefig(os.path.join(gif_images_path,'img_%d.jpg') % image_number,bbox_inches = 'tight')
                        plt.close(fig)
                        
                        image_number+=1

    
            if autophot_input['subtraction_ready'] or autophot_input['injected_sources_use_beta']:
                # recovered_sources = np.concatenate([abs(np.array(recovered_magnitude[k][step_name]) - np.array(inserted_magnitude[k][step_name])) < lim_SNR_err for k in range(len(injection_df))])
                # recovered_sources = np.concatenate([np.array(beta_probability[k][step_name]) <= beta_limit for k in range(len(injection_df))])
                recovered_sources = np.concatenate([np.array(recovered_max_flux[k][step_name]) >= injection_df['f_ul'].values[k]  for k in range(len(injection_df))])
            else:
  
                recovered_sources = np.concatenate([np.array(recovered_SNR[k][step_name]) >= injection_df['limit_SNR'].values[k] for k in range(len(injection_df))])
            
            
            # For the sources injects - did they pass the recovered test?
            # TODO: make sure this makes sense in both criteria
            detect_percentage = 100*np.sum(recovered_sources)/len(recovered_sources)
            
            recover_test = np.sum(recovered_sources)/len(recovered_sources) >=  1-detection_cutout

            recovered_criteria[step_name] = recover_test
    
            # If the first source inject comes back negative on the first interation - flip the dmag steps i.e. fainter -> brighter
            if recovered_criteria[step_name] == citeria and ith == 0 and not use_dmag_fine:
                if print_progress:
                    print('\nInitial injection sources not recovered - injecting brighter sources')
                
                gradient *= -1
                
                citeria = not citeria 
            
            # else if the detection meetings the criteria i.e. sources are (not) detected 
            elif recovered_criteria[step_name] == citeria:
                
                
                if iter_stop == 0 and not use_dmag_fine:
                    if print_progress:
                        print('\n\nApproximate limiting magnitude: %.3f - using finer scale\n' % (autophot_input['zp']+start_mag+dmag_step))
                    
                    use_dmag_fine = True
    
                    fine_nsteps = int(1/fine_dmag)
                    
                    # reset start magnitude
                    start_mag = start_mag+dmag_step
                    
                    fine_dmag_range =  np.linspace(0,fine_dmag*fine_nsteps,int(fine_nsteps))
                    
                    gradient *= -1
                    
                    citeria = not citeria 
                    
                    # Restart the count
                    ith = 0
                    
                elif iter_stop == 0:
                    
                    if print_progress:
                        print('\n\nLimiting mag found, checking overshoot...\n')
                    
                    # First time finer scale meets criteria, not magnitude
                    inject_lmag = start_mag + dmag_step
                    inject_lmag_minus_1 = start_mag + gradient*fine_dmag_range[ith-1]
                    
                    lmag_found = True
                
                    iter_stop+=1
                    
                else:
                    # print('Resetting...')
                    
                    iter_stop+=1
                
            else:
                
                if lmag_found:
                    if print_progress:
                        print('\n\nOvershoot discrepancy, resetting...\n')
                
                # didn't meet criteria, keep going
                lmag_found = False
                iter_stop = 0
                
            if iter_stop > iter_stop_limit:
                #Done
                if print_progress:
                    print('\nLimiting magnitude: %.3f \n' % ( inject_lmag+autophot_input['zp'] ))
                    
                
                break
            
            else:
                
                ith = ith + 1
                
                
                
        except Exception as e:
            import os,sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # print(exc_type, fname, exc_tb.tb_lineno)
            print('\nInjection failed: ' + str(e) )
            print(exc_type, fname, exc_tb.tb_lineno)
            inject_lmag_minus_1 = np.nan
            inject_lmag = np.nan
            save_cutouts = False
            break

    injection_params = {'mag': inserted_magnitude,
                        'inject_mag': inserted_magnitude,
                        'recover_mag': recovered_magnitude,
                        'recover_mag_e': recovered_magnitude_e,
                        'recover_fwhm':recovered_fwhm,
                        # 'recover_fwhm_e': recovered_fwhm_e,
                        'recovered_SNR':recovered_SNR,
                        'd_SNR':SNR_gradient_change,
                        'd_beta':beta_gradient_change
                        }
    

    if autophot_input['injected_sources_save_output']:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_dict = {}
            df_dict['source'] = []
        
            for k in range(len(injection_df)):
                
                for param_key, param_values in injection_params.items():
                    df_dict[param_key] = []
                    for key,val in param_values.items():
                        if param_key == 'mag':
                            val1 = list(val.values()) + autophot_input['zp']
                        else:
                            val1 = list(val.values())
                        df_dict[param_key]+=[np.mean(i) if not np.isnan(np.mean(i)) else 999 for i in val1 ]
                        
                df_dict['source'] += [k] * len(val.values())
                
            
                # warnings.simplefilter('error', UserWarning)
                recover_df = pd.DataFrame(df_dict)

    image_minus_1 = image.copy()
    image_limited = image.copy()
    
    # save_cutouts = True
    if save_cutouts:
        
        cutout_size = 2 * autophot_input['fwhm']
        
        rows = int(np.ceil(len(injection_df)/2))
        
        if rows % 2 != 0:
            rows=rows+1

        plt.ioff()
        fig = plt.figure(figsize = set_size(250,2))
        
        gs = gridspec.GridSpec(rows, 2)
        
        gs.update(wspace=0., hspace=0.)
        
        for idx in range(len(injection_df)):
            
            ax = fig.add_subplot(gs[idx])
            
            
            fake_kth_source = input_model(injection_df.x_pix.values[idx],
                                          injection_df.y_pix.values[idx],
                                          mag2image(inject_lmag_minus_1),
                                          )
            
            image_kth = image_limited + fake_kth_source
            
            
            kth_cutout = image_kth[int(injection_df.y_pix.values[idx]-cutout_size): int(injection_df.y_pix.values[idx] + cutout_size),
                                   int(injection_df.x_pix.values[idx]-cutout_size): int(injection_df.x_pix.values[idx] + cutout_size)]
            
            ax.imshow(kth_cutout,
                      interpolation = None,
                       aspect = 'equal',
                      origin = 'lower')
            
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
            circle = plt.Circle((kth_cutout.shape[1]/2,kth_cutout.shape[1]/2), 1*autophot_input['fwhm'], 
                            color='r',
                            lw = 0.25,
                            fill=False)
            ax.add_patch(circle)
            

            
        fig.savefig(autophot_input['write_dir']+'inject_lmag_cutouts_'+str(autophot_input['base'].split('.')[0])+'.pdf',
                    format = 'pdf')
            
        plt.close(fig)

    # Plot the results
   
    # TODO: fix this plot
    # save_plot = True
    if save_plot :
        
        import matplotlib.ticker as ticker
        
        
        plt.ioff()
        fig = plt.figure(figsize = set_size(250,3))
        layout = gridspec.GridSpec(ncols=3, 
                                   nrows=4,
                                   figure=fig,
                                   wspace = 0.25,
                                    # hspace = 0.15,
                                   # height_ratios=[1,1,0.75,0.75],
                                   )
        ax1 = fig.add_subplot(layout[0, :])
        ax2 = fig.add_subplot(layout[1, :], sharex=ax1)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax3 = fig.add_subplot(layout[2:3, 0:1])
        ax4 = fig.add_subplot(layout[3:4, 0:1])
        
        
        ax5 = fig.add_subplot(layout[2:3, 1:2])
        ax6 = fig.add_subplot(layout[2:3, 2:3])
        ax7 = fig.add_subplot(layout[3:4, 1:2])
        ax8 = fig.add_subplot(layout[3:4, 2:3])
                
        # import itertools
        
        # marker = itertools.cycle((',', '+', '.', 'o', '*')) 
        if not autophot_input['subtraction_ready'] or not autophot_input['injected_sources_use_beta']:
            for k in range(len(injection_df)):
                for i in inserted_magnitude[k].keys():
                    markers, caps, bars = ax1.errorbar(inserted_magnitude[k][i]+autophot_input['zp'],recovered_magnitude[k][i]+autophot_input['zp'],
                                  yerr = [np.sqrt(i**2 + autophot_input['zp_err']**2) for i in recovered_magnitude_e[k][i]],
                                  ls = '',
                                  marker = 'o',
                                  ecolor = 'black',
                                  color = cols[k],
                                  # edgecolor = None,
                                  label = 'Recovered Magnitude')
                    
            for i in inserted_magnitude[k].keys():
                ax1.plot(inserted_magnitude[k][i]+autophot_input['zp'],
                         inserted_magnitude[k][i]+autophot_input['zp'],
                          ls = '--',
                          color = 'red',
                          alpha = 0.5,
                          label = 'True Magnitude')
            
            xrange = np.linspace(ax1.get_xlim()[0],ax1.get_xlim()[1])
            
            ax1.fill_between(xrange, 
                             xrange-lim_SNR_err,
                             xrange+lim_SNR_err,
                             # ls = '--',
                             color = 'red',
                             alpha = 0.5,
                             label = r'%d$\sigma_{bkg}$' % autophot_input['lim_SNR'])
            
            [bar.set_alpha(0.25) for bar in bars]
            [cap.set_alpha(0.25) for cap in caps]
        
            ax1.set_ylim(ax1.get_xlim()[0]-lim_SNR_err,ax1.get_xlim()[1]+lim_SNR_err)
            
            ax1.set_ylabel(r'$M_{Recovered}$')
            
        else:
            
            f_ul_range = []
            # ax21 = ax2.twinx()
            
            
            for i in inserted_magnitude[k].keys():
                for k in range(len(injection_df)):
                
                    markers, caps, bars = ax1.errorbar(inserted_magnitude[k][i]+autophot_input['zp'],
                                                       recovered_sigma_detection[k][i],
                                                        # yerr = [np.sqrt(i**2 + autophot_input['zp_err']**2) for i in recovered_magnitude_e[k][i]],
                                                        ls = '',
                                                        marker = 'o',
                                                        ecolor = 'black',
                                                        color = cols[k],
                                                        # edgecolor = None,
                                                        label = '')
                    
   
                f_ul_range.append(f_ul(3,beta_limit,injection_df['initial_noise'].values[k]) / injection_df['initial_noise'].values[k])
 
            
            ax1.set_ylabel(r'$F_{fake} / F_{UL}$')
            
            

        x = []
        y = []
        
        cum_detection = {}
            
        ax21 = ax2.twinx()
        
        for i in inserted_magnitude[k].keys():
            
            cum_detection[i] = []
            
            for k in range(len(injection_df)):
            
                # print(autophot_input['subtraction_ready'],autophot_input['injected_sources_use_beta'])
                if not autophot_input['subtraction_ready'] and not autophot_input['injected_sources_use_beta']:
                    markers, caps, bars = ax2.errorbar(inserted_magnitude[k][i]+autophot_input['zp'],
                                                       recovered_SNR[k][i],
                                                      # yerr = recovered_fwhm_e,
                                                       ls = '',
                                                       marker = 'o',
                                                       ecolor = 'black',
                                                       color = cols[k],
                                                       label = r'Recovered SNR')
                    [bar.set_alpha(0.25) for bar in bars]
                    [cap.set_alpha(0.25) for cap in caps]
                    
                    cum_detection[i].append(recovered_SNR[k][i][0])
                    
                else:
                    
        
            
                    ax2.scatter(inserted_magnitude[k][i]+autophot_input['zp'],
                                  1-np.array(beta_probability[k][i]),
                                
                                    color = cols[k],
                                    label = r"$1-\beta'$")
                    
                    
                    cum_detection[i].append(1-np.array(beta_probability[k][i]))
                    
                    
        for i in inserted_magnitude[k].keys():
            
            if  autophot_input['subtraction_ready'] or autophot_input['injected_sources_use_beta']:
                detected_percent = np.sum(np.array(cum_detection[i]) <= beta_limit) / len(cum_detection[i])
                
            else:

                detected_percent = np.sum(np.array(cum_detection[i]) <= lim_SNR) / len(cum_detection[i])
                
            
            cum_detection[i] = detected_percent
            
        # print(cum_detection)
        x = np.array(list(cum_detection.keys()))
        idx = np.argsort(x)
        x = x[idx]
        # TODO I'm not sure why this isn't normailsed to 100%
        
        y = np.array(list(cum_detection.values()))[idx]
   
        ax21.plot(x,y,
                  color = 'black',
                  marker = 'o',
                  label = 'Cumlative Detections')
                               
        ax21.set_ylim(-0.05,1.05)
        

        import matplotlib.ticker as mticker
        
        # fixing yticks with matplotlib.ticker "FixedLocator"
        ticks_loc = ax21.get_yticks().tolist()
        ax21.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax21.set_yticklabels([str(int(x*100))+'%' for x in ticks_loc])


        ax21.annotate('Detection Cutoff [%d%%]' % int((detection_cutout)*100), xy=(1,detection_cutout),
             xytext = (0.90,detection_cutout),
             va = 'center',
             ha = 'right',
             color = 'black',
             fontsize = 6,
             xycoords = ax21.get_yaxis_transform(),  
             arrowprops=dict(arrowstyle="->", color='black',lw = 0.5),
             # bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1')
             annotation_clip=False)
        
        
        if not autophot_input['subtraction_ready'] and  not autophot_input['injected_sources_use_beta']:
            ax2.axhline(3,
                        color = 'green',
                        ls = '--',
                        label = r'3\\sigma_{bkg}')
            
            ax2.axvline(inject_lmag+autophot_input['zp'],
                        color = 'blue',
                        ls = '--',
                        label = r'Detection Limit')
            ax2.set_ylabel(r'Signal to Noise Ratio [$\sigma_{bkg}$]')
            
        else:
            
            ax2.set_ylabel(r"Detection Probability [1-$\beta'$]")
            ax21.set_ylabel(r"Sources lost [%]")
        
        text_lim = '$M_{lim} = %.3f$' % float(inject_lmag+autophot_input['zp'])
        
        ax2.annotate(text_lim, xy=(inject_lmag+autophot_input['zp'],0),
             xytext = (inject_lmag+autophot_input['zp'],-0.2),
             va = 'center',
             ha = 'center',
             color = 'red',
             fontsize = 6,
             xycoords = ax2.get_xaxis_transform(),  
             arrowprops=dict(arrowstyle="->", color='red',lw = 0.5),
             annotation_clip=False)
            
        
        ax3.imshow(image,interpolation = None,origin = 'lower') 
        ax3.set_title(r'No fake sources')
        
        from matplotlib.lines import Line2D   
        
        red_circle = Line2D([0], [0], marker='o',
                            label='Test locations',
                            markerfacecolor='none',
                            markeredgecolor='black',
                            markersize = 7)
        ax3.legend(handles=[red_circle],
                   loc = 'upper right',
                   frameon = False)
        
        
        for k in range(len(injection_df)):
                
            circle = plt.Circle((injection_df['x_pix'].values[k],injection_df['y_pix'].values[k]),
                                1.3*autophot_input['fwhm'], 
                                color=cols[k],
                                ls = '--',
                                lw = 0.25,
                                fill=False)

            ax3.add_patch(circle)
        
        if autophot_input['plot_injected_sources_randomly']:
            spaced_sample = sample_with_minimum_distance(n=[int(autophot_input['scale']/2),
                                                            int(image.shape[0]-autophot_input['scale']/2)
                                                            ], 
                                                         k=4,
                                                         d=int(1.5*autophot_input['fwhm']))
            x_spaced = [i[0] for i in spaced_sample]
            y_spaced = [i[1] for i in spaced_sample]

        else:
            x_spaced = injection_df['x_pix'].values[k]
            y_spaced = injection_df['y_pix'].values[k]

    
    
        for k in range(len(x_spaced)):
            fake_source_on_target = input_model(x_spaced[k],
                                                y_spaced[k],
                                                mag2image(inject_lmag))
            
            image_limited+=fake_source_on_target
            
        ax4.imshow(image_limited,
                   interpolation = None,
                   origin = 'lower')
        
        # ax4.set_title(r'$M_{lim} = %.3f~mag$' % (inject_lmag+autophot_input['zp']))
        ax4.set_title('Randomly Injected sources')
        
        for ax in [ax4]:
            
            for k in range(len(x_spaced)):
                
                circle = plt.Circle((x_spaced[k],y_spaced[k]),
                                    1.3*autophot_input['fwhm'], 
                                    color='black',
                                    ls = '--',
                                    lw = 0.25,
                                    fill=False)
                ax.add_patch(circle)
                
                ax.text(x_spaced[k],y_spaced[k], 
                      str(k), 
                      va = 'center',
                      ha = 'center',
                      # fontsize = 7,
                      color='black',
                      fontsize=5)
                
        closeup_axes = [ax5,ax6,ax7,ax8]
        
        for i in  range(len(closeup_axes)):
            ax = closeup_axes[i]
            fake_source_on_target = input_model(x_spaced[i],
                                                y_spaced[i],
                                                mag2image(inject_lmag))
            
            
            inject_image = image+fake_source_on_target
            
            
            ax.imshow(inject_image[int(y_spaced[i]-autophot_input['scale']/2):int(y_spaced[i]+autophot_input['scale']/2),
                                   int(x_spaced[i]-autophot_input['scale']/2):int(x_spaced[i]+autophot_input['scale']/2)],
                      # aspect = 'auto',
                      interpolation = None,
                      origin = 'lower')
            ax.set_title('Position: %d' % i)
            # ax.axis('off')
            

        
        
        ax1.axvline(inject_lmag+autophot_input['zp'],color='black',ls=':',alpha=0.5)
        ax2.axvline(inject_lmag+autophot_input['zp'],color='black',ls=':',alpha=0.5)
        
        if abs(ax1.get_xlim()[1] - ax1.get_xlim()[0]) <1:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(fine_dmag))
        for ax in [ax3,ax5,ax6]:
            pos1 = ax.get_position() # get the original position 
            pos2 = [pos1.x0 , pos1.y0 - 0.03 ,  pos1.width , pos1.height] 
            ax.set_position(pos2) # set a new position
        
        ax2.set_ylim(-0.05,1.05)
        
        # ax21.set_ylabel(r'$SNR_{i-1} - SNR_{i}$')
        ax2.set_xlabel(r'$M_{Injected}$')
        
        

        ax1.label_outer() 
    
        if save_plot_to_folder:
            
            save_loc = os.path.join(autophot_input['write_dir'],'lmag_analysis')

            os.makedirs(save_loc, exist_ok=True)
            save_name =  os.path.join(save_loc,'Inject_lmag_'+str(autophot_input['base'].split('.')[0])+'_0'+'.pdf' )
            count = 1
            while os.path.exists(save_name):
                fname = 'Inject_lmag_'+str(autophot_input['base'].split('.')[0])+'_%d' % count 
                save_name =  os.path.join(save_loc,fname + '.pdf')
                count+=1
            fig.savefig(save_name,bbox_inches = 'tight',format = 'pdf')
        
        
        else:
        
            fig.savefig(autophot_input['write_dir']+'Inject_lmag_'+str(autophot_input['base'].split('.')[0])+'.pdf',bbox_inches = 'tight',format = 'pdf')
  
        plt.close(fig)
    
    if autophot_input['injected_sources_save_output']:
        import warnings
    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            recover_df.round(3).to_csv(autophot_input['write_dir']+'inject_lmag_'+str(autophot_input['base'].split('.')[0])+'.csv')
        
    return inject_lmag,autophot_input
    
    
    
    
