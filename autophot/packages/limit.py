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

def detection_probability(n,sigma,beta ):
    
    from scipy.special import erfinv
    import numpy as np

    '''

    Probabilistic upper limit computation base on:
    http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf

    Assuming Gassauin nose distribution

    n: commonly used threshold value integer above some background level

    sigma: sigma value from noise distribution found from local area around source

    beta: Detection probability


    '''
    counts_upper_limit = (n + np.sqrt(2)*erfinv(2*beta - 1)) * sigma

    return counts_upper_limit

def PointsInCircum(r,shape,n=8):
    
    '''
    Generate series of x,y coordinates a distance r from the center of the image
    
    :param r: Distance from center of image 
    :type r: Float

    :param shape: Image shape
    :type shape: Tuple
    :param n: Number of points, defaults to 8
    :type n: Integert, optional
    :return: List of x,y coorindates placed around the center of the image  at angles 2*pi/n * i for each ith sources in the n sample size
    :rtype: List of tuples

    '''
    import numpy as np
    
    return [(np.cos(2*np.pi/n*x)*r + shape[1]/2 ,np.sin(2*np.pi/n*x)*r + shape[0]/2) for x in range(0,n)]
    


def limiting_magnitude_prob(syntax,image,model = None,r_table = None):

    """Find the probable limiting magnitude arond a given tagrte location.
    Methodlology discussed in  'F. Masci 2011 <https://www.semanticscholar.org/paper/Computing-flux-upper-limits-for-non-detections-Masci/6c14bb440637fb6c6c3bad6ce42c2bca7383c735>'__.

    :param syntax: AutoPhOT control dictionary
    :type syntax: dict
    :Required syntax keywords for this package:
        
        - c_counts/r_counts (float): Check to make sure PSF successfully created. These param gives the number of counts under a normalised PSF model as well as within the residual table
        - exp_time (float): Exposure time in seconds
        - fwhm (float): Full width half maximum of image in pixels
        - use_moffat (boolean): Boolean on whether to use moffat (True) or Gaussian (False) in PSF model, default to False.
        - image_params (dict): this dictionary describes the analytical model usinged in the PSF model. If a gaussian function is used this dictionary should contain the key "sigma" and the corresponding value. If a moffat function is function this dictionary should contained 'alpha' and 'beta' values'. These values are found and updated the syntax in find.fwhm. 
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
        syntax: updated AutoPhOT control dictionary
        
    """
    try:
        import matplotlib.pyplot as plt
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))
        from photutils import CircularAperture
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib.gridspec import  GridSpec
        import random

        from scipy.optimize import curve_fit
        import warnings

        from photutils.datasets import make_noise_image
        from photutils import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from autophot.packages.rm_bkg import rm_bkg
        from autophot.packages.functions import set_size

        from astropy.visualization import  ZScaleInterval
        
        from autophot.packages.functions import gauss_1d

        import logging

        logger = logging.getLogger(__name__)

        # level for detection - Rule of thumb ~ 5 is a good detection level
        level = syntax['lim_SNR']

        logger.info('Limiting threshold: %d sigma' % level)

        image_no_surface,surface = rm_bkg(image,syntax,image.shape[0]/2,image.shape[0]/2)

        # =============================================================================
        # find and mask sources in close up
        # =============================================================================

        image_mean, image_median, image_std = sigma_clipped_stats(image,
                                        sigma = syntax['bkg_level'],
                                        maxiters = 10)
        
        if not syntax['subtraction_ready']:

            daofind = DAOStarFinder(fwhm    = syntax['fwhm'],
                                    threshold = syntax['bkg_level']*image_std,
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
                positions = [(image.shape[0]/2,image.shape[1]/2)]

        # "size" of source
        source_size =  2*syntax['fwhm']

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

        vmin,vmax = (ZScaleInterval(nsamples = 500)).get_limits(mask_image)

        excluded_points = np.where(mask_image == 0)

        exclud_x = excluded_points[0]
        exclud_y = excluded_points[1]

        exclud_zip = list(zip(exclud_x,exclud_y))

        included_points = np.where(mask_image != 0)

        includ_x = list(included_points[0])
        includ_y = list(included_points[1])

        includ_zip = list(zip(includ_x,includ_y))

        number_of_points = 150

        fake_points = {}


        if len(includ_zip) < pixel_number:

            includ_zip=includ_zip+exclud_zip

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



        popt,pcov = curve_fit(gauss_1d,center,hist,
                              p0=[A_guess,mean_guess,sigma_guess],
                              absolute_sigma=True )

        mean = popt[1]
        std  = abs(popt[2])

        logging.info('Mean: %s - std: %s' % (round(mean,3),round(std,3)))

        if syntax['probable_detection_limit']:

            beta = float(syntax['probable_detection_limit_beta'])

            logging.info("Using Probable detection limit [b' = %d%% ]" % (100 * beta))

            count_ul = mean+detection_probability(level,std,beta)

            logging.info("Count Upper limit: %.3f" % count_ul)

        else:
            
            count_ul = abs(mean + level*std)
            
            logging.info('Detection at %s std: %.3f' % (level,count_ul))

        # =============================================================================
        # Plot histogram of background values
        # =============================================================================

        limiting_mag_figure = plt.figure(figsize = set_size(250,aspect = 1))

        ncols = 2
        nrows = 2
        # heights = []
        heights = [0.5,1]
        # gs = gridspec.GridSpec(2, 2)
        gs = GridSpec(nrows, ncols ,
                      wspace=0.5,
                      hspace=0.5,
                      height_ratios=heights,
                       # width_ratios = widths
                       )
        
        ax0 = limiting_mag_figure.add_subplot(gs[0, :])
        ax1 = limiting_mag_figure.add_subplot(gs[1, 0])
        ax2 = limiting_mag_figure.add_subplot(gs[1, 1])
        
        line_kwargs = dict(alpha=0.5,color='black',ls = '--')

        # ax1.scatter(exclud_y,exclud_x,color ='black',marker = 'X',alpha = 0.5  ,label = 'excluded_pixels',zorder = 1)
        ax1.scatter(includ_y,includ_x,
                    color ='red',
                    marker = '+',
                    alpha = 0.25,
                    label = 'Included pixels',
                    zorder = 2)

        # the histogram of the data
        n, bins, patches = ax0.hist(list(fake_mags.values()),
                                    density=True,
                                    bins = 30,
                                    facecolor='blue',
                                    alpha=1,
                                    label = 'Pseudo-Counts\nDistribution')

        ax0.axvline(mean,**line_kwargs)
        ax0.axvline(mean + 1*std,**line_kwargs)
        ax0.text(mean + 1*std,np.max(n),r'$1\sigma$',rotation = -90,va = 'top')
        ax0.axvline(mean + 2*std,**line_kwargs)
        ax0.text(mean + 2*std,np.max(n),r'$2\sigma$',rotation = -90,va = 'top')

        if syntax['probable_detection_limit']:

            ax0.axvline(count_ul,**line_kwargs)
            ax0.text(count_ul,np.max(n),r"$\beta'$ = %d%%" % (100*beta),rotation = -90,va = 'top')

        else:
            ax0.axvline(mean + level*std,**line_kwargs)
            ax0.text(mean + level*std,np.max(n),r'$'+str(level)+r'\sigma$',rotation = -90,va = 'top')

        x_fit = np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1], 250)

        ax0.plot(x_fit, gauss_1d(x_fit,*popt),label = 'Gaussian Fit',color = 'red')

        ax0.ticklabel_format(axis='y', style='sci',scilimits = (-2,0))
        ax0.yaxis.major.formatter._useMathText = True

        ax0.set_xlabel('Pseudo-Counts')
        ax0.set_ylabel('Normalised Probability')

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

        flux  = count_ul / syntax['exp_time']

        mag_level = -2.5*np.log10(flux)

        # =============================================================================
        # We now have an upper and lower estimate of the the limiting magnitude
        # =============================================================================

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

        inject_source_mag = mag2image(mag_level)

        # Number of sources
        source_no = syntax['inject_source_cutoff_sources']
        random_sources = PointsInCircum(2*syntax['fwhm'],image.shape,n=source_no)
        xran = [abs(i[0]) for i in random_sources]
        yran = [abs(i[1]) for i in random_sources]

        # =============================================================================
        # Inject sources
        # =============================================================================
        try:
            if syntax['inject_source_random']:

                for i in range(0,len(random_sources)):

                    fake_source_i = input_model(xran[i], yran[i],inject_source_mag)

                    if syntax['inject_source_add_noise']:

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
                                alpha = 0.5
                                )
                    ax2.scatter([],[],
                                marker = 'o',
                                facecolors='none',
                                edgecolors='r',
                                alpha = 0.5,
                                label = 'Injected Source')

            if syntax['inject_source_on_target']:

                fake_source_on_target = input_model(image.shape[1]/2,image.shape[0]/2,inject_source_mag)

                if syntax['inject_source_add_noise']:
                    
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
            ax2.set_title(' Fake [%s] Sources ' % model_label)

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

        limiting_mag_figure.legend(by_label.values(), by_label.keys(),
                   bbox_to_anchor=(0.5, 0.9), loc='lower center',
                   ncol = 2,
                   frameon=False)

        limiting_mag_figure.savefig(syntax['write_dir']+'limiting_mag_porb.pdf',
                                        # box_extra_artists=([l]),
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


    syntax['maglim_mean'] = mean
    syntax['maglim_std'] = std

    return mag_level,syntax



def inject_sources(syntax,
                   image,
                   model = None,
                   r_table = None,
                   plot_steps = False,
                   save_cutouts = False,
                   lmag_guess= None):
    '''
    
    Artificial source injection within a given image to determine the maximum magnitude a source may have and not be detected to within a given signal to noise ratio (SNR)
    
    :param syntax: Option Dictionary for autophot
    :type syntax: dict
    :Required syntax keywords for this package:
        
        - c_counts/r_counts (float): Check to make sure PSF successfully created. These param gives the number of counts under a normalised PSF model as well as within the residual table
        - exp_time (float): Exposure time in seconds
        - fwhm (float): Full width half maximum of image in pixels
        - use_moffat (boolean): Boolean on whether to use moffat (True) or Gaussian (False) in PSF model, default to False.
        - image_params (dict): this dictionary describes the analytical model usinged in the PSF model. If a gaussian function is used this dictionary should contain the key "sigma" and the corresponding value. If a moffat function is function this dictionary should contained 'alpha' and 'beta' values'. These values are found and updated the syntax in find.fwhm. 
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
    :param lmag_guess: Initial guess at limiting magnitude. If None use value from 'inject_source_mag' in syntax dictionary, defaults to None
    :type lmag_guess: Float, optional
    :raises Exception: DESCRIPTION
    :return: Returns Value for injected limiting magnitude and updated syntax file
    :rtype: Tuple of limiting magnitude (float) and syntax file (dictionary)

    '''

    import os
    import logging
    import numpy as np
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    from autophot.packages import psf
    from autophot.packages.functions import find_mag
    from autophot.packages.uncertain import SNR
    from autophot.packages.functions import set_size
    from autophot.packages.uncertain import sigma_mag_err
    
    from photutils.datasets import make_noise_image
    from photutils.datasets.make import apply_poisson_noise
    
    # Get location of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))

    logger = logging.getLogger(__name__)
    
    try:
        if (r_table is  None):
        # Check if the PSF model is available
            raise Exception('PSF MODEL not available - using Gaussian function')
        
        if syntax['c_counts']:
            pass
        
            # model_label = 'PSF'
        
        def mag2image(m):
            '''
            Convert magnitude to height of PSF
            '''
            Amplitude  = (syntax['exp_time']/(syntax['c_counts']+syntax['r_counts']))*(10**(m/-2.5))
    
            return Amplitude


        # PSF model that matches close-up shape around target
        def input_model(x,y,flux):
            return model(x, y,0,flux,r_table, pad_shape = image.shape)
            
    except Exception as e:
        
        #if PSF model isn't available - use Gaussian instead

        logger.info('PSF model not available - Using Gaussian:' + e)
        # model_label = 'Gaussian'
    
        sigma = syntax['fwhm'] / 2*np.sqrt(2*np.log(2))
    
        def mag2image(m):
            
            '''
            Convert instrumental magnitude of to the amplitude of analytical model
            
            :param m: Instrumental maghitude
            :type m: float
            :return: Amplitude of source in counts
            :rtype: float

            '''
            #  Volumne/counts under 2d gaussian for a magnitude m
            volume =  (10**(m/-2.5)) * syntax['exp_time']
    
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
    
            x = np.arange(0,image.shape[0])
            xx,yy= np.meshgrid(x,x)
    
            from autophot.packages.functions import gauss_2d,moffat_2d
    
            if syntax['use_moffat']:
                model = moffat_2d((xx,yy),x,y,0,A,syntax['image_params']).reshape(image.shape)
    
            else:
                model = gauss_2d((xx,yy),x,y,0,A,syntax['image_params']).reshape(image.shape)
    
            return model
    
    print('\nPerforming full magnitude recovery test')
    
    # Dictionaries to track sources 
    inserted_magnitude = {}
    recovered_magnitude = {}
    recovered_magnitude_e = {}
    recovered_fwhm = {}
    recovered_fwhm_e = {}
    recovered_counts = {}
    recovered_counts_e = {}
    recovered_sources_list  = {}
    recovered_pos = {}
    recovered_SNR={}
    
    if (lmag_guess is None):
        # Start with this magnitude as an initiall guess
        user_mag_level = syntax['inject_source_mag'] - syntax['zp']
    else:
        user_mag_level = lmag_guess - syntax['zp']
            
    start_mag = user_mag_level
    
    # List containing detections criteria
    recovered_criteria = {}
    
    # User defined detection limit
    lim_err =  sigma_mag_err(syntax['lim_SNR'])
    
    # Number of steps - this should be quite large to avoid timeing out
    nsteps = syntax['inject_source_recover_nsteps']
    
    # Large magnitude step
    dmag   = syntax['inject_source_recover_dmag']
    
    # Finer magnitude magnitude step
    fine_dmag = syntax['inject_source_recover_fine_dmag']
    
    # If random possion noise is added, how many time is this repeated
    redo   = syntax['inject_source_recover_dmag_redo']
    
    # Number of sources
    source_no = syntax['inject_source_cutoff_sources']
    
    # Percentage of sources needed to be 'lost' to define the limiting magnitude
    detection_cutout = syntax['inject_source_cutoff_limit']
    
    if detection_cutout > 1:
        detection_cutout = 0.8
        print('Detection limit cannot be > 1 - setting to 0.8')
        
    # No need to inject sources multiple time if not adding poission noise
    if not syntax['inject_source_add_noise']:
        redo = 1
    
    # Backstop criteria - this will check that the limiting magnitude is consistent beyond "iter_stop_limit" numer of steps
    iter_stop_limit = 5
    iter_stop = 0

    '''
    TODO Check what values for distance we can use here!
    '''
    
    # Choose locations to inject sources
    random_sources = PointsInCircum(2*syntax['fwhm'],image.shape,n=source_no)
    xran = [abs(i[0]) for i in random_sources]
    yran = [abs(i[1]) for i in random_sources]
    
    injection_df = pd.DataFrame([list(xran),list(yran)])
    injection_df = injection_df.transpose()
    injection_df.columns = ['x_pix','y_pix']
    injection_df.reset_index(inplace = True,drop = True)
    
    print('Starting Magnitude: %.3f' % (start_mag + syntax['zp']))
    
    # Measure flux at each point proir than adding any fake sources -to account for a irregular SNR at a specific location
    psf_fit,_ = psf.fit(image ,
                        injection_df,
                        r_table,
                        syntax,
                        # syntax['fwhm'],
                        return_fwhm = True,
                        no_print = True,
                        hold_pos = False,
                        cutout_base = False,
                        save_plot = False
                        )
                
    psf_params,_ = psf.do(psf_fit,
                          r_table,
                          syntax,
                          syntax['fwhm'])
    
    psf_counts = psf_params['psf_counts'].values/syntax['exp_time']
    psf_counts_err = psf_params['psf_counts_err'].values/syntax['exp_time']
    bkg_counts = psf_params['bkg'].values/syntax['exp_time']

    SNR_source = SNR(psf_counts,bkg_counts,syntax['exp_time'],0,syntax['ap_size']*syntax['fwhm'],syntax['gain'],0)
    
    SNR_source = np.array([round(num, 1) for num in SNR_source])
    
    # if SNR at a position is greater than detection limit - set the SNR limit at this position to this (higher) limit.
    SNR_source[(SNR_source <= syntax['lim_SNR']) | (np.isnan(SNR_source))] = syntax['lim_SNR']
    
    injection_df['limit_SNR'] = SNR_source
    
    # Begin each list for each source - used in plotting
    for k in range(len(injection_df)):
        
        inserted_magnitude [k] = []
        recovered_magnitude[k] = []
        recovered_magnitude_e[k] = []
        recovered_fwhm[k] = []
        recovered_fwhm_e[k] = []
        recovered_counts[k] = []
        recovered_counts_e[k] = []
        recovered_sources_list[k]  = []
        recovered_SNR[k]=[]
        recovered_pos[k] = []
        
        
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

    # dmag fine scale - inital parameters
    use_dmag_fine = False
    fine_dmag_range = None
    
    # Inital counter
    ith = 0 

    while True:
        
        if use_dmag_fine:
            
             dmag_step = gradient*fine_dmag_range[ith]
            
        else:

            dmag_step =  gradient*dmag_range[ith]
            
        recoverd_sources_dmag = []
        
        # Sources are put in one at a time to avoid contamination
        for k in range(len(injection_df)):
            
            SNR_no_source = injection_df['limit_SNR'].values[k]
            
            for j in range(redo):
        
                print('\rStep: %d / %d :: Source %d / %d :: Iteration  %d / %d :: Mag %.3f' % (ith+1,nsteps,k+1,len(injection_df),j+1,redo,start_mag+dmag_step+syntax['zp']),
                       end = '',
                       flush = True)
        
                fake_source_on_target = input_model(injection_df['x_pix'].values[k],
                                                    injection_df['y_pix'].values[k],
                                                    mag2image(start_mag+dmag_step)
                                                    )
                
                

                if syntax['inject_source_add_noise']:
                    
                    # add ranomd possion noise to artifical star
                    nan_idx = np.isnan(fake_source_on_target)
                    neg_idx = fake_source_on_target < 0
        
                    fake_source_on_target[nan_idx] = 0
                    fake_source_on_target[neg_idx] = 0
        
                    fake_source_on_target = make_noise_image(fake_source_on_target.shape,
                                                    distribution = 'poisson',
                                                    mean = fake_source_on_target,
                                                    seed = np.random.randint(0,1e3))
        
                    fake_source_on_target = apply_poisson_noise(fake_source_on_target)
                    
                psf_fit,_ = psf.fit(image + fake_source_on_target,
                                    injection_df.iloc[[k]],
                                    r_table,
                                    syntax,
                                    # syntax['fwhm'],
                                    return_fwhm = True,
                                    no_print = True,
                                    hold_pos = False,
                                    cutout_base = False,
                                    save_plot = False
                                    )

                psf_params,_ = psf.do(psf_fit,
                                      r_table,
                                      syntax,
                                      syntax['fwhm'])
                
                psf_counts = psf_params['psf_counts'].values[0]/syntax['exp_time']
                psf_counts_err = psf_params['psf_counts_err'].values[0]/syntax['exp_time']
                bkg_counts = psf_params['bkg'].values[0]/syntax['exp_time']

                SNR_source = SNR(psf_counts,bkg_counts,syntax['exp_time'],0,syntax['ap_size']*syntax['fwhm'],syntax['gain'],0)

                recovered_counts[k].append(psf_params['psf_counts'].values[0])
                recovered_counts_e[k].append(psf_params['psf_counts_err'].values[0])
                
                
                mag_recovered =  find_mag(psf_counts,0)
                mag_recovered_error = find_mag(psf_counts,0) - find_mag(psf_counts+psf_counts_err,0)
                
                recovered_pos[k].append((psf_params['x_pix'],psf_params['y_pix']))
                 
                 

                inserted_magnitude[k].append(start_mag+dmag_step)
                recovered_magnitude[k].append(mag_recovered[0])
                recovered_magnitude_e[k].append(mag_recovered_error[0])
                recovered_fwhm[k].append(psf_params['target_fwhm'].values[0])
                
                recovered_SNR[k].append(SNR_source)
                
                if plot_steps:
                    
                    fig = plt.figure(figsize = set_size(250,1))
                    
                    ax1 = fig.add_subplot(111)

                    
                    ax1.imshow(fake_source_on_target,origin = 'lower')
                    ax1.set_xlabel('X Pixel')
                    ax1.set_ylabel('Y Pixel')
                    
                    if SNR_source<syntax['lim_SNR']:
                        text = r'Source lost'
                        color = 'red' 
                    else:
                        text = r'Source recovered'
                        color = 'green'
                    
                    ax1.annotate(text,
                        xy=(0.9, 0.9),
                        xycoords='axes fraction',
                        # xytext=(0.2, 0.5),
                        va = 'top',
                        ha = 'right',
                        color = color,
                        bbox=dict(
                                  fc="white",
                                  ec="black",
                                  lw=2)
                            )
                    
                    ax1.set_title(r'Recovered SNR: %.1f$\sigma$' % SNR_source)
                    
                    gif_images_path = os.path.join(syntax['write_dir'],'limiting_gif')
                    
                    os.makedirs(gif_images_path, exist_ok=True)
                    
                    
                    plt.savefig(os.path.join(gif_images_path,'img_%d.jpg') % image_number,bbox_inches = 'tight')
                    plt.close(fig)
                    
                    image_number+=1
                    
                    

            # Sources greater than limit
            recoverd_sources_dmag.append([SNR_source>syntax['lim_SNR'] or SNR_source>SNR_no_source][0])
            
            inserted_mags_k = inserted_magnitude[k][int(ith*redo):int((ith*redo)+redo)]
            recovered_mags_k = recovered_magnitude[k][int(ith*redo):int((ith*redo)+redo)]
            
            recovered_sources = np.sum([abs(inserted_mags_k[i] - recovered_mags_k[i]) < lim_err for i in range(len(inserted_mags_k))])/len(inserted_mags_k)
            recovered_SNR_k = recovered_SNR[k][int(ith*redo):int((ith*redo)+redo)]
            
            recovered_sources = np.sum([recovered_SNR_k[i] > 3 or recovered_SNR_k[i] > SNR_no_source for i in range(len(recovered_SNR_k))])/len(recovered_SNR_k)
                
            recovered_sources_list[k]+=[100*recovered_sources]*redo
            
        # For the sources injects - did they pass the recovered test?
        recover_test = [np.sum(recoverd_sources_dmag)/len(recoverd_sources_dmag) > 1-detection_cutout ][0]
        
        # name each iteration of inject sources with a given magnitude
        step_name = round(start_mag+dmag_step+syntax['zp'],3)
        
        recovered_criteria[step_name] = recover_test
        
        # If the first source inject comes back negative on the first interation - flip the dmag steps i.e. fainter -> brighter
        if recovered_criteria[step_name] == citeria and ith == 0 and not use_dmag_fine:
        
            print('\nInitial injection sources not recovered - injecting brighter sources')
            
            gradient *= -1
            
            citeria = not citeria 
        
        # else if the detection meetings the criteria i.e. sources are (not) detected 
        elif recovered_criteria[step_name] == citeria:
            
            
            if iter_stop == 0 and not use_dmag_fine:
                print('\nApproximate limiting magnitude: %.3f' % (syntax['zp']+start_mag+dmag_step))
                
                use_dmag_fine = True
                
                # dmag_step_old = gradient * dmag_range[ith]
                fine_nsteps = int(1/fine_dmag)
                
                # reset start magnitude
                start_mag = start_mag+dmag_step
                
                fine_dmag_range =  np.linspace(0,fine_dmag*fine_nsteps,int(fine_nsteps))
                
                print('\nAproximate limting magnitude found -  Using finer scale')
                
                gradient *= -1
                
                citeria = not citeria 
                
                # Restart the count
                ith = 0
                
            else:
                
                inject_lmag = start_mag + dmag_step
                inject_lmag_minus_1 = start_mag + gradient*fine_dmag_range[ith-1]
            
            iter_stop+=1
            
        else:
            
            iter_stop = 0
            
        if iter_stop > iter_stop_limit:
            
                break
        else:
            
            ith = ith + 1

    injection_params = {'inject_mag': inserted_magnitude,
                        'recover_mag': recovered_magnitude,
                        'recover_mag_e': recovered_magnitude_e,
                        'recover_fwhm':recovered_fwhm,
                        'recover_fwhm_e': recovered_fwhm_e,
                        'recovered_SNR':recovered_SNR
                        }
    
    df_dict = {}
    df_dict['source'] = []
    
    for param_key, param_values in injection_params.items():
        df_dict[param_key] = []
        
    
    for k in range(len(injection_df)):
        
        for param_key, param_values in injection_params.items():
            
            if len(param_values[k])==0:
                df_dict[param_key] += [np.nan] * len(inserted_magnitude[k])
            else:
                df_dict[param_key]+=param_values[k]
                
        df_dict['source'] += [k] * len(inserted_magnitude[k])

    recover_df = pd.DataFrame(df_dict)

    image_minus_1 = image.copy()
    image_limited = image.copy()
    
    if save_cutouts:
        
        cutout_size = 2 * syntax['fwhm']
        
        rows = int(np.ceil(len(injection_df)/2))
        
        if rows % 2 != 0:
            rows=rows+1
        
        fig, axes = plt.subplots(rows,2, sharex=True)
        
        axes = axes.flatten()
        fig = plt.figure(figsize = set_size(250,1))
        
        gs = gridspec.GridSpec(rows, 2)
        
        gs.update(wspace=0.025, hspace=0.05)
        
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
            
            circle = plt.Circle((kth_cutout.shape[1]/2,kth_cutout.shape[1]/2), 1.5*syntax['fwhm'], 
                            color='r',
                            lw = 0.25,
                            fill=False)
            ax.add_patch(circle)
            
        fig.savefig(syntax['write_dir']+'Inject_lmag_cutouts_'+str(syntax['base'].split('.')[0])+'.pdf',
                    format = 'pdf')
            
        plt.close(fig)

    # Plot the results
    fig = plt.figure(figsize = set_size(250,1))
    layout = gridspec.GridSpec(ncols=2, nrows=3, figure=fig,
                               hspace = 0.5)
    ax1 = fig.add_subplot(layout[0, :])
    ax2 = fig.add_subplot(layout[1, :], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax3 = fig.add_subplot(layout[2, 0])
    ax4 = fig.add_subplot(layout[2, 1])
    
    import itertools
    
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    
    for k in range(len(injection_df)):
        markers, caps, bars = ax1.errorbar(inserted_magnitude[k]+syntax['zp'],recovered_magnitude[k]+syntax['zp'],
                      yerr = [np.sqrt(i**2 + syntax['zp_err']**2) for i in recovered_magnitude_e[k]],
                      ls = '',
                      marker = next(marker),
                      ecolor = 'black',
                      markerfacecolor = 'red',
                      edgecolor = None,
                      label = 'Recovered Magnitude')

    ax1.plot(inserted_magnitude[k]+syntax['zp'],inserted_magnitude[k]+syntax['zp'],
              ls = '--',
              color = 'red',
              alpha = 0.5,
              label = 'True Magnitude')
    
    ax1.fill_between(inserted_magnitude[k]+syntax['zp'], 
                     inserted_magnitude[k]+syntax['zp']-lim_err,
                     inserted_magnitude[k]+syntax['zp']+lim_err,
                     # ls = '--',
                     color = 'red',
                     alpha = 0.5,
                     label = r'%d$\sigma$' % syntax['lim_SNR'])
    
    [bar.set_alpha(0.25) for bar in bars]
    [cap.set_alpha(0.25) for cap in caps]
        
    for k in range(len(injection_df)):
        markers, caps, bars = ax2.errorbar(inserted_magnitude[k]+syntax['zp'],
                                           recovered_SNR[k],
                                          # yerr = recovered_fwhm_e,
                                           ls = '',
                                           marker = 'o',
                                           ecolor = 'black',
                                           color = 'green',
                                           label = r'Recovered FWHM')
    
    [bar.set_alpha(0.25) for bar in bars]
    [cap.set_alpha(0.25) for cap in caps]

    ax2.axhline(3,
                color = 'green',
                ls = '--',
                label = r'3\\sigma')
    
    ax2.axvline(inject_lmag+syntax['zp'],
                color = 'blue',
                ls = '--',
                label = r'Detection Limit')
    

    for k in range(len(injection_df)):
        fake_source_on_target = input_model(injection_df['x_pix'].values[k],
                                            injection_df['y_pix'].values[k],
                                            mag2image(inject_lmag_minus_1))
        
        image_minus_1+=fake_source_on_target
        
    ax3.imshow(image_minus_1,interpolation = None)
    ax3.set_title(r'%.3f mag' % (inject_lmag_minus_1+syntax['zp']))


    for k in range(len(injection_df)):
        fake_source_on_target = input_model(injection_df['x_pix'].values[k],
                                            injection_df['y_pix'].values[k],
                                            mag2image(inject_lmag))
        
        image_limited+=fake_source_on_target
        
    ax4.imshow(image_limited,interpolation = None)
    ax4.set_title(r'%.3f mag' % (inject_lmag+syntax['zp']))
    
    for k in range(len(injection_df)):
        
        circle = plt.Circle((injection_df['x_pix'].values[k],injection_df['y_pix'].values[k]), syntax['fwhm'], 
                            color='r',
                            lw = 0.25,
                            fill=False)
        ax3.add_patch(circle)
        
    for k in range(len(injection_df)):
        
        circle = plt.Circle((injection_df['x_pix'].values[k],injection_df['y_pix'].values[k]), syntax['fwhm'], 
                            color='r',
                            lw = 0.25,
                            fill=False)
        ax4.add_patch(circle)
        
    for ax in [ax3,ax4]:
        for k in range(len(injection_df)):
            x = [i[0] for i in recovered_pos[k]]
            y = [i[1] for i in recovered_pos[k]]
            ax.scatter(x,y,marker = '+',color = 'red')
    
    ax2.set_ylim(0,10)

    ax1.set_ylabel(r'$M_{Recovered}$')
    ax2.set_ylabel(r'Signal to Noise Ratio')
    ax2.set_xlabel(r'$M_{Injected}$')
    
    box = ax1.get_position()
    box.y0 = box.y0 - 0.08
    box.y1 = box.y1 - 0.08
    ax1.set_position(box)

    ax1.label_outer() 
    
    fig.savefig(syntax['write_dir']+'Inject_lmag_'+str(syntax['base'].split('.')[0])+'.pdf',bbox_inches = 'tight',format = 'pdf')
    recover_df.round(3).to_csv(syntax['write_dir']+'Inject_lmag_'+str(syntax['base'].split('.')[0])+'.csv')
    
    plt.close(fig)
    
    return inject_lmag,syntax
    
    
    
    
