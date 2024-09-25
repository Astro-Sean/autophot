#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:58:29 2022

@author: seanbrennan
"""


def set_size(width,aspect=1,fraction = 1):
    '''
     Function to generate size of figures produced by AutoPhot. To specify the dimensions of a figure in matplotlib we use the figsize argument. However, the figsize argument takes inputs in inches and we have the width of our document in pts. To set the figure size we construct a function to convert from pts to inches and to determine an aesthetic figure height using the golden ratio. The golden ratio is given by:

     .. math ::
         
        \\phi = (5^{0.5} + 1) / 2 \\approx 1.618

    The ratio of the given width and height is set to the golden ratio


    Credit: `jwalton.info <https://jwalton.info/Embed-Publication-Matplotlib-Latex/>`_

    :param width: Width of figure in pts. 1pt == 1/72 inches
    :type width: float
    :param aspect: Aspect of image i.e. :math:`height  = width / \\phi \\times \mathit{aspect}`, default  = 1
    :type aspect: float
    :return: Returns tuple of width, height in inches ready for use.
    :rtype: Tuple

    '''


    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 + 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / golden_ratio

    fig_dim = (fig_width_in, fig_height_in * aspect)

    return fig_dim


def is_undersampled(fwhm, pixel_scale):
    """
    Determine if an image is undersampled using the Nyquist sampling theory.

    Parameters:
    fwhm (float): Full Width at Half Maximum (FWHM) of the PSF in arcseconds.
    pixel_scale (float): Pixel scale of the image sensor in arcseconds per pixel.

    Returns:
    bool: True if the image is undersampled, False otherwise.
    """
    nyquist_limit = 2 * pixel_scale
    return fwhm > nyquist_limit



def convert_to_mjd_astropy(date_string):
    
    from astropy.time import Time

    try:
        # Try parsing with 'T' separator
        t = Time(date_string, format='isot', scale='utc')
    except ValueError:
        # If parsing fails, try without 'T' separator
        t = Time(date_string, format='fits', scale='utc')

    # Access the Modified Julian Date (MJD)
    mjd = t.mjd

    return mjd



def convolve_to_match_FWHM(image1, FWHM1, FWHM2, output_FWHM=None):
    
    
    import numpy as np
    from astropy.convolution import Gaussian2DKernel,Box2DKernel, convolve,Tophat2DKernel

    """
    Convolve image1 with a Gaussian kernel to match FWHM2.

    Parameters:
    -----------
    image1 : array_like
        Input image to be convolved.

    FWHM1 : float
        Full Width at Half Maximum (FWHM) of image1.

    FWHM2 : float
        Target FWHM to match.

    output_sigma : float or None, optional
        If specified, the sigma value to use for the output Gaussian kernel.
        If None (default), it will be adjusted to match FWHM2.

    Returns:
    --------
    convolved_image : array_like
        Convolved image with adjusted FWHM.
        
    """
    # Calculate sigmas
    sigma1 = FWHM1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = FWHM2 / (2 * np.sqrt(2 * np.log(2)))

    # Difference in sigmas
    delta_sigma = sigma2 - sigma1
    
    # delta_sigma = np.sqrt((FWHM1 /2.355)**2 - (FWHM2 /2.355)**2)

    if delta_sigma > 0:
        print(f'Convolving science image  [fwhm: {FWHM1:.1f}] to match template [fwhm: {FWHM2:.1f}]')
        # Adjust sigma for Image 1
        if output_FWHM is None:
            sigma1_adjusted = sigma1 + delta_sigma
        else:
            output_sigma = output_FWHM / (2 * np.sqrt(2 * np.log(2)))
            sigma1_adjusted = output_sigma
            
        sigma1_adjusted = int(sigma1_adjusted)
        if sigma1_adjusted % 2 == 0: sigma1_adjusted+=1
        # Create Gaussian kernel
        kernel1 = Gaussian2DKernel(sigma1_adjusted)
        # Convolve Image 1 with the Gaussian kernel
        convolved_image = convolve(image1, kernel1)
    else:
        # No need to adjust, FWHM1 is already greater than or equal to FWHM2
        convolved_image = image1

    return convolved_image

def get_image_stats(image, sigma=3, maxiters=None):
    
    import numpy as np
    from astropy.stats import sigma_clipped_stats, mad_std

    # Perform sigma clipping and calculate mean, median, and MAD in one step
    mean_value, median_value, std_value = sigma_clipped_stats(
                                            image,
                                            sigma=sigma,
                                            maxiters=maxiters,
                                            # background=sigma,
                                            cenfunc=np.nanmedian,  # Use nanmedian for the center function
                                            stdfunc=mad_std  # Use mad_std for the standard deviation function
                                        )

        

    return mean_value, median_value, std_value



def calculate_bins(x):
    import numpy as np
    """
    Calculate the number of bins for a histogram using the Freedman-Diaconis rule.

    Parameters:
    x (array-like): Input data array or list of data values.

    Returns:
    int: Number of bins to use for the histogram.
    
    The Freedman-Diaconis rule is used to determine an optimal number of bins
    by considering the interquartile range (IQR) and the number of data points.
    """
    # Compute the 25th and 75th percentiles of the data
    q25, q75 = np.nanpercentile(x, [25, 75])
    
    # Calculate the interquartile range (IQR)
    iqr = q75 - q25
    
    # Calculate the bin width using the Freedman-Diaconis rule
    bin_width = 2 * iqr * len(x) ** (-1/3)
    
    # Determine the number of bins
    data_range = np.nanmax(x) - np.nanmin(x)
    bins = round(data_range / bin_width)
    
    return bins



def save_to_fits(data, output_filename):
    try:
        from astropy.io import fits
        
        
        # Create a PrimaryHDU object with the data
        hdu = fits.PrimaryHDU(data)
    
        # Create an HDU list and append the PrimaryHDU
        hdulist = fits.HDUList([hdu])
    
        # Write the HDU list to a FITS file
        hdulist.writeto(output_filename,
                     overwrite = True,
                     output_verify = 'silentfix+ignore')
        
    except Exception as e:    
        import os,sys
       
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        return 0,0
            
        
    
    return None


def nearest_inf_distance(arr, x, y):
    import numpy as np
    
    try:
    
        if not (0 <= x < arr.shape[0]) or not (0 <= y < arr.shape[1]):
            raise ValueError("Invalid coordinates (x, y)")
    
        # if np.isinf(arr[x, y]):
        #     return 0, 0
    
        min_x_distance = np.inf
        min_y_distance = np.inf
    
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isinf(arr[i, j]):
                    # print('nan at ',i,j)
                    min_x_distance = min(min_x_distance, abs(i - x))
                    min_y_distance = min(min_y_distance, abs(j - y))
    except Exception as e:    
        import os,sys
       
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        return 0,0
        

    return min_x_distance if min_x_distance != np.inf else None, min_y_distance if min_y_distance != np.inf else None



def nearest_zero_distance(image, x, y):
    
    import numpy as np
    if not image or x < 0 or y < 0 or y >= len(image) or x >= len(image[0]):
        return None

    rows, cols = len(image), len(image[0])
    # Initialize arrays to store distances to the nearest zeroed column and row
    min_dist_row = np.full(rows, np.inf)
    min_dist_col = np.full(cols, np.inf)

    # Calculate distances to the nearest zeroed row
    for i in range(rows):
        if all(val == 0 for val in image[i]):
            min_dist_row[i] = abs(i - y)

    # Calculate distances to the nearest zeroed column
    for j in range(cols):
        if all(image[i][j] == 0 for i in range(rows)):
            min_dist_col[j] = abs(j - x)

    # Find the minimum distances to the nearest zeroed row and column for the given (x, y) coordinate
    min_distance_row = min(min_dist_row)
    min_distance_col = min(min_dist_col)

    return min_distance_row, min_distance_col


def get_distance_modulus(redshift,H0 = 70, omega = 0.3):
    
    from astropy.cosmology import FlatLambdaCDM
    import numpy as np
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega)
    d = cosmo.luminosity_distance(redshift).value * 1e6
    dm = 5 * np.log10(d / 10)

    return dm





def apply_mask_image(image_path, mask_path, sigma=3.0, maxiters=5):
    
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats

    """
    Replaces masked regions in a 2D FITS image with the sigma-clipped median of the unmasked regions.
    
    Parameters:
    -----------
    image_path : str
        File path to the FITS image.
    mask_path : str
        File path to the FITS mask (should be a binary mask where masked areas are 1 and unmasked areas are 0).
    sigma : float, optional
        The number of standard deviations to use for both the lower and upper clipping limit (default is 3.0).
    maxiters : int, optional
        The maximum number of sigma-clipping iterations to perform (default is 5).
    
    Returns:
    --------
    None
        The function overwrites the FITS image with the modified data.
    """
    
    # Load the FITS image
    with fits.open(image_path, mode='update') as hdul:
        image_data = hdul[0].data  # Assuming the image is in the primary HDU
        
        # Load the mask FITS file
        with fits.open(mask_path) as mask_hdul:
            mask_data = mask_hdul[0].data
        
        # Ensure the mask is boolean (mask regions are 1, unmasked regions are 0)
        mask = mask_data.astype(bool)
        
        # Find the unmasked regions
        unmasked_values = image_data[~mask]
        
        # Use astropy's sigma_clipped_stats to compute the median with sigma clipping
        mean, median_value, stddev = sigma_clipped_stats(unmasked_values, sigma=sigma, maxiters=maxiters)
        
        # Replace the masked regions with the sigma-clipped median
        image_data[mask] = median_value
        
        # Overwrite the FITS file with the modified image data
        hdul.flush()  # This writes changes to the file

    # print(f"Masked regions in {image_path} have been replaced with the sigma-clipped median and the file is overwritten.")
    return 

def centroid_com2(data, mask=None, noise_mean=0., noise_std=0, 
                 thresh_sigma=3.):
    """
    
    Motivated by https://github.com/astropy/photutils/issues/593
    Calculate the centroid of a 2D array as its "center of mass"
    determined from image moments.

    Invalid values (e.g. NaNs or infs) in the ``data`` array are
    automatically masked.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    noise_mean : float, optional
        The averaged noise level. The simplest way to calculate it is to take
        the median or mean of the ``data`` after 3-sigma clipping with few 
        iterations. This, however, may overestimate the noise level (since the
        nearby pixels are brightened by the light source), so it is best to use
        the pixels well outside the source to estimate background level. 
        Defaults to 0.
    
    noise_std : float, optional
        The fluctuation (standard deviation) of noise (error source). It can
        also be calculated as ``noise_mean`` by using sigma clipping, but it
        is better to use pixels well apart from the source. Defaults to 0.
    
    thresh_sigma : float, optional
        The threshold above the ``noise`` to mask pixels for centroiding. Only 
        the pixels having value above ``noise_mean + thresh_sigma * noise_std`` 
        will be used for the centroiding. It is known that the 3-sigma
        criterion, i.e., ``noise + 3 * noise_sigma``, is a good threshold 
        (Ma et al. 2009, Optics Express, 17, 8525), so the recommended value is 
        ``thresh_sigma=3``. Defaults to 3.

    Returns
    -------
    centroid : `~numpy.ndarray`
        The ``x, y`` coordinates of the centroid.
    """
    import numpy as np

    from skimage.measure import moments
    from astropy.stats import sigma_clipped_stats
    

    

    data = np.ma.asanyarray(data)
        
    if noise_mean ==0 and noise_std ==0:
        
                    
        noise_mean, noise_median, noise_std = sigma_clipped_stats(data, sigma=3.0)  

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask
    
    mask_noise = data < (noise_mean + thresh_sigma * noise_std)
    data.mask |= mask_noise
    # print('threshold =', noise_mean + thresh_sigma * noise_std)
    # print('N_masked =', np.count_nonzero(data.mask))
    
    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        # warnings.warn('Input data contains input values (e.g. NaNs or infs), '
        #               'which were automatically masked.', AstropyUserWarning)

    # Convert the data to a float64 (double) `numpy.ndarray`,
    # which is required for input to `skimage.measure.moments`.
    # Masked values are set to zero.
    data = data.astype(float)
    data.fill_value = 0.
    data = data.filled()

    m = moments(data, 1)
    xcen = m[1, 0] / m[0, 0]
    ycen = m[0, 1] / m[0, 0]
    
    return np.array([xcen, ycen])







# from contextlib import contextmanager

# @contextmanager
class suppress_stdout:
    
    def __enter__(self):
        import sys,os
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys,os
        sys.stdout.close()
        sys.stdout = self._original_stdout

def beta_value(n,f_ul,sigma,noise = 0):

    '''
    False negatives function for the fraction of real sources that go undetected. This is implemented in AutoPHOT to determine whether a source,  more specifically it's flux, can be confidently assumed to arise from the source and not be associated with the background noise distribution. We describe the probability that a suspected source can be confidently assumed to be seperate from the flux due to the background by:
    
    .. math::
        
       1 - \\beta = \\beta{\\prime} = \\frac{1}{2}(1 - \mathit{erf}(\\frac{n\sigma_{bkg} - f_{source}}{\sigma_{bkg} \\sqrt{2}}))
       
    where :math:`\\beta{\\prime}` is the confident that source is not apart of the underlying noise distribution, :math:`\mathit{erf}` is the `Error function <https://mathworld.wolfram.com/Erf.html>`_, :math:`f_{source}` is the brightest pixel that can be associated with a source, and :math:`\sigma_{bkg}` is the standard deviation of the background noise distribution.
     
    In other words, :math:`100\\times (1-\\beta) \%` of the
    sources with flux :math:`f_{source}` will have flux measurements > :math:`n\sigma_{bkg}`
    
    
    Credit: `F. Masci <http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf>`_
    
    
    :param n:  Level above background to be considered a genuine detection. Typical values are :math:`n = 3`, which equations to a :math:`3\sigma` confidence level, or in other words there is a 0.135% probability that the measured flux is a spurious noise spike. 
    :type n: float
    :param sigma: Background standard deviation.
    :type sigma: float
    :param f_ul: Brightest pixel value that can be associated with a suspected source.
    :type f_ul: float
    :param noise Mean offset of flux measurement. This value should be included if the measurement of the source is not background subtracted, defaults to 0
    :type noise: float, optional
    :return: False negatives probability which describes the confidence to which a source can be consider real and not be associated with a spurious noise spike.
    :rtype: float
    '''

    from scipy.special import erf

    from numpy import sqrt

    source_flux = f_ul - noise

    if isinstance(source_flux,list):
        source_flux = [i if i>0 else 0 for i in source_flux]


    z = ((n*sigma) - (f_ul-noise))/(sqrt(2)*sigma)

    beta = 0.5 * (1 - erf(z))

    return beta




def convolve_and_remove_nans_zeros(image, psf_fpath = None , kernel_sigma = 3):
    
    import numpy as np
    from astropy.io import fits
    from astropy.convolution import convolve_fft, convolve,Gaussian2DKernel
    from astropy.convolution import interpolate_replace_nans
# result = interpolate_replace_nans(image, kernel)
    
    # image = get_image(image_fpath)
    # # header = get_header(image_fpath)
    
    try:
        
        if  psf_fpath is None:
            print('Convolving image with Gaussian Kernel')
            kernel = Gaussian2DKernel(x_stddev=kernel_sigma)
        else:
            print('Convolving image with its PSF kernel')
            kernel = get_image(psf_fpath)

        from astropy.convolution import interpolate_replace_nans
        convolved_image  = interpolate_replace_nans(image, kernel)

        
    except Exception as e:    
        import os,sys
       
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        convolved_image = image
        
    return             convolved_image


def create_masked_array(input_file,input_yaml,snr_limit = 300,name = 'MASK_'):
    
    
    try:
        import numpy as np
        import os,sys
        from astropy.io import fits
        import matplotlib.pyplot as plt
        from astropy.visualization import SqrtStretch
        from astropy.visualization.mpl_normalize import ImageNormalize
        from photutils.background import Background2D, MedianBackground
        from photutils.segmentation import detect_sources, SourceCatalog, make_2dgaussian_kernel
        from astropy.convolution import convolve
        from photutils.aperture import RectangularAperture
        
        
    
        data = get_image(input_file)
        header = get_header(input_file)
        fwhm = int(np.ceil(header['fwhm']))
        
        ap_size = input_yaml['photometry']['ap_size']*fwhm
        
        write_dir = input_yaml['write_dir']
        base = input_yaml['base']
        
        # Subtract the background
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        
        # Convolve the data
        gsize = 5*ap_size
        if int(gsize) % 2 == 0:
            gsize = int(gsize)+1
            
        threshold = 1.5 * bkg.background_rms + bkg.background_median
        kernel = make_2dgaussian_kernel(fwhm, size=int(gsize))
        convolved_data = convolve(data, kernel)
        
        # Detect sources and create a catalog
        segment_map = detect_sources(convolved_data, threshold, npixels=10)
        cat = SourceCatalog(data, segment_map, convolved_data=convolved_data, localbkg_width=int(2*(gsize + fwhm)),background=bkg.background)
        
        # Calculate SNR and select sources with SNR > 10
        tbl = cat.to_table().to_pandas()
        
        tbl['snr'] = (tbl['max_value']) / np.sqrt(tbl['local_background'])
        tbl = tbl[tbl['snr'] > snr_limit]
        
        mask = np.zeros_like(data, dtype=int)
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        ax1.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
        
        for idx, source in tbl.iterrows():
            ax1.scatter(source['xcentroid'], source['ycentroid'], marker='o', s=50, facecolors='none', edgecolors='red')
        
            center_x = (source['bbox_xmin'] + source['bbox_xmax']) / 2
            center_y = (source['bbox_ymin'] + source['bbox_ymax']) / 2
            width = source['bbox_xmax'] - source['bbox_xmin']
            height = source['bbox_ymax'] - source['bbox_ymin']
        
            rectangular_aperture = RectangularAperture(
                (center_x, center_y),
                w=width,
                h=height,
                theta=0  # Angle in degrees
            )
            rectangular_aperture.plot(ax=ax1, color='green', lw=1.5, alpha=0.5)
        
            # Create a mask using rectangular apertures
            mask_rectangular_aperture = RectangularAperture(
                (center_x, center_y),
                w=width,
                h=height,
                theta=0  # Angle in degrees
            )
            mask = mask + mask_rectangular_aperture.to_mask().to_image(shape=mask.shape).astype(int)
        
        
        ax1.set_title('Background-subtracted Data')
        
        save_loc = os.path.join(write_dir,name+base+'.pdf')
        fig.savefig(save_loc,bbox_inches='tight')
        plt.close()
        
        # mask = 1 - mask
        
        
        mask = (mask.astype(bool)) | (~np.isfinite(data))
        hdu = fits.PrimaryHDU(mask.astype(int))
        hdul = fits.HDUList([hdu])
    
        footprint_loc = os.path.join(write_dir,name+base+'.fits')
    
        hdul.writeto(footprint_loc,
                      overwrite=True,
                      output_verify = 'silentfix+ignore')

    except Exception as e:
        import os,sys
       
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)

    
    
    return mask

def border_msg(msg,body = '-',corner = '+'):
    '''
    
    Produce a print statement where the text is surrounded by a box. For example:
    
    .. code-block :: python
    
       > border_msg('Hello There')
       > +-----------+ 
         |Hello There|
         +-----------+
    
       > border_msg('General Kenobi',body = '=',corner = '!')
       > !==============!
         |General Kenobi|
         !==============!
                   
    :param msg: String message which will be printed to screen.
    :type msg: str
    :param body: String character to represent body around message, defaults to '-'
    :type body: str, optional
    :param corner: String character to represent the four corners around message, defaults to '+'
    :type corner: trs, optional
    :return: Original string with borner around it.
    :rtype: Print statement

    '''

    row = len(msg.strip()) - ((len(corner)-1)*len(corner))
    h = ''.join([corner] + [body*row] + [corner])
    result= h + '\n'"|"+msg+"|"'\n' + h
    output_str = '\n' + result + '\n'
    return output_str
    
class autophot_yaml():

    def __init__(self,filepath = None,dict_name = None,wdir = None):

         self.filepath  = filepath
         self.dict_name = dict_name
         self.wdir = wdir

    def load(self):

        import yaml
        import os

        if self.wdir != None:
            file_path = os.path.join(self.wdir, self.filepath )
        else:
            file_path = self.filepath


        with open(file_path, 'r') as stream:
            var = yaml.load(stream, Loader=yaml.FullLoader)

        if self.dict_name != None:
            data = var[self.dict_name]
        else:
            data = var

        return data

    def update(self,tele,inst_key,inst,key,new_val):

        import yaml
        import copy

        doc = {key:new_val}

        with open(self.filepath,'r') as yamlfile:


            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml_backup = copy.deepcopy(cur_yaml)

            try:

                cur_yaml[tele][inst_key][inst].update(doc)
            except:
                cur_yaml = cur_yaml_backup


        with open(self.filepath,'w+') as yamlfile:

            yaml.safe_dump(cur_yaml, yamlfile,default_flow_style=False)



    def create(fname,data):
        
        import yaml
        import os

        target_name = fname
        
        if '.yml' not in fname:
            fname+='.yml'

        data_new  = {os.path.basename(target_name.replace('.yml','')):data}
        with open(fname, 'w') as outfile:
            yaml.dump(data_new, outfile, default_flow_style=False)



def get_header(fpath):

    """
    
    Robust function to get header from :math:`FITS` image for use in AutoPHOT. Due to a :math:`FITS` image typically having multiple headers, which may have useful important information spread across multiples heres, this function returns (if appropriate) a single header file containing all needed header files. This function aims to find correct telescope header info based on "Telescop" header key.

    :param fpath: file of fits image which contains a header file
    :type fpath: str
    :return: header information
    :rtype: header object
    
    """
    
    from astropy.io.fits import getheader
    from astropy.io import fits

    try:
        try:
            with fits.open(fpath,ignore_missing_end=True) as hdul:
                # Need to verify/fix or can get errors down the line
                hdul.verify('silentfix+ignore')
                headinfo = hdul
            try:
                # try for Telescop keyword
                headinfo['Telescop']
            except:
                raise Exception
        except:
            # try to find 'sci' extension
            headinfo = getheader(fpath,'sci')
            raise Exception
    except:
        with fits.open(fpath) as hdul:
            hdul.verify('silentfix+ignore')
            headinfo = hdul

    try:
        # If header is a list
        if isinstance(headinfo,list):
            # If list length contains multiple headers, concat them
            if len(headinfo)>1:
                headinfo_list = headinfo[0].header
                for i in range(1,len(headinfo)):
                    headinfo_list.update(headinfo[i].header)
                    
                headinfo = headinfo_list
            else:
                # is length of list choose this a header
                headinfo = headinfo[0].header
    except Exception as e:
        print(e)

    return headinfo



def get_image(fpath):
    
    '''
    For a given :math:`\mathit{FITS}` file, search through header a look for 2D image using the ":math:`\mathit{sci}`" attribute. A error is raise if the image found is not a 2D array e.g. if a :math:`\mathit{FITS\ cube}` is given,
    
    :param fpath: File path towards :math:`\mathit{FITS}` file.
    :type fpath: str
    :return: 2D image
    :rtype: array

    '''

    import os
    from astropy.io import fits
    try:

        try:
            image = fits.getdata(fpath,'sci')
            
            
            if image.shape is None: raise Exception
        except Exception as e:
            # print(e)
            image = fits.getdata(fpath)
            if image.shape is None: raise Exception

    except Exception as e:
        # print(e)
        with fits.open(fpath,ignore_missing_end=True) as hdul:
            hdul.verify('silentfix+ignore')
            image = hdul[0].data
            
            if image.shape is None: 
            
                image = hdul[1].data
            
            

    # print(image)
    if len(image.shape) != 2:

        base=os.path.basename(fpath)

        raise Exception('Warning:: %s not 2-D array' % base)

    return image



def find_center_without_nan(fpath):
    import numpy as np
    
        
    data = get_image(fpath)
    # Find the first and last non-NaN rows
    non_nan_rows = np.where(~np.all(np.isnan(data), axis=1))[0]
    first_row = non_nan_rows[0]
    last_row = non_nan_rows[-1]

    # Find the first and last non-NaN columns
    non_nan_columns = np.where(~np.all(np.isnan(data), axis=0))[0]
    first_column = non_nan_columns[0]
    last_column = non_nan_columns[-1]

    # Calculate the center based on the bounds
    center_x = (first_column + last_column) / 2
    center_y = (first_row + last_row) / 2

    return center_x, center_y

# Example usage
def remove_nan_rows_columns(fpath,new_center_x,new_center_y):
    
    
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    # Read the FITS file
    
    data = get_image(fpath)
    header = get_header(fpath)
    
    
    data[np.abs(data) < 1.1e-30] = np.nan

    # Find NaN rows and columns
    nan_rows = np.all(np.isnan(data), axis=1)
    nan_columns = np.all(np.isnan(data), axis=0)
    

    
    # Trim data and update header
    data_new = data[~nan_rows][:, ~nan_columns]
    header['NAXIS1'] = data_new.shape[1]
    header['NAXIS2'] = data_new.shape[0]

    # Calculate the new image center
 
    # Create a new WCS object using the updated header
    wcs = WCS(header)
    # wcs.wcs.crpix = [new_center_x, new_center_y]
    # print(wcs)

    # Create a Cutout2D object using the new center and updated WCS
    cutout = Cutout2D(data, (new_center_x, new_center_y),
                      (data_new.shape[1], data_new.shape[0]), wcs=wcs)
    


    # Save the updated FITS file
    # Save the updated FITS file
    image = cutout.data
    header.update(cutout.wcs.to_header())

    
    
    fits.writeto(fpath,
                  image,
                  header,
                  overwrite = True,
                  output_verify = 'silentfix+ignore')
   
   
   
    return image,header

def concatenate_csv_files(folder_path, output_filename,loc_file = 'output.csv'):
    
    
    import os
    import pandas as pd
    
    print(border_msg(fr'Searching for {loc_file} files in {folder_path}...'))
    # Initialize an empty DataFrame to hold the concatenated data
    concatenated_data = []
    

    # Traverse the folder using os.walk
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == loc_file :

                file_path = os.path.join(root, file)
                # print(file_path)
                # Read the content of each output.csv file into a DataFrame
                df = pd.read_csv(file_path)
                # Concatenate the data to the main DataFrame
                
                concatenated_data.append(df)
    
    concatenated_data = pd.concat(concatenated_data, ignore_index=True)

    # Remove columns with "unnamed" in their names
    # concatenated_data = concatenated_data.loc[:, ~concatenated_data.columns.str.contains('unnamed', case=False)]

    # Write the concatenated data to the output file without the index
    concatenated_data.to_csv(output_filename, index=False)
    print(f"Concatenated data saved to {output_filename}")


def pix_dist(x1,x2,y1,y2):

    '''
    Find the linear distance between two sets of points (x1,y1) -> (x2,y2) 
    given by:
    
    .. math ::
        
       d = \\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
    
    :param x1: x position of point 1
    :type x1: float
    :param x2: x position of point 2
    :type x2: float
    :param y1: y position of point 1
    :type y1: float
    :param y2: y position of point 2
    :type y2: float
    :return: Distance between to points
    :rtype: float

    '''

    import numpy as np

    z1 = (x1-x2)**2
    z2 = (y1-y2)**2

    r = np.sqrt(z1+z2)

    return r


def arcmins2pixel(value,pixel_scale):

    '''
    Convert distance given in Arc minutes to distance in linear pixel coordinates  using the following equation:
    
    .. math::
        
       pixels = 60/3600  \\times pixel\_scale \\times arcmins
    
    :param value: linear distance in arcmins
    :type value: float
    :param pixel_scale: Pixel scale of image in degrees. This is typically the output format when using Astropy's :math:`wcs.utils.proj\_plane\_pixel\_scales(w)` function, where :math:`w` is an images WCS object. 
    :type pixel_scale: float
    :return: linear distance in pixels
    :rtype: float


    '''

    pixels = (60 * (1/(3600*pixel_scale))) * value

    return pixels

def gauss_1d(x,A,x0,sigma):

    '''
    1D gaussian function given by:
    
    .. math::
        
       G = A \\times e^{-\\frac{x-x_o}{2\\times \sigma^2}}
    
    where *G* is the 1D gaussian function, *A* is the amplitude, *x* is the linear
    range of the function, :math:`x_0` is the center of the function, and
    :math:`\sigma` is the standard deviation.
    
    
    :param x: Linear range of gaussian function
    :type x: 1D array
    :param A: Amplitude of gaussian function
    :type A: float
    :param x0: Center/maximum of gaussian function
    :type x0: float
    :param sigma: sigma/width of gaussian function
    :type sigma: float
    :return: Returns 1 dimensional function with length equal to length of input x
    array
    :rtype: 1D array

    '''

    import numpy as np

    G = A*np.exp(-(x-x0)**2/(2*sigma**2))

    return G

def gauss_sigma2fwhm(image_params):

    '''
        Convert sigma value to full width half maximum (FWHM) for gaussian
    function. The FWHM of a gaussian is then given by:
    
    .. math::
        
       FWHM = 2\\times \\sqrt{2\\times Log_e(2)}\\times \sigma
    
    where :math:`\sigma` is the standard deviation of the Gaussian profile
    
    :param image_params: Dictionary containing the key *sigma* with corresponding
    value
    :type image_params: dict
    :return: Full width half maximum value
    :rtype: float

    '''

    import numpy as np

    sigma = image_params['sigma']

    fwhm = 2*np.sqrt(2*np.log(2)) * sigma

    return fwhm


def gauss_fwhm2sigma(fwhm,image_params = None):

    '''
        Convert  full width half maximum (FWHM) for gaussian
    function to sigma value. The FWHM of a gaussian is then given by:
    
    .. math::
        
       \sigma= \\frac{FWHM}{2\\times \\sqrt{2\\times Log_e(2)}}
    
    
    
    :param fwhm: full width half maximum of the gaussian profile
    :type fwhm: float
    :return: Sigma value (standard deviation) of the gaussian profile
    :rtype: float

    '''

    import numpy as np

    sigma= fwhm / (2*np.sqrt(2*np.log(2)))

    return sigma

def gauss_2d(image, x0, y0, sky , A, image_params):

    '''
    2D gaussian function given by:
    
    .. math::
        
       G = A \\times e^{-\\frac{(x-x_o)^2 - (y-y_0)^2}{2\\times \sigma^2}} + sky
     
    where *G* is the 2D gaussian function, *A* is the amplitude, *x* and *y* are the linear
    range of the function, :math:`x_0` and :math:`y_0` are the centers of the function,
    :math:`\sigma` is the standard deviation, and *sky* is the amplitude offset of the function
    
    :param image: 2 dimensional grid to map Gaussian onto 
    :type image: 2D array
    :param x0: x-center of gaussian function
    :type x0: float
    :param y0: y-center of gaussian function
    :type y0: float
    :param sky: sky/offset of gaussian function
    :type sky: float
    :param A: Amplitude of gaussian function
    :type A: float
    :param image_params: Dictionary containing the key *sigma* with corresponding value
    :type image_params: dict
    :return: 2D gaussian function with the same shape as image input
    :rtype: 2D array

    '''

    from numpy import exp,array

    sigma = image_params['sigma']

    (x,y) = image

    a = array(x-x0)**2

    b = array(y-y0)**2

    c = 2*sigma**2

    d =  A *  exp( -1*(a+b)/c )

    e =  d + sky

    return  e.flatten()



def moffat_fwhm(image_params):
    
    '''

    Calculate FWHM from Moffat function using: 
    
    .. math::
        
       FWHM = 2 \\times \alpha \\times \\sqrt{2^{\\frac{1}{\\beta}}-1}
    
    where :math:`\alpha` corresponds to the width of moffat function and :math:`\\beta` describes the wings
    
    :param image_params: Dictionary containing 2 keys: *alpha* corresponding to the fitted width of the moffat function and *beta* describing the wings.
    :type image_params: dict
    :return: Full width half maximum of moffat function
    :rtype: float

    '''
    
    from numpy import sqrt

    alpha = image_params['alpha']
    beta = image_params['beta']

    fwhm  = 2 * alpha *  sqrt((2**(1/beta))-1)

    return fwhm


def SNR(maxPixel,noiseBkg):
    
    
    import warnings
    
    with warnings.catch_warnings():
    
        SNR = maxPixel/noiseBkg
    
    return SNR



def SNR_err(SNR):

    
    '''
    Error associated with signal to noise ratio (S/N). Equation  taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_. Whe can associate the error on the instrumental magnitude of a source as:


    .. math :: 
        
       m \\pm \delta m = -2.5 \\times log_{10} ( S \\pm N) 

       m \\pm \delta m = -2.5 \\times log_{10} ( S  (1 \\pm N / S ) )

       m \\pm \delta m = -2.5 \\times log_{10} ( S )   - 2.5 \\times log_{10}(1 \\pm N / S ) )

       \delta m = \mp 2.5\\times log_{10} (1 + \\frac{1}{S/N}) \\approx \mp 1.0875 (N / S)

    :param SNR: Signal to noise ratio of a point-like source. 
    :type SNR: float
    :return: Error associated with that source's S / N
    :rtype: float

    '''
    
    from numpy import log10,errstate

    with errstate(divide='ignore', invalid='ignore'):
        SNR_err = 2.5 * log10(1 + (1/SNR))

    return SNR_err

def quadratureAdd(values):
    from numpy import sqrt
    return sqrt(sum([i**2 for i in values]))

def norm(array,norm2one = True):
    
    '''
    Normalise array to between 0 and 1 while ignoring nans using the following:
    
    
    .. math::
        
       |A| = \\frac{A - min(A)}{max(A)-min(A)}
    
    
    
    :param array: array of values
    :type array: arrat
    :return: Normalised array
    :rtype: arrat

    '''

    from numpy import nanmin as np_min
    from numpy import nanmax as np_max

    norm_array = (array - np_min(array))/(np_max(array)-np_min(array))

    return norm_array
  
def moffat_2d(image, x0,y0, sky , A, image_params):

    '''
    Returns 2D moffat function which is given by:
    
    
    .. math::
    
       M = A\\times (1+\\frac{(x-x_o)^2 + (y-y_0)^2}{\sigma^2})^{-\\beta} +
    sky
    
    
    `Credit: ltam
    <https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html>`_
    
    
    :param image: 2 dimensions grid to map Moffat on
    :type image: 2D  array
    :param x0: x-center of Moffat function
    :type x0: float
    :param y0: y-center of Moffat function
    :type y0: float
    :param sky: sky/offset of Moffat function
    :type sky: float
    :param A: Amplitude of Moffat function
    :type A: float
    :param image_params: Dictionary containing the keys "alpha" and "beta" with
    their corresponding values
    :type image_params: dict
    :return: 2D Moffat function with the same shape as image input
    :rtype: 2D  array

    '''
    (x,y) = image

    alpha = image_params['alpha']
    beta = image_params['beta']

    a = (x-x0)**2

    b = (y-y0)**2

    c = (a+b)/(alpha**2)

    d = (1+c) ** -beta

    e = (A*d)+sky

    return e.flatten()


def mag(flux):
    '''
    Calculate magnitude of a point source
    
    :param flux: Flux in counts per second measured from source
    :type flux: float or array
    :return: Magnitude of source on standard system, returns nan if flux is <= 0
    :rtype: float or array
    '''
    import numpy as np
    import pandas as pd
    if isinstance(flux, (int, float)):
        if flux <= 0:
            return np.nan
    elif isinstance(flux, (pd.core.series.Series,np.ndarray)):
        flux[flux <= 0] = np.nan
        
    else:
        print(flux)
        
        


    mag_inst = -2.5 * np.log10(flux)

    return mag_inst

def rebin(arr, new_shape):
    
    '''
     Rebin an array into a specific 2D shape
     
    :param arr: Array of values
    :type arr: array
    :param new_shape: New shape with which to rebin array into
    :type new_shape: tuple
    :return: rebinned array
    :rtype: array

    '''
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def scale_roll(x,xc,m):
    
    '''
    Used in building PSF function. When shiting and aligning residual tables this
    functions translates pixel shifts between different images cutouts.
    
    
    :param x: pixel position
    :type x: gloat
    :param xc: pixel position to which we want to move to
    :type xc: float
    :param m: scale multiplier
    :type m: int
    :return: DESCRIPTION
    :rtype: TYPE

    '''

    dx = (x - xc)
    
    if m !=1:
        
        shift = int(round(dx * m))
        
    else:
        
        shift = int(dx * m)
        
        
    return shift


def Gaussian(gridx,gridy, x0, y0, sky , A, sigma):

    '''
    2D gaussian function given by:
    
    .. math::
        
       G = A \\times e^{-\\frac{(x-x_o)^2 - (y-y_0)^2}{2\\times \sigma^2}} + sky
     
    where *G* is the 2D gaussian function, *A* is the amplitude, *x* and *y* are the linear
    range of the function, :math:`x_0` and :math:`y_0` are the centers of the function,
    :math:`\sigma` is the standard deviation, and *sky* is the amplitude offset of the function
    
    :param image: 2 dimensional grid to map Gaussian onto 
    :type image: 2D array
    :param x0: x-center of gaussian function
    :type x0: float
    :param y0: y-center of gaussian function
    :type y0: float
    :param sky: sky/offset of gaussian function
    :type sky: float
    :param A: Amplitude of gaussian function
    :type A: float
    :param image_params: Dictionary containing the key *sigma* with corresponding value
    :type image_params: dict
    :return: 2D gaussian function with the same shape as image input
    :rtype: 2D array

    '''

    from numpy import exp,array

    x = gridx
    y = gridy
    

    a = array(x-x0)**2

    b = array(y-y0)**2

    c = 2*sigma**2

    d =  A *  exp( -1*(a+b)/c )

    e =  d + sky
    
    return  e


def Moffat(gridx,gridy, x0,y0, sky , A, alpha,beta = 4.675):

    '''
    Returns 2D moffat function which is given by:
    
    
    .. math::
    
       M = A\\times (1+\\frac{(x-x_o)^2 + (y-y_0)^2}{\sigma^2})^{-\\beta} +
    sky
    
    
    `Credit: ltam
    <https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html>`_
    
    
    :param image: 2 dimensions grid to map Moffat on
    :type image: 2D  array
    :param x0: x-center of Moffat function
    :type x0: float
    :param y0: y-center of Moffat function
    :type y0: float
    :param sky: sky/offset of Moffat function
    :type sky: float
    :param A: Amplitude of Moffat function
    :type A: float
    :param image_params: Dictionary containing the keys "alpha" and "beta" with
    their corresponding values
    :type image_params: dict
    :return: 2D Moffat function with the same shape as image input
    :rtype: 2D  array

    '''
    x = gridx
    y = gridy

    a = (x-x0)**2

    b = (y-y0)**2

    c = (a+b)/(alpha**2)

    d = (1+c) ** -beta

    e = (A*d)+sky

    return e


def fwhmMoffat(alpha,beta):
    
    '''

    Calculate FWHM from Moffat function using: 
    
    .. math::
        
       FWHM = 2 \\times \alpha \\times \\sqrt{2^{\\frac{1}{\\beta}}-1}
    
    where :math:`\alpha` corresponds to the width of moffat function and :math:`\\beta` describes the wings
    
    :param image_params: Dictionary containing 2 keys: *alpha* corresponding to the fitted width of the moffat function and *beta* describing the wings.
    :type image_params: dict
    :return: Full width half maximum of moffat function
    :rtype: float

    '''
    
    from numpy import sqrt

    fwhm  = 2 * alpha *  sqrt((2**(1/beta))-1)

    return fwhm

def fwhmGaussian(sigma):

    from numpy import sqrt,log
    
    fwhm = 2*sqrt(2*log(2)) * sigma

    return fwhm



def sigmaGaussian(fwhm):

    from numpy import sqrt,log
    
    sigma= fwhm / (2*sqrt(2*log(2)))

    return sigma

def alphaMoffat(fwhm,beta = 4.675):
    
    '''

    Calculate FWHM from Moffat function using: 
    
    .. math::
        
       FWHM = 2 \\times \alpha \\times \\sqrt{2^{\\frac{1}{\\beta}}-1}
    
    where :math:`\alpha` corresponds to the width of moffat function and :math:`\\beta` describes the wings
    
    :param image_params: Dictionary containing 2 keys: *alpha* corresponding to the fitted width of the moffat function and *beta* describing the wings.
    :type image_params: dict
    :return: Full width half maximum of moffat function
    :rtype: float

    '''
    
    from numpy import sqrt



    alpha = 0.5 *fwhm * 1/(sqrt(2**(1/beta) -1))

    return alpha

def trim_zeros_slices(arr):
    
    '''

    TriM a 2D array of horizontal or vertical rows completely filled with zeroes. This is useful when aligning two images When  doing so if there isn't significant overlap between the two images, the resultant images may have vertical and horizontal lines completely filled with zeroes. This function will accept an image with said zeroed columns/row and return a smaller image with those arrays removed. This function will not exclude partially filled columns or rows.
    
    Credit: `Stackoverflow <https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d>`_

    
    :param arr: 2D array with horizontal or vertical rows/columns filled with zeroes.
    :type arr: 2D array.
    :return: 2D array which has been cleaned of zero columns and index map for original array.
    :rtype: tuple

    '''
    
    import numpy as np

    # if all([len(i)==0 for i in np.nonzero(arr)]):

    #     return arr,None


    boolean_array = np.zeros(arr.shape).astype(bool)
    
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))

    boolean_array[slices] = True

    return arr[slices],boolean_array



def distance_to_uniform_row_col(image, x, y):
    
    import numpy as np 
    # Ensure the input is a numpy array for easier manipulation
    image = np.array(image)
    rows, cols = image.shape
    
    # Find rows with all same values
    uniform_rows = [i for i in range(rows) if np.all(image[i] == image[i, 0])]
    
    # Find columns with all same values
    uniform_cols = [j for j in range(cols) if np.all(image[:, j] == image[0, j])]

    # Calculate the Manhattan distance to the nearest uniform row
    if uniform_rows:
        row_distances = [abs(x - row) for row in uniform_rows]
        min_row_distance = min(row_distances)
    else:
        min_row_distance = float('inf')  # If no uniform rows are found

    # Calculate the Manhattan distance to the nearest uniform column
    if uniform_cols:
        col_distances = [abs(y - col) for col in uniform_cols]
        min_col_distance = min(col_distances)
    else:
        min_col_distance = float('inf')  # If no uniform columns are found

    # Return the minimum distance to a uniform row or column
    return min(min_row_distance, min_col_distance)



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


def fluxUpperlimit(n,sigma,beta_p = 0.75 ):
    
   
    from numpy import sqrt
    from scipy.special import erfinv

    # beta_p = 1-beta
    f_ul = (n + (sqrt(2) * erfinv( (2*beta_p) - 1))) * sigma

    return f_ul


def create_ds9_region_file(x_list, y_list, radius, 
                           filename="ds9_region.reg",
                           color = 'green',
                           text = '', 
                           overwrite = False,
                           correct_position = True):
    """
    Create a DS9 region file containing circular regions for multiple points.

    Parameters:
    - x_list (list): List of x positions for the centers of the circles.
    - y_list (list): List of y positions for the centers of the circles.
    - radius (float): Radius of the circles.
    - filename (str): Name of the DS9 region file to be created.

    Returns:
    - None
    """
    
    cor = 0
    if correct_position:
        cor = 1
    if len(x_list) != len(y_list):
        raise ValueError("Number of x and y positions must be the same.")

    region_content = ""
    for x, y in zip(x_list, y_list):
        region_content += f"circle({x+cor}, {y+cor}, {radius}) # color={color} text={text}\n"

    if overwrite:
        n = "w"
    else:
        n = "a+"

    with open(filename, n) as file:

        file.write(region_content)

    # print(f"DS9 region file '{filename}' created successfully.")


def write_position_2_ascii(dataframe, output_file):
    
    """
    Write x_pix and y_pix columns from a Pandas DataFrame to an ASCII file.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame with x_pix and y_pix columns.
    - output_file (str): Output ASCII file name.

    Returns:
    - None
    """
    
    import pandas as pd
    import numpy as np



    # Extract x_pix and y_pix columns
    if 'x_pix' not in dataframe.columns or 'y_pix' not in dataframe.columns:
        raise ValueError("DataFrame must have 'x_pix' and 'y_pix' columns.")

    x_pix_column = dataframe['x_pix'].values
    y_pix_column = dataframe['y_pix'].values

    # Combine columns into a new DataFrame
    output_dataframe = pd.DataFrame({'x_pix': x_pix_column, 'y_pix': y_pix_column})

    # Write DataFrame to ASCII file with "X Y" header
    with open(output_file, 'w') as file:
        file.write("x y\n")  # Header line
        output_dataframe.to_csv(file, sep=' ', header=None, index=False,float_format = '%.3f')

# Example Usage:
# Assuming you have a DataFrame called 'df' with columns 'x_pix' and 'y_pix'
# and you want to write the data to a file named 'output.asc'
# write_to_asciifile(df, 'output.asc')

# Example Usage:
# Assuming you have a DataFrame called 'df' with columns 'x_pix' and 'y_pix'
# and you want to write the data to a file named 'output.asc'
# write_to_asciifile(df, 'output.asc')



def print_progress_bar(iterable, total=None, prefix='', length=50, fill='█', title=None):
    
    import sys
    """
    Print a progress bar in the terminal for a loop.

    Parameters:
        iterable (iterable): The iterable object (e.g., list, range) that you're iterating over.
        total (int, optional): Total number of iterations. If None, the length of the iterable will be used.
        prefix (str, optional): Prefix to display before the progress bar.
        length (int, optional): Length of the progress bar in characters.
        fill (str, optional): Character used to fill the progress bar.
        title (str, optional): Title to be displayed above the progress bar.

    Example usage:
        for i in print_progress_bar(range(100), title="Processing", prefix='Progress', length=40):
            # Your loop code here
    """
    if total is None:
        total = len(iterable)



    def print_bar(iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}%')
        
        
    for i, item in enumerate(iterable):
        sys.stdout.write(f'\r{title}\n' if title else '')
        print_bar(i + 1)

        yield item

    if total > 0 and i + 1 == total:
        print('\n'*3)  # Print a newline after 100% completion
# Example usage:


# # Example usage:
# import time

# for _ in print_progress_bar(range(100), prefix='Progress', length=40):
#     # Simulate some work being done
#     time.sleep(0.1)


    
    


def get_normalized_histogram(data, bins='auto'):
    import numpy as np
    # Create the histogram
    data = data[~np.isnan(data)]
    if bins =='auto':
        bins = calculate_bins(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    # Calculate the normalization factor
    normalization_factor = np.nanmax(hist)

    # Normalize the histogram
    normalized_hist = hist / normalization_factor

    return  normalized_hist,bin_edges


def dict_to_string_with_hashtag(dictionary,float_format = '%.3f'):
    result = ""
    for key, value in dictionary.items():
        
        if isinstance(value,list):
            if len(value)==1:value = value[0]
            
        if isinstance(value, float):
            value = float_format % value
        result += f"#{key}: {value}\n"
    return result


def beta_value(n,f_ul,sigma,noise = 0):

    '''
    False negatives function for the fraction of real sources that go undetected. This is implemented in AutoPHOT to determine whether a source,  more specifically it's flux, can be confidently assumed to arise from the source and not be associated with the background noise distribution. We describe the probability that a suspected source can be confidently assumed to be seperate from the flux due to the background by:
    
    .. math::
        
       1 - \\beta = \\beta{\\prime} = \\frac{1}{2}(1 - \mathit{erf}(\\frac{n\sigma_{bkg} - f_{source}}{\sigma_{bkg} \\sqrt{2}}))
       
    where :math:`\\beta{\\prime}` is the confident that source is not apart of the underlying noise distribution, :math:`\mathit{erf}` is the `Error function <https://mathworld.wolfram.com/Erf.html>`_, :math:`f_{source}` is the brightest pixel that can be associated with a source, and :math:`\sigma_{bkg}` is the standard deviation of the background noise distribution.
     
    In other words, :math:`100\\times (1-\\beta) \%` of the
    sources with flux :math:`f_{source}` will have flux measurements > :math:`n\sigma_{bkg}`
    
    
    Credit: `F. Masci <http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf>`_
    
    
    :param n:  Level above background to be considered a genuine detection. Typical values are :math:`n = 3`, which equations to a :math:`3\sigma` confidence level, or in other words there is a 0.135% probability that the measured flux is a spurious noise spike. 
    :type n: float
    :param sigma: Background standard deviation.
    :type sigma: float
    :param f_ul: Brightest pixel value that can be associated with a suspected source.
    :type f_ul: float
    :param noise Mean offset of flux measurement. This value should be included if the measurement of the source is not background subtracted, defaults to 0
    :type noise: float, optional
    :return: False negatives probability which describes the confidence to which a source can be consider real and not be associated with a spurious noise spike.
    :rtype: float
    '''

    from scipy.special import erf

    from numpy import sqrt

    source_flux = f_ul - noise

    if isinstance(source_flux,list):
        source_flux = [i if i>0 else 0 for i in source_flux]


    z = ((n*sigma) - (f_ul-noise))/(sqrt(2)*sigma)

    beta = 0.5 * (1 - erf(z))

    return beta
