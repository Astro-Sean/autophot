def SNR(flux_star,flux_sky,exp_t,radius,G = 1,RN = 0,DC = 0 ):
    '''

    Compute the signal to noise ratio (S/N) equation taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_.


    .. math::
        
       S/N = \\frac{ F_{ap} }{ F_{ap} + F_{sky,ap,n} + (RN ^2 + \\frac{G^2}{4} \\times n_{pix}) + (D \\times n_{pix} \\times t_{exp}) } ^{0.5}
   
    where :math:`F_{ap}` is the flux under and aperture of a specific radius, and likewise :math:`F_{sky,ap,n}` is the flux due to the sky background under the same aperture. In other words :math:`F_{sky,ap,n} = \\langle counts_{sky_ap}  \\rangle / T_{exp} \\times n` where :math:`n = \\pi r ^2`.
    
    :param flux_star: Flux in :math:`counts / second` that we assume is coming from source.
    :type flux_star: float
    :param flux_sky: Flux in :math:`mean\ counts /  second / pixel` that we assume is coming from sky background.
    :type flux_sky: float
    :param exp_t: Exposure time in seconds.
    :type exp_t: float
    :param radius: Radius of aperture in pixels.
    :type radius: float
    :param G: GAIN on CCD in :math:`e^{-1} /  ADU` , defaults to 1.
    :type G: float, optional
    :param RN: Read noise of CCD in :math:`e^{-1} /  pixel`, defaults to 0.
    :type RN: float, optional
    :param DC: Dark Current of CCD in :math:`e^{-1} /  pixel  /  second`, defaults to 0.
    :type DC: float, optional
    :return: Signal to noise of given source
    :rtype: float

    '''

    G = float(G)
    

    
    import numpy as np

    counts_source = flux_star * exp_t

    Area = np.pi * radius ** 2

    sky_shot = flux_sky  * exp_t * Area

    read_noise = ((RN**2) + (G/2)**2) * Area

    dark_noise =  DC * exp_t * Area

    SNR = counts_source / np.sqrt(counts_source + sky_shot + read_noise  + dark_noise)

    return SNR


def SNR_err(SNR):
    
    '''
    Error associated with signal to noise ratio (S/N). Equation  taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_. Where can associated the error on the instrumental magnitude of a source as:


    .. math :: 
        
       m \\pm \delta m = -2.5 \\times log_{10} ( S \\pm N) 

       \\rightarrow = -2.5 \\times log_{10} ( S  (1 \\pm N / S ) )

       \\rightarrow = -2.5 \\times log_{10} ( S )   -2.5 \\times log_{10}(1 \\pm N / S ) )

       \delta m = \mp 2.5\\times log_{10} (1 + \\frac{1}{S/N}) \\approx \mp 1.0875 (N / S)

    :param SNR: Signal to noise ratio of a point-like source. 
    :type SNR: float
    :return: Error associated with that source's S / N
    :rtype: float

    '''
    
    import numpy as np

    # If SNR is given as single value and not a list or array
    if isinstance(SNR,int) or isinstance(SNR,float):
        if SNR <= 0:
            return 0
        SNR_err = np.array([2.5 * np.log10(1 + (1/SNR))])
        return SNR_err[0]
    
    else:
        SNR = np.array(SNR)

    # Remove SNR values if less than zero, replace with zero i.e source not detected
    SNR_cleaned = [i if i>0 else 0 for i in SNR]

    if isinstance(SNR_cleaned,float):
        SNR_cleaned = np.array(SNR_cleaned)

    SNR_err = np.array([2.5 * np.log10(1 + (1/snr)) if snr>0 else np.nan for snr in SNR_cleaned])

    return SNR_err



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

    boolean_array = np.zeros(arr.shape).astype(bool)
    
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))

    boolean_array[slices] = True

    return arr[slices],boolean_array



def set_size(width,aspect=1):
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
    fraction = 1

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



def get_distinct_colors(n):
    
    '''
    Given a number of colors desired, :math:`\mathit{n}` generate a list of :math:`\mathit{n} approximately distinct colours.

    Credit: `StackOverflow <https://stackoverflow.com/questions/37299142/how-to-set-a-colormap-which-can-give-me-over-20-distinct-colors-in-matplotlib>`_
    
   
    :param n: number of desired distinct colours
    :type n: int
    :return: List of rgb colours to be used in figure
    :rtype: list
    
    '''
    colors = []
    
    from colorsys import hls_to_rgb
    import numpy as np

    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return colors


def border_msg(msg,body = '-',corner = '+',):
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
    print('\n' + result + '\n')
    return
    

def error_log(value,error):
    
    '''
    
    Calculate relative error on value which is displayed in log form
    
    Credit: `Here <https://faculty.washington.edu/stuve/log_error.pdf>`_
    
    
    :param value: Value with associated error
    :type value: float
    :param error: Error associated with value
    :type error: float
    :return: Error in log form
    :rtype: float

    '''

    log10_err = 0.434 * error / value

    return log10_err


def getheader(fpath):

    """
    
    Robust function to get header from :math:`FITS` image for use in AutoPHOT. Due to a :math:`FITS` image typically having multiple headers, which may have useful important information spread across multiples heres, this function returns (if appropriate) a single header file containing all needed header files. This function aims to find correct telescope header info based on "Telescop" header key.

    :param fpath: Location of fits image which contains a header file
    :type fpath: str
    :return: header information
    :rtype: header object
    
    """

    # Need to rename this function
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


def order_shift(x):
    
    '''
    Get the order of magnitude of an array. Use to shift array to between 0-10: The fucntion uses the following equation to find the order of magnitude of an array.
    
    .. math::
    
       order\_of\_mag (x) = 10^{MAX(FLOOR(Log_{10}(x)))}
    
    :param x: Array of values 
    :type x: float
    :return: Order of magnitude
    :rtype: float

    '''

    import numpy as np

    idx = (np.isnan(x)) | (np.isinf(x)) | (x<=0)

    order_of_mag = 10**np.nanmax(np.floor(np.log10(x[~idx])))

    return order_of_mag


def pixel2arcsec(value,pixel_scale):

    '''

    Convert distance in pixel to distance in arcsecs using the following equation:
    
    .. math::
        
       arcsec = pixel \\times 3600 \\times pixel\_scale
    
    :param value: linear distance in pixels
    :type value: float
    :param pixel_scale: Pixel scale of image in degrees. This is typically the output format when using Astropy's :math:`wcs.utils.proj\_plane\_pixel\_scales(w)` function, where :math:`w` is an images WCS object. 
    :type pixel_scale: float
    :return: linear distance in arcseconds
    :rtype: float
    
    '''

    arcsecs =  value * (3600*pixel_scale)

    return arcsecs


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


def getimage(fpath):
    
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
        except:
            image = fits.getdata(fpath)

    except:
        with fits.open(fpath,ignore_missing_end=True) as hdul:
            hdul.verify('silentfix+ignore')
            image = hdul[0].data


    if len(image.shape) != 2:

        base=os.path.basename(fpath)

        raise Exception('Warning:: %s not 2-D array' % base)

    return image


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


def f_ul(n,beta_p,sigma,):
    
    '''
    Flux upper limit for an environment with a Gaussian like noise distribution. This value represents the flux required, above which we can confidently assume that a flux measurement is likely due to a genuine source and not apart of the underlying noise distribution i.e a spurious noise spike. We calculate the flux upper limit :math:`F_{UL}` by:
    
    .. math::
        
       F_{UL} = [n + \\sqrt{2}\\times \mathit{erf} ^{-1}(2\\beta{\\prime} - 1)]\sigma_{bkg}
    
    
    where *n* is the confidence of the measure (typically set to *3*, equivalent to a :math:`3\sigma` confidence),  :math:`\mathit{erf}^{-1}` is the `Inverse Error function <https://mathworld.wolfram.com/Erf.html>`_, :math:`\\beta{\\prime}` is the desired confidence of the measurement, and :math:`n\sigma_{bkg}` is the standard deviation of the background noise distribution.
    
    Credit: `F. Masci <http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf>`_
    
    
    :param n: Level above background to be considered a genuine detection. Typical values are :math:`n = 3`, which equations to a :math:`3\sigma` confidence level, or in other words there is a 0.135% probability that the measured flux is a spurious noise spike. 
    :type n: float
    :param beta_p: Probability that a flux measurement is likely due to a genuine source and not apart of the underlying noise distribution. Set to  :math:`0 \\rightarrow 1`, although a more realistic range  :math:`0.5 \\rightarrow 0.98`.  Defaults to 0.75
    :type beta_p: float, optional
    :param sigma: Standard deviation of background
    :type sigma: flost
    :return: Flux upper limit that allows use to place a confidence on a measure given by the :math:`\\beta{\\prime}` term.
    :rtype: float


    '''
    
    from numpy import sqrt
    from scipy.special import erfinv

    # beta_p = 1-beta
    f_ul = (n + (sqrt(2) * erfinv( (2*beta_p) - 1))) * sigma

    return f_ul


def calc_mag_error(flux,noise):
    
    '''
    Calculate magnitude error due to noise on flux measurement. 
    Magnitude error is given by:

    .. math::
        
       \delta m = \\pm 2.5Log_{10}(1 + \\frac{1}{S/N}) \\approx 1.0857 \\times N/S
       
    :param flux: Flux measurement
    :type flux: float
    :param noise: Error on flux measurement
    :type noise: float
    :return: Magnitude error due to noise on flux measurement
    :rtype: float

    '''
    
    return 1.0857 * noise/flux

def calc_mag(flux, gain=1, zp=0):
    
    '''
    Calculate magnitude of apoint source
    
    :param flux: Flux in counts per second measured from source
    :type flux: array
    :param zp: Zeropoint to place measurement on standard system
    :type zp: array
    :return: Magnitude of source on standard system
    :rtype: array

    '''


    import numpy as np

    # Iitial list with zeropoint
    if isinstance(flux,float):
        flux = [flux]

    gain = 1

    mag_inst = np.array([-2.5*np.log10(i*gain)+zp if i > 0.0 else np.nan for i in flux ])

    return mag_inst




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

def pix_dist(x1,x2,y1,y2):

    '''
    Find the linear distance between two sets of points (x1,y1) -> (x2,y2) 
    given by:
    
    .. math ::
        
       d = \\sqrt{(x_1 - x_2) + (y_1 - y_2)^2}
    
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

def weighted_avg_and_std(values, weights):

    '''
    Return the average of an array of values with given weights

    :param values: array of values
    :type values: array
    :param weights: weights associated with values
    :type weights: array
    :return: weighted average and varience
    :rtype: tuple

    '''

    import numpy as np
    import math

    mask = ~np.ma.masked_array(values, np.isnan(values)).mask

    values = np.array(values)[mask]
    weights = np.array(weights)[mask]

    if len(values) == 0 or len(weights) == 0:
        values  = np.array(np.nan)
        weights = np.array(np.nan)


    average = np.average(values, weights=weights)

    variance = np.average((values-average)**2, weights=weights)

    return (average, math.sqrt(variance))


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


def array_correction(x):

    '''

    Correct the float nature of pixel position used in numpy. If a pixel position 
    is more than halfway accros a pixel roundup, else round down.

    :param x: pixel position
    :type x: float
    :return: corrected position
    :rtype: float

    '''
    from numpy import ceil, floor

    diff =  float(x) % 1

    if diff > 0.5:

        return int(ceil(x))

    elif diff <= 0.5:

        return int(floor(x))


def norm(array):
    
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


def round_half(number):
    
    '''
    Round a number to half integer values i.ie +/-0.5
    
    :param number: Values to be rounded
    :type number: array
    :return: Values rounded to their nearest 0.5
    :rtype: array

    '''
    return round(number * 2) / 2

def to_latex(autophot_input,peak = None):

    
    '''
    work in progress


    '''


    import pandas as pd
    from astropy.time import Time
    import numpy as np
    import os

    from autophot.packages.functions import border_msg

    border_msg('Preparing Latex ready table of photometry')

    wdir = autophot_input['fits_dir']
    obj = autophot_input['target_name']

    new_dir = '_' + autophot_input['outdir_name']

    base_dir = os.path.basename(wdir)
    work_loc = base_dir + new_dir

    new_output_dir = os.path.join(os.path.dirname(wdir),work_loc)

    fname = 'REDUCED.csv'

    fname = os.path.join(new_output_dir,fname)

    if peak is None:
        peak = input('Reference epoch in mjd [press enter to ignore]: ') or None

    if peak is None:
        print('*Not including reference epoch')
    else:
        try:
            peak = float(peak)
        except:
            raise Exception('Could not convert %s to float' % peak)

    data = pd.read_csv(fname)

    data.drop_duplicates(inplace = True)


    filters  = ['u','g','r','i','z','U','B','V','R','I' ,'J','H','K']

    data.reset_index(inplace = True)

    delete_filters = []
    for f in filters:
        if 'zp_'+f not in data.columns:

            delete_filters.append(f)

    for i in delete_filters:
        filters.remove(i)

    tele_list = list(set(data.telescope))

    df_list = []

    fwhm_limit = 10

    for tele in tele_list:

        data_t = data[data.telescope == tele]

        mjd = list(set(round_half(data_t.mjd)))

        for m in mjd:
            # print(m)

            # data_t_mjd = data_t[round_half(data_t.mjd) == m]

            df_tmp = {}

            # m_pick = data[data.mjd.round(0)== m].index

            m_pick = data_t[round_half(data_t.mjd)== m].index

            m_table= data_t[data_t.index == m_pick[0]].mjd.round(2).values[0]

            date_time = Time(str(m_table), format='mjd')

            date = date_time.iso.split(' ')[0]

            df_tmp['Date'] = date
            df_tmp['MJD'] = '%.1f' % m_table
            if not(peak is None):
                df_tmp['Phase [w.r.t %.2f mjd]' % peak] = '%.1f' % (m_table - peak)

            night_obs_filter = {}

            for f in filters:

                night_obs_filter[f] = []

                for i in m_pick:

                    row = data.iloc[i]

                    if not np.isnan(row['zp_'+f]):

                        d_fwhm = abs(row['fwhm'] - row['target_fwhm'])

                        fwhm_check = d_fwhm < fwhm_limit

                        lmag_check = (float(row[f]) < float(row['lmag_inject'])) | ( np.isnan(float(row['lmag_inject']) ))

                        if not np.isnan(float(row[f])) and lmag_check and fwhm_check :

                            filter_entry =  '%.2f  (%.2f)' % (row[f],row[f+'_err'])


                        else:

                            filter_entry =  '$>%.2f$' % (row['lmag_inject'])


                        night_obs_filter[f].append(filter_entry )

            df_tmp['FWHM'] = row['fwhm']

            for f in night_obs_filter:
                if len(night_obs_filter[f])==1:

                    filter_entry = night_obs_filter[f][0]
                elif len(night_obs_filter[f])==0:
                    filter_entry = np.nan

                else:
                    night_exps = [i.split(' ')[0] for i in night_obs_filter[f]]
                    lmag_search = ['>' in i for i in night_obs_filter[f]]
                    if np.sum(lmag_search) >=1:
                        filter_entry = np.array(night_obs_filter[f])[lmag_search][0]

                    else:
                        night_exps = [float(i.split(' ')[0]) for i in night_obs_filter[f]]
                        err = np.nanstd(night_exps)
                        if round(err,2)<0.01:
                            err = 0.01

                        # filter_entry =  '%.2f  (%.2f)^(%d)' % (np.nanmean(night_exps),err,len(night_exps))
                        filter_entry =  '%.2f  (%.2f)' % (np.nanmean(night_exps),err)

                df_tmp['$%s$' % f] = filter_entry


            df_tmp['Instrument'] = row['telescope']


            df_list.append(pd.DataFrame.from_dict([df_tmp]))


    new_df = pd.concat(df_list)

    new_df.reset_index()
    # new_df.dropna(subset=['$%s$'%f for f in filters],inplace = True,axis = 0,how = 'all')
    cols_order = ['Date','MJD'] + ['$%s$'%f for f in filters] + ['Instrument']
    if not (peak is None):
        cols_order.insert(2,'Phase [w.r.t %.2f mjd]' % peak,)


    new_df = new_df[cols_order]
    new_df.sort_values(by=['MJD'],inplace = True)


    out_file = 'latex_table_%s_%s.tex' % (obj,''.join(['%s'%f for f in filters]))
    output_fpath = os.path.join(new_output_dir,out_file)
    print('\nLatex table saved as: %s' % output_fpath)
    new_df.to_latex(output_fpath, #< where you want to save it
                    index = False,
                    na_rep = '-',# <- what to replace empty enrties with
                    longtable = False,
                    # float_format="{:0.2f}".format,
                    escape = False,
                    caption = 'caption',
                    label = 'label',
                    multicolumn = True)
