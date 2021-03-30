#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def set_size(width,
             aspect=1,
             fraction=1):

        """This is a summary of the aperture packge

        :param positions: list of tuples containing (x,y) positions of object
        :type positions: [ParamType](, optional)

        :return: Returns lost of aperature measurements
        :rtype: list
        """

        # Width of figure
        fig_width_pt = width * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        golden_ratio = (5**.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        fig_dim = (fig_width_in, fig_height_in * aspect)

        return fig_dim


def getheader(fpath):

    """Get fits image header. Find telescope header info based on "Telescop" header key
    if multiple headers are found, concatination into one large header file

    :param fpath: Location of fits image
    :type fpath: str

    :return: header information
    :rtype: object
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
                    headinfo_list += headinfo[i].header
                headinfo = headinfo_list
            else:
                # is length of list choose this a header
                headinfo = headinfo[0].header
    except Exception as e:
        print(e)

    return headinfo


def order_shift(x):

    import numpy as np
    idx = (np.isnan(x)) | (np.isinf(x)) | (x<=0)

    return 10**np.nanmax(np.floor(np.log10(x[~idx])))



def getimage(fpath):

    '''
    Function to find image for filepath
    '''

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
        print('error not 2-D array')

    return image

def beta_value(n,sigma,f_ul,mean = 0):

    '''
    detection probability from
    http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf
    '''
    from scipy.special import erf
    from numpy import sqrt

    z = (mean +( n*sigma) - f_ul)/(sqrt(2)*sigma)

    beta = 0.5 *(1-erf(z))

    return beta

def zeropoint(mag, counts, ct_gradient = None, dmag = None, airmass = None):

    """
    Calculate zeropint using:

        mag = -2.5 * log10 (counts) + ct(dmag) + zp + airmass
    """

    import numpy as np

    zp_list = [mag]

    mag_inst = -2.5 * np.log10(counts)
    zp_list.append(-1 * mag_inst)


    if type(counts) is np.array:
        counts[np.where(counts < 0.0)] = np.nan

    if ct_gradient != None and all(dmag) != None:
        ct = ct_gradient * dmag
        zp_list.append(-1 * ct)


    zp = sum(zp_list)
    return zp

def mag(counts, zp,ct_gradient = False,dmag = False,airmass = None):

    '''
    Calculate zeropint using:

    mag = - 2.5 * log10 (counts) + ct(dmag) + zp + airmass

    if negative counts code will return nan for

    '''

    try:

       import numpy as np
       import sys
       import os

       # Iitial list with zeropoint
       if isinstance(counts,float):
           counts = [counts]

       mag_inst = np.array([-2.5*np.log10(c)+zp if c > 0.0 else np.nan for c in counts ])


    except Exception as e:
       exc_type, exc_obj, exc_tb = sys.exc_info()
       fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
       print(exc_type, fname, exc_tb.tb_lineno,e)

    return mag_inst




def gauss_sigma2fwhm(image_params):
   '''
    Gaussian Function
   '''


   sigma = image_params['sigma']

   import numpy as np

   fwhm = 2*np.sqrt(2*np.log(2)) * sigma

   return fwhm


def gauss_fwhm2sigma(fwhm,image_params = None):

   import numpy as np

   sigma= fwhm / (2*np.sqrt(2*np.log(2)))

   return sigma


def gauss_2d(image, x0,y0, sky , A, image_params):

    sigma = image_params['sigma']
    from numpy import exp

    (x,y) = image

    a = (x-x0)**2

    b = (y-y0)**2

    c = (2*sigma**2)

    d =  A*abs(exp( -(a+b)/c))

    e =  d + sky

    return  e.ravel()



def moffat_fwhm(image_params):

   '''
    Moffat Profile
   '''

   alpha = image_params['alpha']
   beta = image_params['beta']

   from numpy import sqrt

   fwhm  = 2 * alpha *  sqrt((2**(1/beta))-1)

   return fwhm


def moffat_2d(image, x0,y0, sky , A, image_params):

    # https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html

    (x,y) = image

    alpha = image_params['alpha']
    beta = image_params['beta']

    a = (x-x0)**2

    b = (y-y0)**2

    c = (a+b)/(alpha**2)

    d = (1+c) ** -beta

    e = (A*d)+sky

    return e.ravel()

def pix_dist(x1,x2,y1,y2):
    import numpy as np


    z1 = (x1-x2)**2
    z2 = (y1-y2)**2
    r = np.sqrt(z1+z2)

    return np.array(r)

def weighted_avg_and_std(values, weights):
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
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def weighted_median(data, weights):
    import numpy as np

    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median




def pixel_correction(x,m):

    from numpy import ceil, floor

    diff =  x - int(x)

    if diff/m >= 0.5 or  diff ==0:
        return ceil(x) - 0.5

    elif diff/m < 0.5:
        return floor(x)


def array_correction(x):
    from numpy import ceil, floor

    diff =  float(x) % 1

    if diff > 0.5:

        return int(ceil(x))

    elif diff <= 0.5:

        return int(floor(x))

def norm(array):

    from numpy import nanmin as np_min
    from numpy import nanmax as np_max

    norm_array = (array - np_min(array))/(np_max(array)-np_min(array))


    return norm_array



def renorm(array,lb,up):
    import numpy as np
    s = up - lb
    n =  (array - np.min(array))/(np.nanmax(array)-np.min(array))
    m = (s * n) + lb
    return m

def find_2d_int_percent(count_percent,fwhm):
    from scipy.optimize import least_squares
    import numpy as np

    from scipy.integrate import dblquad
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))

    A = 1 / (2 * np.pi * sigma **2)
    gauss = lambda y,x: A * np.exp( -1 * (((y)**2+(x)**2)/(2*sigma**2)))
    fit = lambda x0: (count_percent - dblquad(gauss, -1*x0,x0,lambda x: -1*x0,lambda x: x0)[0])

    r = least_squares(fit,x0 = 3)
    return r.x[0]


def scale_roll(x,xc,m):
    dx = (x - xc)
    if m !=1:
        shift = int(round(dx *m))
    else:
        shift = int(dx *m)
    return shift
