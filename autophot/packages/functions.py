def SNR(flux_star,flux_sky,exp_t,radius,G = 1,RN = 0,DC = 0 ):
    '''
    
    S/N eqaution taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_

    
    :param flux_star: flux (counts per second) coming from source
    :type flux_star: array-like
    :param flux_sky: flux (counts per second) associated with sky background
    :type flux_sky: array-like
    :param exp_t: Expsoure time in seconds
    :type exp_t: float
    :param radius: Radius of aperture in pixels
    :type radius: float
    :param G: GAIN on CCD, defaults to 1
    :type G: float, optional
    :param RN: Readnoise of CCD, defaults to 0
    :type RN: float, optional
    :param DC: Dark Current of CCD, defaults to 0
    :type DC: float, optional
    :return: Signal to noise of given soures
    :rtype: array-like

    '''




    import numpy as np
    import warnings

    with warnings.catch_warnings():
        # Ignore  Runtime warnings
        warnings.simplefilter('ignore')


        G = float(G)

        counts_source = flux_star * exp_t

        star_shot_2 = flux_star * exp_t

        Area = np.pi * radius ** 2

        sky_shot_2 = flux_sky  * exp_t * Area

        read_noise_2 = ((RN**2) + (G/2)**2) * Area

        dark_noise_2 =  DC * exp_t * Area

        SNR = counts_source / np.sqrt((star_shot_2 + sky_shot_2 +read_noise_2  + dark_noise_2))

        return SNR

def SNR_err(SNR):
    '''
    S/N eqaution and associated error taken from `here <https://www.ucolick.org/~bolte/AY257/s_n.pdf>`_

    :param SNR: Signal to noise ratio
    :type SNR: array-like
    :return: Error associated with SNR
    :rtype: array-like

    '''
    
    import numpy as np

    # If SNR is given as single value and not a list or array
    if isinstance(SNR,int) or isinstance(SNR,float):
        if SNR <= 0:
            return 0
        SNR_err = np.array([2.5 * np.log10(1 + 1/SNR)])
        return SNR_err[0]


    else:
        SNR = np.array(SNR)

    # Remove SNR values if less than zero, replace with zero i.e source not detected
    SNR_cleaned = [i if i>0 else 0 for i in SNR]

    if isinstance(SNR_cleaned,float):
        SNR_cleaned = np.array(SNR_cleaned)

    SNR_err = np.array([2.5 * np.log10(1 + 1/snr) if snr>0 else np.nan for snr in SNR_cleaned])

    return SNR_err



def trim_zeros_slices(arr):
    '''
    Traim 2d array of horizontal or vertical rows completely filled with zeroes
    
    Credit: https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    
    :param arr: 2D array-like with horizontal or vertical rows/columns filled with zeroes 
    :type arr: array-like
    :return: 2d array which has been cleaned of zero columns and index map for original array 
    :rtype: tuple

    '''
    
    
    import numpy as np

    boolean_array = np.zeros(arr.shape).astype(bool)
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))

    boolean_array[slices] = True

    return arr[slices],boolean_array



def set_size(width,aspect=1):
    '''
     Function to generate size of figures produced by AutoPhot

    :param width: Width of figure in pts. 1pt == 1/72 inches
    :type width: float
    :param aspect: Aspect of image i.e. height  = width / golden ratio * aspect, default  = 1
    :type aspect: float
    :return: Returns tuple of width,height in inches ready for use.
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
    Get a list of n distinct colors
    :param n: number of desired distinct colours
    :type n: int
    :return: List of rgb colours to be used in figure
    :rtype: list

    '''
    # https://stackoverflow.com/questions/37299142/how-to-set-a-colormap-which-can-give-me-over-20-distinct-colors-in-matplotlib

    colors = []
    from colorsys import hls_to_rgb
    import numpy as np

    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return colors


def border_msg(msg):
    '''
    Print a message to console surrounded by a border
    
    :param msg: Message to display
    :type msg: str
    :return: displayed string with border
    :rtype: N/A

    '''
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= h + '\n'"|"+msg+"|"'\n' + h
    print('\n' + result + '\n')
    return
    

def error_log(value,error):
    '''
    Calculate error on value which is displyed in log form
    
    :param value: Value with associated error
    :type value: array-like
    :param error: Error assocaited with vlaue
    :type error: array-like
    :return: error in log form
    :rtype: array-like

    '''

    log10_err = 0.434 * error.value

    return log10_err


def getheader(fpath):

    """
    Get fits image header. Find correcct telescope header info based on "Telescop" header key
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
    
    '''
    Get the order of magnitude of an array. Use to shift array to between 0-10

    :param x: Array of values
    :type x: numpy array
    :return: Order of magnitude
    :rtype: float

    '''

    import numpy as np

    idx = (np.isnan(x)) | (np.isinf(x)) | (x<=0)

    order_of_mag = 10**np.nanmax(np.floor(np.log10(x[~idx])))

    return order_of_mag


def pixel2arcsec(value,pixel_scale):

    '''
    Convert distance in pixel to distance in arcsecs

    :param value: linear distance in pixels
    :type value: array_like
    :param pixel_scale: pixel scale of image in degrees
    :type pixel_scale: float
    :return: linear distance in arsecs
    :rtype: array_like

    '''

    arcsecs =  value * (3600*pixel_scale)

    return arcsecs


def arcmins2pixel(value,pixel_scale):

    '''
    Convert distance given in arcmins to distance in pixels

    :param value: linear distance in arcmins
    :type value: array_like
    :param pixel_scale: pixel scale of image in degrees
    :type pixel_scale: float
    :return: linear distance in pixels
    :rtype: array_like

    '''

    arcsecs = (60 * (1/(3600*pixel_scale))) * value

    return arcsecs


def getimage(fpath):
    '''
    Find a 2D image from a given file path. 
    If image does not have correct shape i.e. 2 dimensional it will raise an error

    :param fpath: Filepath towards fits image
    :type fpath: str
    :return: returns 2D image
    :rtype: numpy array

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

def beta_value(n,sigma,f_ul,noise = 0):

    '''

    Detection probability from
    http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf

    :param n:  Level above background to be considered a genuine detection
    :type n: float
    :param sigma: background standard deviation
    :type sigma: float
    :param f_ul: detected counts from measured photometry
    :type f_ul: float
    :param mean: Mean offset, defaults to 0
    :type mean: float, optional
    :return: Beta prime value which describes the confidence to which a source can be consider real
    :rtype: float

    '''

    from scipy.special import erf

    from numpy import sqrt

    source_flux = f_ul - noise

    if isinstance(source_flux,list):
        source_flux = [i if i>0 else 0 for i in source_flux]


    z = ((n*sigma) - (f_ul-noise))/(sqrt(2)*sigma)

    beta = 0.5 *(1 - erf(z))

    return beta

def f_ul(n,beta_p,sigma):

    '''
    Detection flux upper limit from
    http://web.ipac.caltech.edu/staff/fmasci/home/mystats/UpperLimits_FM2011.pdf

    :param n: Equivalnt sigma detection limit e.g. 3sigma
    :type n: float
    :param beta_p: Detection Probability, should not be set to below 3 for practicality
    :type beta_p: float
    :param sigma: Standard deviation of background values
    :type sigma: flost
    :return: Flux upper limit with some probabilistic meaning
    :rtype: float

    '''
    
    from numpy import sqrt
    from scipy.special import erfinv

    # beta_p = 1-beta
    f_ul = (n + (sqrt(2) * erfinv( (2*beta_p) - 1))) * sigma

    return f_ul



def calc_mag(flux, zp):
    '''
    Calculate magnitude of apoint source
    
    :param flux: Flux in counts per second measured from source
    :type flux: array-like
    :param zp: Zeropoint to place measurement on standard system
    :type zp: array-like
    :return: Magnitude of source on standard system
    :rtype: array-like

    '''


    import numpy as np

    # Iitial list with zeropoint
    if isinstance(flux,float):
        flux = [flux]

    mag_inst = np.array([-2.5*np.log10(i)+zp if i > 0.0 else np.nan for i in flux ])

    return mag_inst




def gauss_sigma2fwhm(image_params):

    '''
    Convert sigma value to full width half maximum for gaussian function

    :param image_params: Dictionary containing the key "sigma" with corrosponding value
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
    Convert fwhm to sigma value for gaussian funciton

    :param fwhm: full width half maximum
    :type fwhm: float
    :param image_params: Not needed for this function, defaults to None
    :type image_params: dict, optional
    :return: sigma value
    :rtype: float

    '''

    import numpy as np

    sigma= fwhm / (2*np.sqrt(2*np.log(2)))

    return sigma


def gauss_1d(x,A,x0,sigma):

    '''
    Returns 1 d gaussian function


    :param x: linear range of gaussian function
    :type x: numpy array
    :param A: Amplitude of gaussian function
    :type A: float
    :param x0: Center/maximum of gaussian function
    :type x0: float
    :param sigma: sigma/width of gaussian function
    :type sigma: float
    :return: Returns 1 dimention function with length equal to length of imput x array
    :rtype: numpy array

    '''

    import numpy as np

    G = A*np.exp(-(x-x0)**2/(2*sigma**2))

    return G

def gauss_2d(image, x0, y0, sky , A, image_params):

    '''
    Returns 2D gaussian function

    :param image: 2 dimentions grid to map Guassian on
    :type image: 2D numpy array
    :param x0: x-center of gaussian funciton
    :type x0: float
    :param y0: y-center of gaussian funciton
    :type y0: float
    :param sky: sky/offset of gaussian function
    :type sky: float
    :param A: Amplitude of gaussian function
    :type A: float
    :param image_params: Dictionary containing the key "sigma" with corrosponding value
    :type image_params: dict
    :return: 2D gaussian function with the same shape as image input
    :rtype: 2D numpy array

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

    Get FWHM rom Mofat function

    :param image_params: Dictionary containing 2 keys: "alpha" corrosponding to fitting width of moffat function and 'beta' describing the wings
    :type image_params: dict
    :return: Full width half maximum
    :rtype: float

    '''
    
    from numpy import sqrt

    alpha = image_params['alpha']
    beta = image_params['beta']

    fwhm  = 2 * alpha *  sqrt((2**(1/beta))-1)

    return fwhm


def moffat_2d(image, x0,y0, sky , A, image_params):

    '''
    Returns 2D moffat function
    https://www.ltam.lu/physique/astronomy/projects/star_prof/star_prof.html

    :param image: 2 dimentions grid to map Moffat on
    :type image: 2D numpy array
    :param x0: x-center of Moffat funciton
    :type x0: float
    :param y0: y-center of Moffat funciton
    :type y0: float
    :param sky: sky/offset of Moffat function
    :type sky: float
    :param A: Amplitude of Moffat function
    :type A: float
    :param image_params: Dictionary containing the keys "alpha" and "beta" with their corrosponding values
    :type image_params: dict
    :return: 2D Moffat function with the same shape as image input
    :rtype: 2D numpy array

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
    :type values: numpy array
    :param weights: weighs associated with values
    :type weights: numpy array
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
    :type arr: Numpy array
    :param new_shape: New shape with which to rebin array into
    :type new_shape: tuple
    :return: rebinned array
    :rtype: numpy array

    '''
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def weighted_median(data, weights):


    '''
    Return the weighted median of an array of values with given weights

    :param values: array of values
    :type values: numpy array
    :param weights: weighs associated with values
    :type weights: numpy array
    :return: weighted mean
    :rtype: tuple

    '''

    import numpy as np

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




# def pixel_correction(x,m):
    
#     '''
#     Correct the float nature of pixel position. NOT NEEDED FOR AUTOPHOT

#     '''

#     from numpy import ceil, floor

#     diff =  x - int(x)

#     if diff/m >= 0.5 or  diff ==0:
#         return ceil(x) - 0.5

#     elif diff/m < 0.5:
#         return floor(x)


def array_correction(x):

    '''

    Correct the float nature of pixel position used in numpy. 
    If a pixel position is more than halfway accros a pixel roundup, else round down

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
    Normalise array to between 0 and 1 while ignoring nans

    :param array: Array of values
    :type array: numpy  array
    :return: Normalised array
    :rtype: numpy  array

    '''

    from numpy import nanmin as np_min
    from numpy import nanmax as np_max

    norm_array = (array - np_min(array))/(np_max(array)-np_min(array))

    return norm_array


# def fin1d_2d_int_percent(count_percent,fwhm):

#     '''
#     Not needed for Autophot
#     '''

#     from scipy.optimize import least_squares
#     import numpy as np

#     from scipy.integrate import dblquad
#     sigma = fwhm/(2*np.sqrt(2*np.log(2)))

#     A = 1 / (2 * np.pi * sigma **2)
#     gauss = lambda y,x: A * np.exp( -1 * (((y)**2+(x)**2)/(2*sigma**2)))
#     fit = lambda x0: (count_percent - dblquad(gauss, -1*x0,x0,lambda x: -1*x0,lambda x: x0)[0])

#     r = least_squares(fit,x0 = 3)
#     return r.x[0]


def scale_roll(x,xc,m):
    '''
    Used in building PSF function. when shiting and aligning residual tables this functions trnalets pixel shifts between different images cutouts
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

def find_entry(new_df,date):

    row = new_df[new_df.Date == date]

    print(row.T)

def round_half(number):
    '''
    Round a number to half integer values i.ie +/-0.5
    
    :param number: Values to be rounded
    :type number: array-like
    :return: Values rounded to their nearest 0.5
    :rtype: array-like

    '''
    return round(number * 2) / 2

def to_latex(autophot_input,peak = None):

    
    '''
    Produce output table in latex-ready format
    
    :param autophot_input: Autophot input dictionary
    :type autophot_input: dict
    :param peak: Reference eposh of transient to set Phase column, defaults to None
    :type peak: float, optional
    :return: Latex ready table saved to output folder
    :rtype: N/A

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
