def Rayleigh_extinction(lam,h):
    '''
    
    Extinction due to Rayliehg scattering taken from
    
    https://www.degruyter.com/document/doi/10.1515/astro-2017-0046/html
    
    :param lam: Wavelegth in Angstrom
    :type lam: float 
    :param h: Altiude of observatory/telescope in Km
    :type h: float
    :return: Airmass extinction due to Rayleigh scattering in units of mag per unit airmass 
    :rtype: float

    '''
    
    from numpy import exp
    
    lam = lam*0.0001
    
    
    ns = 0.23465 + (107.6/(146-lam ** -2)) + (0.93161/(41 - lam ** -2))

    aR =  0.0094977 * lam ** -4 * ns**2 * (exp(-h/7.996))
    
    return aR


def Ozone_extinction(lam):
    
    '''
    Extinction due to Ozone and water absorption in the atmosphere.
    
    :param lam: Wavelegth in Ansgtrom
    :type lam: float
    :return: Airmass extinction due to Rayleigh scattering in units of mag per unit airmass 
    :rtype: float

    '''
    
    import os 
    import numpy as np
    from scipy.interpolate import interp1d
    from functools import reduce
   
    filepath = os.path.dirname(os.path.abspath(__file__))
    
    ozone_coeffs = reduce(os.path.join,['/'.join(filepath.split('/')[:-1]),'databases/extinction','ozone_coeff.txt'])
    # print(filepath)

    kappa_ozone = np.genfromtxt(ozone_coeffs)
    
    kappa_lam = np.array([i[0] for i in kappa_ozone])
    kappa_ext_percm = np.array([i[1] for i in kappa_ozone])

    xnew = []
    ynew = []
    
    for i in range(len(kappa_lam)):
        if kappa_lam[i] not in xnew:
            if kappa_lam[i]<1000:
                x = kappa_lam[i]*10
            else:
                x = kappa_lam[i]
            xnew.append(x)
            ynew.append(kappa_ext_percm[i])
            
        
    kappa_lam = np.array(xnew)
    kappa_ext_percm=np.array(ynew)
    idx = np.argsort(kappa_lam)
    
    kappa_lam= kappa_lam[idx]
    kappa_ext_percm= kappa_ext_percm[idx]
        
    if lam<kappa_lam.min() or lam>kappa_lam.max():
        print('Wavelength %d outside Ozone Coeff range [%d,%d]' % (lam,kappa_lam.min(),kappa_lam.max() ))
        return 0 
    
    kappa_lam= kappa_lam[idx]
    kappa_ext_percm= kappa_ext_percm[idx]
    
    kappa_int = interp1d(kappa_lam, kappa_ext_percm,kind='cubic')
    
    # https://ozonewatch.gsfc.nasa.gov/facts/dobson_SH.html
    # Thickness of ozone layer at STP (3mm)
    T_oz = 0.3
    
    a0 = 1.11 * T_oz * kappa_int(lam)
    
    return a0
    
    
def Aerosol_extinction(lam,h,A0,H0,bs):
    
    '''
    Extinction due to particulates in the Air. 
    
    :param lam: Wavelegth in Ansgtrom
    :type lam: float 
    :param h: Altiude of observatory in Km
    :type h: float
    :return: Airmass extinction in mag per unit airmass
    :rtype: float

    '''
    
    from numpy import exp
    
    lam = lam*0.0001
    
    aaer = A0 * (lam**bs) * exp(-h/H0)
    
    return aaer


    
    
def X(secz):
    
    '''
    Airmass equation taken from taken from Photometric calibration cookbook
    
    # http://star-www.rl.ac.uk/star/docs/sc6.htx/sc6.html
    
    :param secz: secant of zenith angle in degress
    :type secz: float
    :return: Airmass
    :rtype: float

    '''
    
    X = secz - 0.0018167 * (secz-1) - 0.002875 * ((secz-1) **2) - 0.0008083*((secz-1)**3)
    
    return X



def find_airmass_extinction(extinction_dictionary,
                            header,
                            image_filter,
                            airmass_key):
    
    
    '''
    
    Compute airmass correction for transient observation in a given filter at a specific airmass
    .. math::
        m_{\lambda,0} = m_{\lambda} + X*\kappa_\lambda
        :label: euler
    
    :param airmass: Airmass at which transient is observed
    :type airmass: float
    :param extinction_dictionary: Dictionary containing extinction in specific filters 
    :type extinction_dictionary: dict
    :param image_filter: Filter used 
    :type image_filter: str
    :param airmass_key: Key corrosponding to airmass in header file
    :type airmass_key: str
    :return: Airmass correction
    :rtype: Float
    
    '''
    
    # TODO: change input parameters for simplicity
    
    import logging
    
    logger = logging.getLogger(__name__)

    airmass = header[airmass_key]
    
    kappa_filter = extinction_dictionary['ex_'+image_filter]
    
    airmass_correction = airmass * kappa_filter

    logger.info('Airmass extinction correction: %.3f' % airmass_correction )

    return airmass_correction