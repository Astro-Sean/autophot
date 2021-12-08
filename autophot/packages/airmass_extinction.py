def Rayleigh_extinction(lam,h):
    
    '''
    Approximation of extinction due to Rayleigh scattering. The extinction can be
    approximated by:
    
    .. math::
    \alpha(\lambda,H) = 0.0094977 \times \lambda^{-4} * n_s ^2(\lambda) \times
    e^{\frac{-H}{7.996}}
    
    where\ n_s(\lambda) = 0.23465 + \frac{107.6}{146 - \lambda^{-2}} +
    \frac{0.93161}{41 - \lambda^{-2}}
    
    
    
    `Credit: E. Pakštienė and J. E. Solheim 2003
    <https://www.degruyter.com/document/doi/10.1515/astro-2017-0046/html>`_
    
    
    :param lam: Wavelength of the image filter in :math:`\AA`. AutoPHOT using the
    effective wavelength as an approximated
    :type lam: float 
    :param h: Altitude of observatory/telescope in Km
    :type h: float
    :return: Airmass extinction due to Rayleigh scattering in units of :math:`mag /
    unit\ airmass`.
    :rtype: float

    '''
    
    from numpy import exp
    
    lam = lam*0.0001
    
    
    ns = 0.23465 + (107.6/(146-lam ** -2)) + (0.93161/(41 - lam ** -2))

    aR =  0.0094977 * lam ** -4 * ns**2 * (exp(-h/7.996))
    
    return aR


def Ozone_extinction(lam):
    
    '''
    Approximate extinction due to Ozone and water absorption in the atmosphere.
    This extinction can be approximated by:
    
    
    .. math::
    \alpha(\lambda) = 1.09 \times T \times \kappa(\lambda) 
    
    where Tis the total thickness of the ozone layer at the standard temperature
    :math:`0^\circ C` and pressure (1 atm) above the observatory. This values is
    assumed to be `3mm <https://ozonewatch.gsfc.nasa.gov/facts/dobson_SH.html>`_
    and :math:`\kappa(\lambda)` k(λ)is the ozone absorption coefficient.
    
    
    `Credit: E. Pakštienė and J. E. Solheim 2003
    <https://www.degruyter.com/document/doi/10.1515/astro-2017-0046/html>`_
    
    :param lam: Wavelength of the image filter in :math:`\AA`. AutoPHOT using the
    effective wavelength as an approximated
    :type lam: float 
    :return: Airmass extinction due to Ozone and water absorption in units of
    :math:`mag / unit\  airmass`.
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
    Extinction due to particulates in the Air. This is a guess at the effect of
    aerosols at a location. This approximation is given by:
    
    .. math::
    \alpha_{aer}(\lambda,H) =  A_{0} \times \lambda^{b_s} \times
    e^{\frac{-h}{H_0}}
    
    where :math:`A_0` is the same extinction for :math:`\lambda=1~\mu m` and
    :math:`b_s` is a coefficient dependent on the size of aerosol particles and
    their size distribution and :math:`H_0` is the scale height.
    
    
    :param lam: Wavelength of the image filter in :math:`\AA`. AutoPHOT using the
    effective wavelength as an approximated
    :type lam: float 
    :param h: Altitude of observatory/telescope in Km
    :type h: float
    :param A0: Normalised extinction at :math:`\lambda=1~\mu m` 
    :type A0: float
    :param H0: Scale height in Km
    :type H0: foat
    :param bs: A coefficient dependent on the size of aerosol particles and their
    size distribution
    :type bs: float
    :return: Airmass extinction due to particulates in the air in units of
    :math:`mag /unit\ airmass`.
    :rtype: float

    '''
    
    from numpy import exp
    
    lam = lam*0.0001
    
    aaer = A0 * (lam**bs) * exp(-h/H0)
    
    return aaer


    
    
def X(secz):
    
    '''
    Calculate the path length through the atmosphere, otherwise known as the air
    mass. For small zenith angles, :math:`X = sec(z)`  is a reasonable
    approximation, but as z increases, refraction effects, curvature of the
    atmosphere and variations of air density with height can become important. We
    use a more refined definition given by:
    
    .. math::
    X = sec(z) − 0.0018167 ( sec(z) − 1) − 0.002875 ( sec(z) − 1 )^2 − 0.0008083 (
    sec(z) − 1 )^3
    
    Although this approximate is valid, general the airmass term is included in an
    images header file.
    
    `Credit: CCD Handbook <http://star-www.rl.ac.uk/star/docs/sc6.htx/sc6.html>`_
    
    
    :param secz: Secant of zenith angle in degrees.
    :type secz: float
    :return: Air mass 
    :rtype: float



    '''
    
    X = secz - 0.0018167 * (secz-1) - 0.002875 * ((secz-1) **2) - 0.0008083*((secz-1)**3)
    
    return X



def find_airmass_extinction(extinction_dictionary,
                            header,
                            image_filter,
                            airmass_key):
    
    
    '''
    
    Compute airmass correction for transient observation in a given filter at a
    specific airmass using the following equation:
    
    .. math::
    m_{\lambda,0} = m_{\lambda} + X \times \kappa_\lambda
    
    where :math:`m_{\lambda,0}` is the extinction corrected magnitude,
    :math:`m_{\lambda}` us the original measurement, uncorrected for airmass
    extinction, :math:`X` is the airmass of the observation, and
    :math:`\kappa_\lambda` is the extinction per unit wavelength at the wavelength
    of the observaiton.
    
    
    :param airmass: Airmass at which transient is observed.
    :type airmass: float
    :param extinction_dictionary: Dictionary containing extinction in specific
    filters. This is created using the *call_datacheck* package in AutoPHOT.
    :type extinction_dictionary: dict
    :param image_filter: Filter used in the observation
    :type image_filter: str
    :param airmass_key: Key corresponding to airmass given in header file.
    :type airmass_key: str
    :return: Airmass correction i.e. :math:`X \times \kappa_\lambda` 
    :rtype: float
        
    '''
    
    # TODO: change input parameters for simplicity
    import logging
    
    logger = logging.getLogger(__name__)

    airmass = header[airmass_key]
    
    kappa_filter = extinction_dictionary['ex_'+image_filter]
    
    airmass_correction = airmass * kappa_filter

    logger.info('Airmass extinction correction: %.3f' % airmass_correction )

    return airmass_correction