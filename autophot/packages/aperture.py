def ap_phot(positions,data, radius ,r_in = None,r_out= None):

    """This is a summary of the aperture packge. This a robust aperture photometry packages for use in autophot


    :param positions: list of tuples containing (x,y) positions of object
    :type positions: tuple

    :param r_in: inner radius of background annulus
    :type r_in: float

    :param r_out: outer radius of background annulus
    :type r_out: float

    :return: Returns lost of aperature measurements
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

        apertures = CircularAperture(positions, r=radius)
        annulus_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        annulus_masks = annulus_apertures.to_mask(method='center')

        if r_out >= data.shape[0] or r_out > data.shape[1]:
            print('Error - Apphot - Annulus size greater than image size')

        bkg_median = []

        if not isinstance(annulus_masks,list):
            annulus_masks = list(annulus_masks)


        for mask in annulus_masks:

            annulus_data = mask.multiply(data)

            if annulus_data is None:

                if np.isnan(np.sum(annulus_data)):
                    print('Error - annulus data is nan')
                print('Error - Annulus == None setting to zero')
                annulus_data = np.zeros((int(2*r_out+1),int(2*r_out+1)))


            annulus_data_1d = annulus_data[mask.data > 0]

            annulus_data_1d_nonan = annulus_data_1d[~np.isnan(annulus_data_1d)]

            _, median_sigclip,_ = sigma_clipped_stats(annulus_data_1d_nonan,

                                                       cenfunc = np.nanmedian,
                                                       stdfunc = np.nanstd)
            bkg_median.append(median_sigclip)

        bkg_median = np.array(bkg_median)
        phot = aperture_photometry(data, apertures)
        phot = phot.to_pandas()

        phot['annulus_median'] = bkg_median
        phot['aper_bkg'] = bkg_median * np.pi * radius**2


        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']

        aperture_sum = np.array(phot['aper_sum_bkgsub'])
        bkg_sum = np.array(phot['annulus_median'])

        aperture_sum[ aperture_sum<=0] = 0

    except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)



    return aperture_sum,bkg_sum