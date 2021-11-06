#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:53:13 2019

@author: seanbrennan
"""


def SNR(R_star,R_sky,exp_t,RN,radius,G,D):
    try:

        '''
        S/N eqaution

        R_star = count raate from star [e / sec]
        R_sky = count rate from sky annululs [median] [ e /second / pixel ]
        exp_t = exposure time [s]
        radius = radius of aperture [pixels]
        G = gain [e /ADU]
        RN = read noise
        D = dark current [e / pixel / second]
        '''

        import numpy as np
        import os
        import sys
        import warnings

        with warnings.catch_warnings():
            # Ignore  Runtime warnings
            warnings.simplefilter('ignore')

            G = float(G)

            counts_source = R_star * exp_t

            star_shot_2 = R_star * exp_t

            sky_shot_2 = R_sky * np.pi * (radius ** 2) * exp_t


            read_noise_2 = ((RN**2) + (G/2)**2) * np.pi * ( radius ** 2)

            dark_noise_2 =  D * exp_t * np.pi * (radius ** 2)

            SNR = counts_source / np.sqrt((star_shot_2 + sky_shot_2 +read_noise_2  + dark_noise_2))

            return SNR

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        return np.nan



def sigma_mag_err(SNR):
    try:

        """
        Magnitude error due to SNR

        """
        import numpy as np
        import os
        import sys

        # If SNR is given as single value and not a list or array
        if isinstance(SNR,int) or isinstance(SNR,float):
            if SNR <= 0:
                return 0
            sigma_err = np.array([2.5 * np.log10(1 + 1/SNR)])
            return sigma_err[0]


        else:
            SNR = np.array(SNR)

        # Remove SNR values if less than zero, replace with zero i.e source not detected
        SNR_cleaned = [i if i>0 else 0 for i in SNR]

#        # If list with single value
#        if len(SNR_cleaned)==1:
#            SNR_cleaned = SNR_cleaned[0]


        if isinstance(SNR_cleaned,float):
#            print('here')
            SNR_cleaned = np.array(SNR_cleaned)
#        print(SNR_cleaned)



        sigma_err = np.array([2.5 * np.log10(1 + 1/snr) if snr>0 else np.nan for snr in SNR_cleaned])

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)


    return sigma_err

