#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
Integration between standard fits file/format and output of Dustin Langs
Astrometry.Net algorithim
'''


def removewcs(parent_fits, delete_keys = False):

    '''
    - removewcs:

    rename pre-existing keywords from header hdu given in keywords list.

    parameters:
        - a: header information given via fits.open
        - delete_keys: delete this keywords and thier values rather than renaming

    retruns:
        - a: new header hdu with/without new keywords

    '''

    import logging

    logger = logging.getLogger(__name__)
    logger.info('Removing any pre-existing WCS keys ')

    try:
        if parent_fits['UPWCS']:
            print('Removed UPWCS key')
            del parent_fits['UPWCS']

    except:
        pass

    keywords = ['CD1_1','CD1_2',
                'CD2_1','CD2_2',
                'CRVAL1','CRVAL2',
                'CRPIX1','CRPIX2',
                'CUNIT1','CUNIT2',
                'CTYPE1','CTYPE2',
                'WCSAXES','EQUINOX',
                'LONPOLE','LATPOLE',
                'CDELT1','CDELT2',
                'A_ORDER',
                'A_0_0',
                'A_0_1','A_0_2',
                'A_1_0','A_1_1',
                'A_2_0',
                'B_ORDER',
                'B_0_0','B_0_1',
                'B_0_2','B_1_0',
                'B_1_1','B_2_0',
                'AP_ORDER',
                'AP_0_0','AP_0_1',
                'AP_0_2','AP_1_0',
                'AP_1_1','AP_2_0',
                'BP_ORDER',
                'BP_0_0','BP_0_1',
                'BP_0_2','BP_1_0',
                'BP_1_1','BP_2_0',
                'PROJP1','PROJP3',
                'RADECSYS',
                'PV1_1','PV1_2',
                'PV2_1','PV2_2',
                'LTV1','LTV2',
                'LTM1_1','LTM2_2',
                'PC1_1','PC1_2',
                'PC2_1','PC2_2',
                'RADESYS'
                           ]

    for i in keywords:

        if i.replace(i[0],'_') in parent_fits:
            del parent_fits[i.replace(i[0],'_')]
        try:
            if delete_keys == True:
                try:
                    del parent_fits[i]
                except:
                    continue
            else:
                parent_fits.rename_keyword(i,i.replace(i[0],'_'))
                continue

        except Exception as e:
            logger.exception(e)
            pass


    return parent_fits



def updatewcs(parent_fits,wcs_fits):

    '''

    - updatewcs(a,b):

    Update header fits file 'a' with the header information with header fits file 'b'

    paramters:
    - a: header information given via fits.open
    - b: header information  from astrometry.net query

    retruns:
    - a: header hdu with update keywords
    '''

    import logging

    logger = logging.getLogger(__name__)
    logger.info('Updating WCS keys')

    keywords = ['AP_0_0', 'AP_0_1', 'AP_0_2', 'AP_1_0', 'AP_1_1', 'AP_2_0',
       'AP_ORDER', 'A_0_0', 'A_0_1', 'A_0_2', 'A_1_0', 'A_1_1', 'A_2_0',
       'A_ORDER', 'BP_0_0', 'BP_0_1', 'BP_0_2', 'BP_1_0', 'BP_1_1',
       'BP_2_0', 'BP_ORDER', 'B_0_0', 'B_0_1', 'B_0_2', 'B_1_0', 'B_1_1',
       'B_2_0', 'B_ORDER', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CDELT1',
       'CDELT2', 'CRPIX1', 'CRPIX1', 'CRPIX2', 'CRPIX2', 'CRVAL1',
       'CRVAL1', 'CRVAL2', 'CRVAL2', 'CTYPE1', 'CTYPE1', 'CTYPE2',
       'CTYPE2', 'CUNIT1', 'CUNIT1', 'CUNIT2', 'CUNIT2', 'EQUINOX',
       'LATPOLE', 'LATPOLE', 'LONPOLE', 'LONPOLE', 'MJDREF', 'PC1_1',
       'PC1_2', 'PC2_1', 'PC2_2', 'PROJP1', 'PROJP3', 'PV1_1', 'PV1_2',
       'PV2_1', 'PV2_2', 'RADECSYS', 'RADESYS', 'WAT0_001', 'WAT1_001',
       'WAT2_001', 'WCSAXES', 'WCSAXES', 'WCSDIM']

    for i in keywords:
        try:
            parent_fits[i] = ((wcs_fits[i]),'WCS by APT')
        except:
            continue

    return parent_fits

