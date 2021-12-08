#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def removewcs(parent_header, delete_keys = False):
    '''

    Function to erase all World Coordinate System (WCS) keywords from a fits
    header. Values are deleted/adjusted if the fall into the list below:
    
    
    *CD1_1*, *CD1_2*, *CD2_1*, *CD2_2*, *CRVAL1*, *CRVAL2*, *CRPIX1*, *CRPIX2*,
    *CUNIT1*, *CUNIT2*, *CTYPE1*, *CTYPE2*, *WCSAXES*, *EQUINOX*, *LONPOLE*,
    *LATPOLE*, *CDELT1*, *CDELT2*, *A_ORDER*, *A_0_0*, *A_0_1*, *A_0_2*, *A_1_0*,
    *A_1_1*, *A_2_0*, *B_ORDER*, *B_0_0*, *B_0_1*, *B_0_2*, *B_1_0*, *B_1_1*,
    *B_2_0*, *AP_ORDER*, *AP_0_0*, *AP_0_1*, *AP_0_2*, *AP_1_0*, *AP_1_1*,
    *AP_2_0*, *BP_ORDER*, *BP_0_0*, *BP_0_1*, *BP_0_2*, *BP_1_0*, *BP_1_1*,
    *BP_2_0*, *PROJP1*, *PROJP3*, *RADECSYS*, *PV1_1*, *PV1_2*, *PV2_1*, *PV2_2*,
    *LTV1*, *LTV2*, *LTM1_1*, *LTM2_2*, *PC1_1*, *PC1_2*, *PC2_1*, *PC2_2*,
    *RADESYS*
    
    
    :param parent_header: Header object containing WCS information which we want to erase
    :type parent_header: *FITS* Header object
    :param delete_keys: If True, delete these keys, else concatenate "*_*" onto them, defaults to False
    :type delete_keys: bool, optional
    :return: Corrected header file will corrected/deleted WCS keys
    :rtype: *FITS* Header object

    '''


    import logging

    logger = logging.getLogger(__name__)
    logger.info('Removing any pre-existing WCS keys ')

    try:
        if parent_header['UPWCS']:
            print('Removed UPWCS key')
            del parent_header['UPWCS']

    except:
        pass

    keywords = ['CD1_1','CD1_2', 'CD2_1','CD2_2', 'CRVAL1','CRVAL2', 'CRPIX1','CRPIX2',
                'CUNIT1','CUNIT2', 'CTYPE1','CTYPE2', 'WCSAXES','EQUINOX', 'LONPOLE','LATPOLE',
                'CDELT1','CDELT2', 'A_ORDER', 'A_0_0', 'A_0_1','A_0_2', 'A_1_0','A_1_1',
                'A_2_0', 'B_ORDER', 'B_0_0','B_0_1', 'B_0_2','B_1_0', 'B_1_1','B_2_0',
                'AP_ORDER', 'AP_0_0','AP_0_1', 'AP_0_2','AP_1_0', 'AP_1_1','AP_2_0',
                'BP_ORDER', 'BP_0_0','BP_0_1', 'BP_0_2','BP_1_0', 'BP_1_1','BP_2_0',
                'PROJP1','PROJP3', 'RADECSYS', 'PV1_1','PV1_2', 'PV2_1','PV2_2', 'LTV1','LTV2',
                'LTM1_1','LTM2_2', 'PC1_1','PC1_2', 'PC2_1','PC2_2', 'RADESYS']

    for i in keywords:

        if i.replace(i[0],'_') in parent_header:
            del parent_header[i.replace(i[0],'_')]
        try:
            if delete_keys == True:
                try:
                    del parent_header[i]
                except:
                    continue
            else:
                parent_header.rename_keyword(i,i.replace(i[0],'_'))
                continue

        except Exception as e:
            logger.exception(e)
            pass


    return parent_header



def updatewcs(parent_header,wcs_fits):
    '''
    
    Update *parent_header* with WCS keys from *wcs_fits* header
    
    :param parent_header: Header file with outdated or missing WCS keywords and  values
    :type parent_header: Header Object
    :param wcs_fits: Header file with corrected WCS keywords and values
    :type wcs_fits: Header Object
    :return: Parent Header with Updated WCS keywords
    :rtype:  Header Object

    '''

    import logging

    logger = logging.getLogger(__name__)
    logger.info('Updating WCS keys with new values')

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
            parent_header[i] = ((wcs_fits[i]),'WCS by APT')
        except:
            continue

    return parent_header

