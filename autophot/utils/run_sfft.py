#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:11:45 2024

@author: seanbrennan
"""

# Import necessary modules

def run_sfft():
    import argparse
    import os
    # Import modules for SFFT operations
    from sfft.EasyCrowdedPacket import Easy_CrowdedPacket
    from sfft.EasySparsePacket import Easy_SparsePacket
    from sfft.CustomizedPacket import Customized_Packet


    # from functions import get_header
    
    parser = argparse.ArgumentParser(description='Perform photometry operations.')

    # Add command-line arguments for input files and options
    parser.add_argument('-sci', dest='sci', type=str, help='Filepath of the science FITS file.')
    parser.add_argument('-ref', dest='ref', type=str, help='Path to the reference FITS file.')
    parser.add_argument('-diff', dest='diff', type=str, default=None, help='Path to save the difference FITS file.')
    parser.add_argument('-mask', dest='mask', type=str, default=None, help='Path to boolen mask array.')
    parser.add_argument('-crowded', dest='crowded', action='store_true', help='Flag to use the crowded field packet.', 
                        default=True)
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # scienceHeader = get_header(args.sci)
    # templateHeader = get_header(args.ref)
         

    # SFFT configuration settings
    BACKEND_4SUBTRACT = 'Numpy'  # Computing backend: 'Cupy' for GPU, 'Numpy' for CPU.
    CUDA_DEVICE_4SUBTRACT = '0'  # CUDA device index (only used if 'Cupy' backend is selected).
    NUM_CPU_THREADS_4SUBTRACT = 1  # Number of CPU threads (only used if 'Numpy' backend is selected).
    
    ForceConv = 'REF'  # Convolution method: 'AUTO', 'REF', or 'SCI'.
                       # 'AUTO': Automatically choose the convolution method.
                       # 'REF': Convolve the reference image.
   # Whether to assume a constant photometric ratio between images.
    
    GAIN_KEY = 'GAIN'       # FITS header keyword for gain.
    SATUR_KEY = 'SATURATE'  # FITS header keyword for saturation.
    
    # Set file paths for the FITS images
    FITS_REF = args.ref    # Path to the reference FITS file.
    FITS_SCI = args.sci    # Path to the science FITS file.
    
    dirpath = os.path.dirname(FITS_SCI)
    
    if args.diff is None:
        FITS_DIFF = os.path.join(dirpath, 'diff_' + os.path.basename(FITS_SCI))  # Default path for difference FITS file.
    else:
        FITS_DIFF = args.diff  # User-specified path for difference FITS file.
        
        
    if args.mask is None:
        from astropy.io import fits
        mask = fits.getdata(args.mask).T

    else:
        mask = None
        
    from astropy.io import fits
    # get the date
    
    
    with fits.open(FITS_REF) as hdul:
        template_fwhm = hdul[0].header['FWHM']
    with fits.open(FITS_SCI) as hdul:
        science_fwhm = hdul[0].header['FWHM']
    
    import numpy as np
    
    FWHM = max(np.ceil(template_fwhm), np.ceil(science_fwhm))
    if FWHM % 2 == 0:
        FWHM += 1
        
    import os.path as pa
    
    
    detect_minarea = min(np.floor(template_fwhm), np.floor(science_fwhm)) * 3
    detect_maxarea = max(np.ceil(template_fwhm), np.ceil(science_fwhm)) * 10
    
    
    
    # detect_maxarea = detect_minarea * 2
    print(f'Min AREA: {detect_minarea}')
    print(f'Max AREA: {detect_maxarea}')
    # ForceConv = 'REF'               # FIXME {'AUTO', 'REF', 'SCI'}
    # Calculate Gaussian Kernel Half Width (GKerHW) based on Full Width at Half Maximum (FWHM)
    kernel_half_width = np.ceil(FWHM * 3)  # Kernel half width is typically 3 times FWHM.
    if kernel_half_width % 2 == 0:
        kernel_half_width += 1  # Ensure kernel_half_width is odd for symmetry.
    
    # Set kernel-to-FWHM ratio and polynomial orders for spatial variations
    kernel_fwhm_ratio = 2.5 # Ratio of kernel half width to FWHM (typically between 1.5 and 2.5).
    kernel_poly_order = 0  # Polynomial degree for kernel spatial variation {0, 1, 2, 3}.
    bg_poly_order = 0  # Polynomial degree for background spatial variation {0, 1, 2, 3}.
    constant_phot_ratio = True # Assume constant photometric ratio between images.
    prior_ban_mask = mask  # Mask to prevent prior contamination (optional, can be None).
    
    # Example of calling a customized subtraction packet
    if False:  # Change to 'True' to activate this block
        Customized_Packet.CP(
            FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=args.mask, FITS_mSCI=args.mask, 
            ForceConv=ForceConv, GKerHW=kernel_half_width, FITS_DIFF=FITS_DIFF, FITS_Solution=None, 
            KerPolyOrder=kernel_poly_order, BGPolyOrder=bg_poly_order, ConstPhotRatio=constant_phot_ratio, 
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, 
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT
        )
    
    # Use the appropriate subtraction method based on the field type (crowded or sparse)
    elif args.crowded:  # Crowded field subtraction
        # Perform subtraction for crowded fields using Easy_CrowdedPacket
        diff_image, prep_data = Easy_CrowdedPacket.ECP(
            FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, ForceConv=ForceConv, 
            GKerHW=kernel_half_width, KerHWRatio=kernel_fwhm_ratio, KerHWLimit=(2, 20), 
            KerPolyOrder=kernel_poly_order, BGPolyOrder=bg_poly_order, ConstPhotRatio=constant_phot_ratio, 
            MaskSatContam=True, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE=0.0, 
            BACK_SIZE=64, BACK_FILTERSIZE=6, DETECT_THRESH=5.0, DETECT_MINAREA=detect_minarea, 
            DETECT_MAXAREA=detect_maxarea, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', ONLY_FLAGS=None, 
            BoundarySIZE=30.0, BACK_SIZE_SUPER=128, StarExt_iter=5, PriorBanMask=prior_ban_mask, 
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, 
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT
        )[:2]
    
    else:  # Sparse field subtraction
        # Set rejection thresholds for variable stars
        coarse_var_rejection = True  # Perform coarse variable rejection
        coarse_var_reject_thresh = 0.1  # Magnitude threshold for coarse rejection
        elaborate_var_rejection = True  # Perform elaborate variable rejection
        
        # Perform subtraction for sparse fields using Easy_SparsePacket
        detect_maxarea = 0
        diff_image, SFFTPrepDict = Easy_SparsePacket.ESP(
            FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, ForceConv=ForceConv, 
            GKerHW=kernel_half_width, KerHWRatio=kernel_fwhm_ratio, KerHWLimit=(2, 20), KerPolyOrder=kernel_poly_order, 
            BGPolyOrder=bg_poly_order, ConstPhotRatio=constant_phot_ratio, MaskSatContam=False, 
            GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE=0.0, 
            BACK_SIZE=64, BACK_FILTERSIZE=6, DETECT_THRESH=5.0, DETECT_MINAREA=detect_minarea, 
            DETECT_MAXAREA=detect_maxarea, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', 
            ONLY_FLAGS=[0], BoundarySIZE=10.0, XY_PriorSelect=None, Hough_MINFR=0.1, 
            Hough_PeakClip=0.7, BeltHW=0.2, PointSource_MINELLIP=0.5, MatchTol=1, 
            MatchTolFactor=3.0, COARSE_VAR_REJECTION=coarse_var_rejection, 
            CVREJ_MAGD_THRESH=coarse_var_reject_thresh, ELABO_VAR_REJECTION=elaborate_var_rejection, 
            EVREJ_RATIO_THREH=5.0, EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=5, 
            XY_PriorBan=None, PostAnomalyCheck=True, PAC_RATIO_THRESH=5.0, 
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, 
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT
        )[:2]

                
        

        
        if 1:
            
            from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
            # open ./output_data/*.fittedPix.fits to check the pixels used in fitting
            # * pixels not used have been replaced by random noise.
            
            # for Science
            PixA_SCI = SFFTPrepDict['PixA_SCI']
            sky, skysig = SkyLevel_Estimator.SLE(PixA_obj=PixA_SCI)
            PixA_NOSCI = np.random.normal(sky, skysig, PixA_SCI.shape)
            
            FITS_FittedSCI =  dirpath + '/%s.fittedPix.fits' %(pa.basename(FITS_SCI)[:-5])
            with fits.open(FITS_SCI) as hdl:
                PixA_SCI = hdl[0].data.T
                NonActive = ~SFFTPrepDict['Active-Mask']
                PixA_SCI[NonActive] = PixA_NOSCI[NonActive]
                hdl[0].data[:, :] = PixA_SCI.T
                hdl.writeto(FITS_FittedSCI, overwrite=True)
            
            # for Reference
            PixA_REF = SFFTPrepDict['PixA_REF']
            sky, skysig = SkyLevel_Estimator.SLE(PixA_obj=PixA_REF)
            PixA_NOREF = np.random.normal(sky, skysig, PixA_REF.shape)
            
            FITS_FittedREF = dirpath + '/%s.fittedPix.fits' %(pa.basename(FITS_REF)[:-5])
            with fits.open(FITS_REF) as hdl:
                PixA_REF = hdl[0].data.T
                NonActive = ~SFFTPrepDict['Active-Mask']
                PixA_REF[NonActive] = PixA_NOREF[NonActive]
                hdl[0].data[:, :] = PixA_REF.T
                hdl.writeto(FITS_FittedREF, overwrite=True)
                
                
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            
            # open ./output_data/varcheck.pdf to see the figure
            # one may deactivate any variabvle rejection and generate 
            # this figure again to see the effect of our rejection.
            # (that is, setting COARSE_VAR_REJECTION = False and ELABO_VAR_REJECTION = False)
            CVREJ_MAGD_THRESH = 0.12
            AstSS = SFFTPrepDict['SExCatalog-SubSource']
            
            plt.figure(figsize=(8, 4))
            ax = plt.subplot(111)
            xdata = AstSS['MAG_AUTO_REF']
            exdata = AstSS['MAGERR_AUTO_REF']
            ydata = AstSS['MAG_AUTO_SCI'] - AstSS['MAG_AUTO_REF']
            eydata = AstSS['MAGERR_AUTO_SCI']
            
            ax.errorbar(xdata, ydata, xerr=exdata, yerr=eydata, 
                fmt='o', markersize=2.5, color='black', mfc='#EE3277',
                capsize=2.5, elinewidth=0.5, markeredgewidth=0.1)
            
            m = np.median(ydata) 
            ml, mu = m - CVREJ_MAGD_THRESH, m + CVREJ_MAGD_THRESH
            ax.hlines(y=[m, ml, mu], xmin=xdata.min(), xmax=xdata.max(), 
                linestyle='--', zorder=3, color='#1D90FF')
            ax.set_xlabel(r'MAG_AUTO (REF)')
            ax.set_ylabel(r'MAG_AUTO (SCI) - MAG_AUTO (REF)')
            plt.savefig(dirpath + '/varcheck.pdf', dpi=300)
            plt.close()
            
    return

if __name__ == "__main__":
    run_sfft()
