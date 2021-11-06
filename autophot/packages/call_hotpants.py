#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:10:34 2019

@author: seanbrennan
"""

def HOTterPANTS(file,template,syntax = None,psf = None):

    import subprocess
    import os
    import sys
    import numpy as np
    from pathlib import Path
    import signal
    import time
    from autophot.packages.functions import  getheader,getimage
    from astropy.io import fits

    try:

        convolve_image = False
        smooth_template = False
        # Get file extension and template data
        fname_ext = Path(file).suffix

        # Open image and template
        file_image     = getimage(file)
        template_image = getimage(template)

        header = getheader(file)

        # some values set to 1e-10 during image alignment - ignore these
        check_values_image = file_image[abs(file_image)>1e-10]

        check_values_template = template_image[abs(template_image)>1e-10]

        if syntax['remove_sat']:

            image_max = [np.nanmax(check_values_image) if np.nanmax(check_values_image) < 2**16 else -500 + 2**16][0]

            template_max = [np.nanmax(check_values_template) if np.nanmax(check_values_template) < 2**16 else  2**16][0]
        else:
            image_max = np.nanmax(check_values_image)
            template_max = np.nanmax(np.nanmax(check_values_template))


        t_header = getheader(template)

        image_FWHM = syntax['fwhm']

        # try:


        #     template_FWHM = t_header['FWHM']

        #     if image_FWHM < template_FWHM:
        #         # print('Need to sharpen template')
        #         convolve_image = True
        #     elif image_FWHM < template_FWHM:
        #         # print('Need to adjust fitting')
        #         # smooth_template = True
        # except:
        #     pass


        # Get filename for saving
        base = os.path.splitext(os.path.basename(file))[0]

        # Location of executable for hotpants
        exe = syntax['hotpants_exe_loc']

# =============================================================================
# Argurments to send to HOTPANTS process - list of tuples
# =============================================================================

        # Arguments to pass to HOTPANTS
        include_args = [
                # Input image
                        ('-inim',   str(file)),
                # Template Image
                        ('-tmplim', str(template)),
                # Output image name
                        ('-outim',  str(file.replace(fname_ext,'_subtraction'+fname_ext))),
                # Image lower values
                        ('-il',     str(np.nanmin(check_values_image))),
                # # Template lower values
                        ('-tl',     str(np.nanmin(check_values_template))),
                # Template upper
                        ('-tu',     str(template_max)),
                # Image upper
                        ('-iu',     str(image_max)),
                # Image gain
                        # ('-ig',     str(syntax['gain'])),
                # Template gain
                #         ('-tg',     str(t_header['gain'])),
                # Normalise to image[i]
                        ('-n',      'i'),
                # Background order fitting
                        # ('-bgo' ,   str(1)),


                        # ('-r' , str(syntax['fwhm']))
                        ]

        args= [str(exe)]






        # print(include_args)


        # print(args)#
        # if convolve_image:
        #     print('Convolving subtracted image with PSF')

        #     from astropy.convolution import convolve

        #     subtraction_image = getimage(file)

        #     convoled_subtraction_image = convolve(subtraction_image,kernel = psf,
        #                                           normalize_kernel = True)

        #     header['CONVOL'] = (True,'Convoled with PSF')

        #     fits.writeto(file,
        #                   convoled_subtraction_image.astype(np.single),
        #                   header,
        #                   overwrite = True,
        #                   output_verify = 'silentfix+ignore')

#



        if smooth_template:
            print('Adjusting fitting to smooth template')
            fwhm_match = np.sqrt(image_FWHM**2 + template_FWHM**2)
            sigma_match = fwhm_match/(2 * np.sqrt(2 * np.log(2)))

            add_flag = ('-ng', '3 6 %.3f 4 %.3f 2 %.3f' % (0.5*sigma_match, sigma_match, 2*sigma_match))

            include_args= include_args + [add_flag]

        # print(args)
        for i in include_args:
            args[0] += ' ' + i[0] + ' ' + i[1]



# =============================================================================
# Call subprocess using executable location and option prasers
# =============================================================================

        start = time.time()

        # print(args)

        with  open(syntax['write_dir'] + base + '_HOTterPANTS.txt', 'w')  as FNULL:

            pro = subprocess.Popen(args,shell=True, stdout=FNULL, stderr=FNULL)

            # Timeout
            pro.wait(syntax['hotpants_timeout'])

            try:
                # Try to kill process to avoid memory errors / hanging process
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                print('HOTPANTS PID killed')
            except:
                pass

        print('HOTPANTS finished: %ss' % round(time.time() - start) )

        output_fpath = str(file.replace(fname_ext,'_subtraction'+fname_ext))

        # check if file is there - if so return filepath if not return original filepath
        if os.path.isfile(output_fpath):

            file_size = os.path.getsize(str(file.replace(fname_ext,'_subtraction'+fname_ext)))

            if file_size == 0:
                print('File was created but nothing written')
                print("FILE CHECK FAILURE - Return original filepath" )
                return file
            else:


                print('Subtraction saved as %s' % os.path.splitext(os.path.basename(file.replace(fname_ext,'_subtraction'+fname_ext)))[0])

                # if convolve_image:
                #     print('Convolving subtracted image with PSF')

                #     from astropy.convolution import convolve

                #     subtraction_image = getimage(output_fpath)

                #     convoled_subtraction_image = convolve(subtraction_image,kernel = psf,
                #                                           normalize_kernel = True)

                #     header['CONVOL'] = (True,'Convoled with PSF')

                #     fits.writeto(output_fpath,
                #                   convoled_subtraction_image.astype(np.single),
                #                   header,
                #                   overwrite = True,
                #                   output_verify = 'silentfix+ignore')



                return output_fpath










        if not os.path.isfile(output_fpath):

            print('File was not created')
            print('>ILE CHECK FAILURE - Return orifinal filepath ')

            return file

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno,e)
        try:
                # Try to kill process to avoid memory errors / hanging process
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
            print('HOTPANTS PID killed')
        except:
            pass

        return file
