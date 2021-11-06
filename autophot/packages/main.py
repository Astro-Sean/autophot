
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

- Main operations of AutoPHoT

Inputs:

    - object_info - yaml file:
        TNS repsonse for target [dictionary]
    - syntax - yaml file:
        yaml file input [dictionary]
    - fpath - str:
        file path [str]

Outputs:
    - output:
        dictionary out output photmetric data:
            fname:
                Orreginal file path
            telescope:
                Telescope used - needed for plotting
            mjd:
                Modified julian date
            zp_[]:
                Photometric zeropoint. dictionary key will be followed by filter name
            zp_[]_err:
                error on photometric zeropoint
            []_inst:
                Instrumental magnitude
            []:
                Calibarted magnitude
            []_err:
                Error on Calibarated magnitude
            SNR:
                Signal to noise ratio of target
            lmag:
                calibrated limiting magnitude
    - fpath:
        filepath to new fits image file

'''


def main(object_info,syntax,fpath):

    # ensure to use copy of original inputed synatc instruction files
    syntax = syntax.copy()

    # Basic Packages
    import sys
    import shutil
    import os
    import numpy as np
    import warnings
    import pandas as pd
    import pathlib
    import collections
    import lmfit
    from pathlib import Path
    import time
    import logging
    # from photutils import CircularAperture

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.pyplot import Circle
    from os.path import dirname


    # Astrpy and photutils
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import astropy.wcs as wcs
    from astropy.stats import sigma_clipped_stats
    from astropy.time import Time
    from functools import reduce
    from astropy.coordinates import Angle
    from matplotlib.ticker import MultipleLocator


    # Proprietary modules
    from autophot.packages.functions import weighted_avg_and_std, getheader,getimage,zeropoint, mag,set_size,pix_dist
    from autophot.packages.functions import gauss_2d,gauss_fwhm2sigma,gauss_sigma2fwhm
    from autophot.packages.functions import moffat_2d,moffat_fwhm
    from autophot.packages.aperture import ap_phot
    from autophot.packages.check_wcs import updatewcs,removewcs
    from autophot.packages.call_astrometry_net import AstrometryNetLOCAL
    from autophot.packages.call_hotpants import HOTterPANTS
    from autophot.packages.call_yaml import yaml_syntax as cs
    from autophot.packages.uncertain import SNR
    from autophot.packages.get_template import get_pstars
    from autophot.packages.uncertain import sigma_mag_err
    from autophot.packages.limit import limiting_magnitude_prob
    from autophot.packages.call_crayremoval import run_astroscrappy

    import autophot.packages.find as find
    import autophot.packages.psf as psf
    import autophot.packages.call_catalog as call_catalog
    from matplotlib.gridspec import  GridSpec



    # Fix the issatty error - https://stackoverflow.com/questions/47069239/consolebuffer-object-has-no-attribute-isatty
    sys.stdout.isatty = lambda: False

    warnings.simplefilter(action='ignore', category=FutureWarning)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.style.use(os.path.join(dir_path,'autophot.mplstyle'))


    def border_msg(msg):
        row = len(msg)+2
        h = ''.join(['+'] + ['-' *row] + ['+'])
        result= h + '\n'"| "+msg+" |"'\n' + h
        print('1'+result)




    try:

        # Preparing output dictionary
        output = collections.OrderedDict({})

        if syntax['fits_dir'] != None:
            if syntax['fits_dir'].endswith('/'):
                syntax['fits_dir'] = syntax['fits_dir'][:-1]

        if syntax['wdir'].endswith('/'):
            syntax['wdir'] = syntax['wdir'][:-1]

# =============================================================================
# Prepare new file
# =============================================================================

        # Change to new working directory set by 'outdir_name' in yaml file
        if syntax['fname']:
            wdir  = syntax['fname']
            work_loc = str(pathlib.Path(dirname(wdir)))

        else:
            wdir = syntax['fits_dir']
            new_dir = '_' + syntax['outdir_name']


            base_dir = os.path.basename(wdir)
            work_loc = base_dir + new_dir

            new_output_dir = os.path.join(dirname(wdir),work_loc)

            # create new output directory is it doesn't exist
            pathlib.Path(new_output_dir).mkdir(parents = True, exist_ok=True)
            os.chdir(new_output_dir)


        '''
        Take in original data, copy it to new location
        New file will have _APT append to fname

        Create names of directories with name=filename
        and account for correct sub directories locations
        '''

        base_wext = os.path.basename(fpath)
        base = os.path.splitext(base_wext)[0]

        # Replace Whitespaces with dash
        base = base.replace(' ','_')

        # Replace full stops (excluding the extension) with dash
        base = base.replace('.','_')

        base = base.replace('_APT','') # If file already has AUTOPHOT in it - for debugging

        base = base.replace('_ERROR','')

        if not syntax['fname']:

            root = dirname(fpath)

            sub_dirs = root.replace(wdir,'').split('/')


            sub_dirs = [i.replace('_APT','').replace(' ','_') for i in sub_dirs]

            cur_dir = new_output_dir

        else:
            sub_dirs = ['']

            cur_dir = dirname(wdir)

        '''
        Move through list of subdirs,
        where the final copied file will be moved
        creating new sub directories with folder name
        extension until it reaches folder

        it will then create a new folder with name of file to which
        every output file is moved to
        '''
        fname_ext = Path(fpath).suffix

        for i in range(len(sub_dirs)):
            if i: # If directory is not blank
                 pathlib.Path(cur_dir +'/' + sub_dirs[i]+'_APT').mkdir(parents = True,exist_ok=True)
                 cur_dir = cur_dir +'/' + sub_dirs[i]+'_APT'

        # create filepath of write directory
        cur_dir = cur_dir + '/' + base
        pathlib.Path(cur_dir).mkdir(parents = True,exist_ok=True)

        # copy new file to new directory
        shutil.copyfile(fpath, (cur_dir+'/'+base + '_APT'+fname_ext).replace(' ','_'))

        # new fpath for working fits file
        fpath = cur_dir + '/' + base + '_APT'+fname_ext

        # base is name [without extension]
        base = os.path.basename(fpath)

        # write dir is where all files will be saved, pre-iteration
        syntax['write_dir'] = (cur_dir + '/').replace(' ','_')

        # Get image and header from function library
        image    = getimage(fpath)
        headinfo = getheader(fpath)

        if object_info == None:
            sys.exit('No Target Info')

        if syntax == None:
            sys.exit("No syntax input file"+"/n"+"*** I dont know what I'm doing! ***")



# =============================================================================
# Set up logging file
# =============================================================================

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


        # set up logging to file
        logging.basicConfig(level=logging.INFO,
                            format='%(filename)s-%(levelname)s: %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(cur_dir,str(base)+'.log'),
                            filemode='w')


        console = logging.StreamHandler()
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # formatter = logging.Formatter('%(filename)s: %(levelname)s: %(message)s')
        # formatter = logging.Formatter('%(filename)s line:%(lineno)d - %(message)s','%m-%d %H:%M:%S')
        # formatter = logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(message)s')

        # tell the handler to use this format
        console.setFormatter(formatter)

        # add the handler to the root logger
        logging.getLogger('').addHandler(console)


        import datetime
        logging.info('File: '+str(base) + ' - PID: '+str(os.getpid()))
        logging.info('Start Time: %s' % str(datetime.datetime.now()) )

#==============================================================================
# Main YAML input and syntax files
#==============================================================================

        # catalog syntax contains header information of selected
        # header given by 'catalog' keyword in yaml file
        filepath ='/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

        catalog_syntax_yml = 'catalog.yml'
        catalog_syntax = cs(os.path.join(filepath+'/databases',catalog_syntax_yml),syntax['catalog']).load_vars()

# =============================================================================
# Get time of observation
# =============================================================================

        '''
        Geting date of observation from header info
        if 'MJD-OBS' in headinfo, use that
        if not:
            look for 'DATE-AVG' or 'DATE-OBS' and convert to mjd
        if not:
            return np.nan and continue onwards
        '''
        try:

            if 'MJD-OBS' in headinfo:
                mjd_date = headinfo['MJD-OBS']
            else:
                if 'DATE-OBS' in headinfo:
                    time_obs = headinfo['DATE-OBS']
                if 'DATE-AVG' in headinfo:
                    time_obs = headinfo['DATE-AVG']

                time_obs_iso = Time(time_obs,  format='isot')
                mjd_date = time_obs_iso.mjd
                headinfo['MJD-OBS'] = (mjd_date,'Modified Julian Date by AutoPhOT')

        except:
            logging.warning(' Modied Julian Date NOT FOUND' )
            mjd_date = np.nan
            pass


        # Find exact time of observations for airmass calculations
        try:
            try:
                time_obs = headinfo['DATE-AVG']
            except:
                time_obs = headinfo['DATE-OBS']
            time_obs_iso = Time(time_obs,  format='isot')
            syntax['obs_time'] = time_obs_iso.mjd

        except:
            syntax['obs_time'] = mjd_date


# =============================================================================
# Telescope and instrument parameters§
# =============================================================================

        # Get f.o.v from tele_syntax if needed for wcs and upaate syntax dictionary
        # if f.o.v parametrs are not present, guess_scale in astometry.net
        try:
            telescope = headinfo['TELESCOP']
        except:
            telescope  = 'UNKNOWN'
            headinfo['TELESCOP'] = 'UNKNOWN'

        if telescope  == '':
            telescope  = 'UNKNOWN'
            headinfo['TELESCOP'] = 'UNKNOWN'



        inst_key = 'INSTRUME'
        inst = headinfo[inst_key]


        tele_syntax_yml = 'telescope.yml'
        teledata = cs(os.path.join(syntax['wdir'],tele_syntax_yml))
        tele_syntax = teledata.load_vars()

        if not syntax['master_warnings']:
            warnings.filterwarnings("ignore")


        # F.o.V using in astrometry
        syntax['scale_type']  = tele_syntax[telescope][inst_key][inst]['scale_type']
        syntax['scale_high']  = tele_syntax[telescope][inst_key][inst]['scale_high']
        syntax['scale_low']   = tele_syntax[telescope][inst_key][inst]['scale_low']

        syntax['mjd'] = mjd_date

        # Update outputs with filename, telescope and observation time in mjd
        output.update({'fname':fpath})
        output.update({'telescope':tele_syntax[telescope][inst_key][inst]['Name']})
        output.update({'TELESCOP':telescope})
        output.update({'INSTRUME':inst_key})
        output.update({'instrument':inst})
        output.update({'mjd':mjd_date})

# =============================================================================
# Find filter
# =============================================================================

        '''
        Get correct filter keyword with the default being 'FILTER'
        In collaboration with 'write_yaml function'
        will serach for 'FILTER' using filter_key_0 key

        if found sets 'filter_key' in syntax file
        if not:
            file search for filter_key_1 key in telescope_syntax and check
            if result value in headinfo.
        will continue until no more filter_key_[] are in telescope_syntax or
        right keyword is found

        if fails:
            returns filter_key = 'no_filter'

        Was implemented to allow for, although the same telescope/instrument,
        the header name for the filter keyoward may be different
        '''

        filter_header = 'filter_key_0'

        import re

        while True:
            if tele_syntax[telescope][inst_key][inst][filter_header] not in list(headinfo.keys()):
                old_n = int(re.findall(r"[-+]?\d*\.\d+|\d+", filter_header)[0])
                filter_header = filter_header.replace(str(old_n),str(old_n+1))
            elif tele_syntax[telescope][inst_key][inst][filter_header].lower() == 'clear':
                continue
            else:
                break

        syntax['filter_key'] = tele_syntax[telescope][inst_key][inst][filter_header]

        try:

            headinfo[syntax['filter_key']]
        except:
            syntax['filter_key'] = 'no_filter'
            logging.warning('Filter keywoard == no_filter')

        logging.debug('Filyer keyoward used: %s' % syntax['filter_key'] )

        '''
        Select filter to cross check with selected catalog

        Specific telescope nomenclature found in telescope_syntax.yml

        if no filter is used, keyword is [Clear, no filter, etc],
        script will use a preselected filter ( force_filter keyword )

        if no force_filter is selected,
        using filter presented in input.yml as force_filter

        -- Future implementation to default based on instrument --
        '''

        use_filter =  tele_syntax[telescope][inst_key][inst][str(headinfo[syntax['filter_key']])]
        syntax['filter'] = use_filter

        if use_filter.lower() == 'no_filter':
            if syntax['force_filter'] != 'None':
                use_filter = syntax['force_filter']
            if syntax['force_filter'].lower() != 'clear':
                use_filter = syntax['force_filter']
            else:
                # List of filters, will check spread and return most appropiate filter
                use_filter = syntax['filter_key']

# ============================================================================
# Begin photometric reductions
# =============================================================================


        '''
        if verbose is true, all prints will be printed inline as well as
        to text file

        if False, it will only print to text file

        crappy idea - will implement logging module

        '''

        start = time. time()
        logging.info('Telescope: %s' % telescope)
        logging.info('Filter: %s'% use_filter)
        logging.info('MJD: %.3f' % mjd_date)

        # Get date of observations
        date = Time([mjd_date], format='mjd', scale='utc')

        logging.info('Observation Date: %s' % date.iso[0].split(' ')[0])


        if tele_syntax[telescope][inst_key][inst]['GAIN'] == None:
            logging.warning('Gain keyword not found - check telescope.yml - SETTING TO 1')
            gain = 1

        elif tele_syntax[telescope][inst_key][inst]['GAIN'] in headinfo:
            gain = float(headinfo[tele_syntax[telescope][inst_key][inst]['GAIN']])
            logging.info('Gain: %.f' % gain)
            '''
            Look for gain ketword, with preset as 'GAIN',
            same procedure as finding filter keyword.
            '''
        else:
            logging.warning('Gain keyword not found - check telescope.yml - SETTING TO 1')
            gain = 1

        syntax['gain'] = gain

# =============================================================================
# File deduction - VERY dependant on
# =============================================================================
        try:
            try:

                fpath = reduce(fpath,use_filter,syntax)
                image    = fits.getdata(fpath)
                headinfo = getheader(fpath)

            except:
                pass

            if fpath == None:
                    raise Exception

# ============================================v=================================
# Expsoure time
# =============================================================================
            try:
                for i in ['EXPTIME','EXP_TIME','TIME-INT']:
                    if i in headinfo:
                        exp_time = headinfo[i]
            except:
                logging.warning('Cannot find exposure time: setting to 1')
                exp_time = 1


            '''
            Known issue -> header values written as string

            Below just splits them apart
            '''
            if isinstance(exp_time, str):
               exp_time = exp_time.split('/')
               exp_time = float(exp_time[0])

            syntax['exp_time'] = exp_time

            try:
                syntax['sat_lvl'] = headinfo['SATURATE']
            except:
                syntax['sat_lvl'] = 2**16

            logging.info('Exposure time: %.fs ' % exp_time)



            if syntax['trim_edges']:

                logging.info('Trimming edges of image by %d pixels' % syntax['trim_edges_pixels'])

                image_trim = image[syntax['trim_edges_pixels']:image.shape[0]-syntax['trim_edges_pixels'],
                                   syntax['trim_edges_pixels']:image.shape[1]-syntax['trim_edges_pixels']]

                logging.info('Image shape (%d,%d) -> (%d,%d)' % (image.shape[0],image.shape[1],image_trim.shape[0],image_trim.shape[1]))


                fits.writeto(fpath,image_trim.astype(np.single),
                             headinfo,
                             overwrite = True,
                             output_verify = 'silentfix+ignore')

                image = image_trim



            if 'NAXIS1' in headinfo and 'NAXIS' in headinfo:
                syntax['NAXIS1'] = headinfo['NAXIS1']
                syntax['NAXIS2'] = headinfo['NAXIS2']
            else:
                syntax['NAXIS1'] = image.shape[0]
                syntax['NAXIS2'] = image.shape[1]






# =============================================================================
# Cosmic ray removal usign astroscrappy
# =============================================================================

        # Removing cosmic rays using 'astroscrappy'
        # documnetation:
        # - https://buildmedia.readthedocs.org/media/pdf/astroscrappy/latest/astroscrappy.pdf


            if syntax['remove_cmrays']:
                try:
                # if cosmic rays have no already been removed
                        if 'CRAY_RM'  not in headinfo:


                            headinfo = getheader(fpath)

                            # image with cosmic rays
                            image_old = fits.PrimaryHDU(image)

                            image,syntax = run_astroscrappy(image_old,syntax)

                            # if np.nanmedian(image) <= 0:
                            #     headinfo['apt_offset'] = ('T', 'Additon of image offset ')
                            #     logging.info('Median of image: %.f - applying offset' % np.nanmedian(image))
                            #     image = image + abs( (np.nanmin(image) ))

                            # Update header and write to new file
                            headinfo['CRAY_RM'] = ('T', 'Comsic rays w/astroscrappy ')

                            fits.writeto(fpath,
                                         image.astype(np.single),
                                         headinfo,
                                         overwrite = True,
                                         output_verify = 'silentfix+ignore')

                            logging.info('Cosmic rays removed - image updated')
                        else:
                            logging.info('Cosmic sources pre-cleaned')

                except Exception as e:
                    logging.exception(e)



# =============================================================================
# WCS and target
# =============================================================================

            '''
            Autophot can be executed with or without target coords

            If valid target name i.e. 2009ip -  target ra/dec will be taken from tns query and converted to pixel coords
            '''

            if syntax['target_name'] != None:

                '''
                Get target info from [pre-saved] TNS_response
                '''

                target_ra  = object_info['ra']
                target_dec = object_info['dec']

                target_coords = SkyCoord(target_ra , target_dec ,unit = (u.hourangle,u.deg))

                syntax['target_ra'] = target_coords.ra.degree
                syntax['target_dec']= target_coords.dec.degree

            elif syntax['target_ra'] != None and syntax['target_dec'] != None:

                target_coords = SkyCoord(syntax['target_ra'] , syntax['target_dec'] ,unit = (u.deg,u.deg))

                syntax['target_ra'] = target_coords.ra.degree
                syntax['target_dec']= target_coords.dec.degree

            else:
                try:
                    if syntax['use_header_radec']:
                        target_coords = SkyCoord(headinfo['CAT-RA'] , headinfo['CAT-DEC'] ,unit = (u.hourangle,u.deg))

                        syntax['target_ra'] =  target_coords.ra.degree
                        syntax['target_dec']=  target_coords.dec.degree

                except:
                    logging.warning('NO RA:DEC keywords found')
    #
    #==============================================================================
    # WCS check
    #===========================================∂==================================

            '''
            -- Various instances of when/if to query using astrometry.net --

            if any instance of wcs_keywords are not found in the header infomation it

            if succesful, it will add the UPWCS = T header/value to the header
            hdu of the newly created file
            '''

            if syntax['remove_wcs'] or syntax['trim_edges']:

                new_header = removewcs(headinfo,delete_keys = True)
                fits.writeto(fpath,image.astype(np.single),new_header,overwrite = True,output_verify = 'silentfix+ignore')
                headinfo = getheader(fpath)

            # search keywords for wcs validation
            wcs_keywords = ['CD1_1','CD1_2','CD2_1','CD2_2',
                            'CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                            'CDELT1','CDELT2','CTYPE1','CTYPE2']

            # if no wcs values are found in headinfo, ignore file and exit loop (raise exception)
            if syntax['ignore_no_wcs']:
                if any(i not in headinfo for i in wcs_keywords):
                    logging.info('No wcs found - ignoring_wcs setting == True')
                    raise Exception('ignore files w/o WCS')

            if 'UPWCS'  in headinfo:
                # if UPWCS already excecuetde and found in header continue
                logging.info('Astrometry.net already excuted')

            elif all(i not in headinfo for i in wcs_keywords):


                '''

                Try to solve for WCS


                Call local instance of astrometry.net alogorithm
                astrometry.net documentation:
                https://buildmedia.readthedocs.org/media/pdf/astrometrynet/latest/astrometrynet.pdf

                '''

                logging.info('No WCS values found - attempting to solve field')


                # Run local instance of Astrometry.net - returns filepath of wcs file
                astro_check = AstrometryNetLOCAL(fpath, syntax= syntax)

                # Open wcs fits file with wcs values
                new_wcs  = fits.open(astro_check,ignore_missing_end = True)[0].header

                old_headinfo = getheader(fpath)

                # script used to update per-existing header file with new wcs values
                headinfo_updated = updatewcs(old_headinfo,new_wcs)

                # update header to show wcs has been checked
                headinfo_updated['UPWCS'] = ('T', 'updated WCS by APT')

                # Write new header
                fits.writeto(fpath,image.astype(np.single),headinfo_updated,overwrite = True,output_verify = 'silentfix+ignore')
                headinfo = getheader(fpath)

                logging.info('WCS saved to new file')





                if syntax['update_wcs_scale']:

                    'Update image scale params from '

                    logging.info('Update scale units from astrometry.net')

                    find_scale =[c for c in list(new_wcs['comment']) if "scale" in c if 'arcsec/pix' in c]

                    pixel_scale =[s for s in find_scale if s.split(':')[0] =='scale']
                    pixel_scale_value = round(float(pixel_scale[0].split(':')[1].split(' ')[1]),3)

                    teledata.update_var(telescope,inst_key,inst,'scale_high',pixel_scale_value*1.1)
                    teledata.update_var(telescope,inst_key,inst,'scale_low' ,pixel_scale_value*0.9)
                    teledata.update_var(telescope,inst_key,inst,'scale_type','arcsecperpix')


                    fov_AREA = pixel_scale_value * image.shape[0]  * image.shape[1]
                    # if syntax['remove_cmrays' ]:
                    #     with open("/Users/seanbrennan/Desktop/autophot_paper/CR_versus_exposure.txt", "a") as myfile:
                    #         myfile.write('%.3f %d %.3f %.3f\n' % (syntax['exp_time'],syntax['CR_detections'],syntax['CR_time_taken'],fov_AREA))







            else:
                logging.info('WCS found')




            #==============================================================================
            # Load target information from TNS query if needed
            #==============================================================================

            w1 = wcs.WCS(headinfo)# WCS values

            if syntax['target_name'] == None:

                '''
                If not target coords are given, will use center of field
                '''

                if syntax['target_ra'] == None and syntax['target_dec'] == None:

                   '''
                   if no object is given i.e name,ra,dec then take the middle
                   of the screen as the target and get region around it
                   '''

                   # translate pixel values to ra,dec at center of image
                   center = w1.all_pix2world([image.shape[1]/2],[image.shape[0]/2],1)

                   # get ra,dec in deg/SkyCoord format
                   target_coords = SkyCoord(center[0][0] , center[1][0] ,unit = (u.deg,u.deg))

                   # update syntax file
                   syntax['target_ra'] = target_coords.ra.degree
                   syntax['target_dec']= target_coords.dec.degree

                   # Target coords are now set to cwnter of image
                   syntax['target_x_pix'] = image.shape[1]/2
                   syntax['target_y_pix'] = image.shape[0]/2

                   target_x_pix = image.shape[1]/2
                   target_y_pix = image.shape[0]/2

                else:

                   '''
                   if no name is given but ra and dec are, use those instead
                   '''

                   target_ra  = syntax['target_ra']
                   target_dec = syntax['target_dec']

                   target_coords = SkyCoord(target_ra , target_dec ,unit = (u.deg,u.deg))

                   target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)

                   syntax['target_ra'] = target_coords.ra.degree
                   syntax['target_dec']= target_coords.dec.degree

                   syntax['target_x_pix'] = target_x_pix
                   syntax['target_y_pix'] = target_y_pix

            elif syntax['target_name'] != None:
                    '''
                    Get target info from [pre-saved] TNS_response
                    '''
                    target_ra  = object_info['ra']
                    target_dec = object_info['dec']

                    target_coords = SkyCoord(target_ra , target_dec ,unit = (u.hourangle,u.deg))

                    target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)

                    syntax['target_ra'] = target_coords.ra.degree
                    syntax['target_dec']= target_coords.dec.degree

                    syntax['target_x_pix'] = target_x_pix
                    syntax['target_y_pix'] = target_y_pix

            else:
               try:
                   target_coords = SkyCoord(headinfo['RA'] , headinfo['DEC'] ,unit = (u.hourangle,u.deg))
                   target_x_pix, target_y_pix = w1.all_world2pix(target_coords.ra.degree, target_coords.dec.degree, 1)

                   syntax['target_ra'] =  target_coords.ra.degree
                   syntax['target_dec']=  target_coords.dec.degree

                   syntax['target_x_pix'] = target_x_pix
                   syntax['target_y_pix'] = target_y_pix

               except Exception as e:
                   logging.exception(e+'> NO RA:DEC keywords found - attempting astrometry without <')

#==============================================================================
# FWHM - Using total image source detection
#==============================================================================

            # get approx fwhm, dataframe of sources used and updated syntax
            # returns fwhm from gaussian fit - and dataframe of sources used
            mean_fwhm,df,syntax = find.fwhm(image,syntax)

            # print(df.columns)

            mean_fwhm_err = np.nanstd(df['FWHM'])
#




            if mean_fwhm < syntax['nyquist_limit'] and syntax['check_nyquist']:
                logging.warning('FWHM [%.1f]<2.5 - Sampling errors - using aperture Photometry' % mean_fwhm)
                syntax['do_ap_phot']  = True
                do_ap = True

            elif mean_fwhm > 25:
                logging.info('FWHM Error %.3f - check image quality' % round(mean_fwhm,3) )
                raise Exception('Reductions failed - skipping')
            else:
                logging.info('\nImage FWHM: %.3f +/- %.3f \n' % (mean_fwhm,mean_fwhm_err))

                for key,val in syntax['image_params'].items():

                    logging.info('%s:%.3f\n'% (key,val))

            if mean_fwhm>6:
                print('Swtching to polynomial backgraound fit')
                syntax['psf_bkg_poly'] = True


            # perform approx aperture photometry on sources
            df = find.phot(image,syntax,df,mean_fwhm)

            '''
            ap_corr needs to be moved to another place

            '''
            ap_corr_base = find.ap_correction(image,syntax,df)

#==============================================================================
# Catalog source detecion
#==============================================================================
            if syntax['do_catalog']:

                logging.info('Searching for viable sources')

                while True:

                    # Search for sources in images that have corrospondong magnityide entry in given catalog
                    specificed_catalog = call_catalog.search(image,headinfo,target_coords,syntax,catalog_syntax,use_filter)

                    # Re-aligns catalog sources with source detection and centroid
                    c,syntax = call_catalog.match(image,headinfo,target_coords,syntax,catalog_syntax,use_filter,specificed_catalog,mean_fwhm)

                    # If  UPWCS keyword, image has been already ran through ASTROMETRY, no need to recheck
                    # c = c.drop(c[c['cp_dist'] > syntax['match_dist']].index)
                    # sigma clip distances - avoid mismatches default sigma = 3
                    sigma_dist =  np.array(sigma_clipped_stats(c['cp_dist']))

                    lower_x_bound = syntax['scale']
                    lower_y_bound = syntax['scale']
                    upper_x_bound = syntax['scale']
                    upper_y_bound = syntax['scale']


                    c = c[c.x_pix < image.shape[1] - upper_x_bound]
                    c = c[c.x_pix > lower_x_bound]
                    c = c[c.y_pix < image.shape[0] - upper_y_bound]
                    c = c[c.y_pix > lower_y_bound]

                    if len(c) ==0:
                        raise Exception('Could NOT find any catalog sources in field')




                    logging.info('Average pixel offset: %.3f '% np.nanmedian(list(sigma_dist)))




                    '''
                    Attempt to catch a systematic offset - if wcs and catalog values are off by more than 'offset_param'
                    - recheck wcs if not already done so by autophot

                    '''
                    if np.nanmedian(list(sigma_dist)) <= syntax['offset_param']:
                        break

                    if not syntax['allow_recheck']:
                        break

                    if 'UPWCS' in headinfo:
                        logging.info('UPWCS found - skipping astrometry')
                        break


                    # Sources from catalog match closly with recentered values, i.e we have a good match
                    # If bad WCS values:
                    #    wipe existing Valeus and run thourgh astrometry.net and
                    #    re-run throgh catalog matching

                    logging.info('Removing and Rechecking WCS')

                    # remove wcs values and update -  will delete keys from header file
                    fit_open = fits.open(fpath,ignore_missing_end = True)
                    fit_header_wcs_clean = removewcs(fit_open,delete_keys = True)

                    fit_header_wcs_clean.writeto(fpath,
                                                 overwrite = True,
                                                 output_verify = 'silentfix+ignore')
                    # fit_open.close()

                    # Run astromety - return filepath of wcs file
                    astro_check = AstrometryNetLOCAL(fpath,syntax = syntax)

                    new_wcs  = fits.open(astro_check,ignore_missing_end = True)
                    fit_open = updatewcs(fit_open,new_wcs)
                    # new_wcs.close()

                    fit_open.writeto(os.getcwd()+'/' + base , overwrite = True,output_verify = 'silentfix+ignore')

                    fpath = os.getcwd()+ '/' + base
                    # fit_open.close()

                    fit_open = fits.open(fpath,ignore_missing_end = True)
                    headinfo = getheader(fpath)

                    headinfo['UPWCS'] = ('T', 'CROSS CHECKED WITH ASTROMETRY.NET')

                    fit_open.writeto(fpath , overwrite = True,output_verify = 'silentfix+ignore')
                    fit_open.close()


#==============================================================================
# Aperture Photometry on catalog sources
#=============================================================================

                c_temp_dict = {}

                dist_list = []

                # Go through each matched catalog source and get its' distance to every other source
                for i in c.index:

                    try:
                        dist = np.sqrt((c.x_pix[i]-np.array(c.x_pix))**2 +( c.y_pix[i]-np.array(c.y_pix))**2)
                        dist = dist[np.where(dist!=0)]
                        # add minimum - used to find isolated sources
                        dist_list.append(np.nanmin(dist))

                    except:
                        dist_list.append(np.nan)
                '''
                loop will attempt to perform psf photometry unless unavailable
                further down into the script
                '''

                #Unless selected - perorm psf over aperture
                if syntax['do_ap_phot']:
                    do_ap = True
                else:
                    do_ap = False



# =============================================================================
# Lets Phot!
# =============================================================================

            '''
            Attempt psf photome
            if it selected in input.yml - default: True
            '''

            if syntax['use_local_stars']:


                from astropy import units as u

                local_star_radius = Angle(float(syntax['use_source_arcmin']), u.arcmin)

                local_points = w1.all_world2pix([syntax['target_ra'], syntax['target_ra']+local_star_radius.degree],
                                                [syntax['target_dec'],syntax['target_dec']+local_star_radius.degree]
                                                , 1)

                local_radius = pix_dist(local_points[0][0],local_points[0][1],  local_points[1][0],local_points[1][1])



                syntax['local_radius'] =local_radius

                logging.info('Using stars within %d arcmin [%d px]' % (syntax['use_source_arcmin'],local_radius))


            if syntax['do_psf_phot'] and not syntax['do_ap_phot'] and not do_ap:
                try:

                    logging.info('Using PSF Photometry')

                    '''
                    Build model psf from good sources

                    returns:
                        - residual table:
                        - sigma fit from gaussian fitting
                        - heights/amplitudes of sources used to make psf with x and y locations
                        - updated syntax
                    '''

                    r_table,fwhm_fit,psf_heights,syntax = psf.build_r_table(image,df,syntax,mean_fwhm)

                    # if it fails, select aperture photometry and exit this attempt
                    if fwhm_fit == None:
                        do_ap=True
                        logging.error('> Sigma fit ERROR <')

                    elif np.any(r_table ==  None):
                        do_ap=True
                        logging.error('PSF fitting unavialable')


                    else:

                        # Fwhm to use for model psf and hereafter
                        # mean_fwhm = np.nanmedian(fwhm_fit)

                        logging.info('FWHM of PSF model: %.3f' % mean_fwhm)

                        if mean_fwhm > 25:
                            logging.warning(' High FWHM %.3f' % round(mean_fwhm,1))
                            raise Exception

                        '''
                        Get data of sources used for psf
                        '''
                        psf_stats,_ = psf.do(psf_heights,r_table,syntax,mean_fwhm)

                        '''
                        Get approximate magnitude of psf to check if target is better than psf
                        very dodgey if target is not specific and target is set to center of image

                        '''

                        approx_psf_mag = float(mag(np.nanmean(psf_stats['psf_counts']),0))
                        logging.info('Approx PSF mag %.3f' % approx_psf_mag)

                        '''
                        Beginning fitting model psf to table of match sources from catalog
                        returns:
                            - c_psf: updated dataframe with fitted outputs
                            - model_psf - function of psf model used - see psf.py module
                        '''

                        c_psf,model_psf = psf.fit(image,c,r_table,syntax,mean_fwhm)

                        '''
                        Get counts/ magnitudes of sources from psf fitting
                        returns:
                            - c_psf: updated dataframe with psf data
                            - syntax: udpated syntax file
                        '''
                        c_psf,syntax = psf.do(c_psf,r_table,syntax,mean_fwhm)

                        # print(c_psf.psf_counts)

                        # background flux
                        bkg = np.median(c_psf['bkg'])/exp_time

                        # source flux
                        ap_sum = c_psf.psf_counts/exp_time

                        # Error in flux
                        ap_sum_err = c_psf.psf_counts_err/exp_time

                        # Signal to noise of source
                        SNR_val = np.array(ap_sum/ap_sum_err)



                except Exception as e:
                    logging.exception(e)

                    # if something goes wrong- do aperture  photometry instead
                    do_ap = True
                    pass

            # add fwhm to syntax file
            syntax['fwhm'] = mean_fwhm

            # Perform aperture photoometry if pre-selected or psf fitting wasn't viable
            if syntax['do_ap_phot'] or do_ap == True:

                logging.info('Using Aperture Photometry')

                # list of tuples of pix coordinates of sources
                positions  = list(zip(np.array(c.x_pix),np.array(c.y_pix)))

                '''
                Aperture photometry model:

                    returns:
                        - ap: total sum of counts wtihitn aperture of radius "radius"
                        - bkg: background sum of counts from annulus of inner/outer radius r_in/r_out
                '''
                ap,bkg = ap_phot(positions,
                                 image,
                                 radius =  syntax['ap_size']    * mean_fwhm,
                                 r_in =    syntax['r_in_size']  * mean_fwhm,
                                 r_out =   syntax['r_out_size'] * mean_fwhm)

                # Background flux from annulus
                bkg = bkg/exp_time

                # Source flux
                ap_sum = ap/exp_time

                # SNR from ccd equation
                # needs to be updated
                SNR_val = SNR(ap_sum,bkg,exp_time,0,syntax['ap_size']* mean_fwhm,gain,0)


            # Adding everything to temporary dataframe - no idea why though
            c_temp_dict['dist'] = dist_list
            c_temp_dict['SNR'] = SNR_val
            c_temp_dict['count_rate_bkg'] = bkg
            c_temp_dict['count_rate_star']= ap_sum

            # print(c.columns)

            # add to exisitng dataframe [c]
            c_add = pd.DataFrame.from_dict(c_temp_dict)
            c_add = c_add.set_index(c.index)

            c = pd.concat([c,c_add],axis = 1,sort = False)

            # drop if the counts are negative - account for mismatched source or very faint source
            c = c.drop(c[c['count_rate_star'] < 0.0].index)


            '''
            if iso_cat == True:
                remove sources that have a neighbouring source within a user-defined distance
                given by iso_cat_dist.
            '''

            # if syntax['remove_sat'] :

            c = c.drop(c[np.isnan(c['count_rate_star'])].index)


            if syntax['iso_cat']:
                c = c.drop(c[c['dist'] < syntax['iso_cat_dist']].index)

            if syntax['use_local_stars']:
                c = c[c['dist2target'] <= local_radius]

            # Instrumental magnitude
            # print(c['count_rate_star'])
            c['inst_'+str(use_filter)] = mag(c['count_rate_star'],0)

            # Error in instrumental Magnitude

            c = c.drop(c[c['inst_'+str(use_filter)]==0].index)
            # c = c[abs(c['inst_'+str(use_filter)])>0.0 ]

            c_SNR_err = sigma_mag_err(c.SNR)

            c['inst_'+str(use_filter)+'_err'] = c_SNR_err

            # print(c)


# =============================================================================
# Aperture correction
# =============================================================================

            if do_ap:
                ap_corr = ap_corr_base
            else:
                ap_corr = 0

# =============================================================================
# Find Zeropoint
# =============================================================================

            if syntax['do_zeropoint']:

                try:
                    '''
                    initilaise dictionaries for zeropoint and errors
                    '''
                    zp = {}
                    zp_err ={}



                    # get magnitude errors on catalog source from SNR

                    dmag = {
                        'U':['U','B'],
                        'B':['B','V'],
                        'V':['V','R'],
                        'R':['V','R'],
                        'I':['R','I'],
                        'u':['u','g'],
                        'g':['g','r'],
                        'r':['g','r'],
                        'i':['r','i'],
                        'z':['i','z']
                        }


                    '''
                    Add zeropoint [zp_[choosen filter]] and zeropoint error [zp_[choosen filter]_err]  to c dataframe:

                    zeropoint:
                        inputs:
                            catalog magnitude:
                            flux of star: given by count_rate_star
                    '''


                    if syntax['apply_ct_zerpoint']:
                        idx = np.where(np.logical_and(~np.isnan(c[dmag[use_filter][0]]) , ~np.isnan(c[dmag[use_filter][1]])))[0]
                        c[dmag[use_filter][0]+'_'+dmag[use_filter][1]] = c[dmag[use_filter][0]] - c[dmag[use_filter][1]]
                        ct_gradient  = -0.0078
                        dmag = c[dmag[use_filter][0]+'_'+dmag[use_filter][1]]
                    else:
                        ct_gradient = None
                        dmag = None

                    zp_mag_err = sigma_mag_err(c['SNR'])


                    c['zp_'+str(use_filter)] = np.array(zeropoint(c[str('cat_'+use_filter)], c['count_rate_star'],ct_gradient = ct_gradient,dmag = dmag))

                    # error is magnitude error from catalog and instrumental revovery mangitude error from SNR added in qaudrature
                    c['zp_'+str(use_filter)+'_err'] = np.array(np.sqrt( c[str('cat_'+use_filter)+'_err']**2 + zp_mag_err**2) )

                    # dataframe has been updated with zeropiint calculations - now to sigma clip to get viable zeropiont
                    zpoint = np.asarray(c['zp_'+str(use_filter)])
                    zpoint_err = np.asarray(c['zp_'+str(use_filter)+'_err'])

                    # remove nan values and apply mask
                    nanmask = np.array(~np.isnan(zpoint))

                    zpoint = zpoint[nanmask]
                    zpoint_err = zpoint_err[nanmask]


                    if len(zpoint) == 0:
                        zp_wa = [np.nan,np.nan]
                        raise Exception('No Zeropoints estimates found')



                    zp_mask = np.array(sigma_clip(zpoint, sigma = syntax['zp_sigma']).mask)


                    # Get instrumental magnitude for catalog sources from autophot photometry
                    zp_inst_mag = mag(c['count_rate_star'],0)[nanmask]

                    # clip according to zp_mask
                    zpoint_clip = zpoint[~zp_mask]
                    zpoint_err_clip = zpoint_err[~zp_mask]
                    zp_inst_mag_clip =  zp_inst_mag[~zp_mask]
                    '''
                    Get weighted average of zeropoints weighted by their magnitude errors
                    '''
                    zpoint_err_clip[zpoint_err_clip == 0] = 1e-5
                    zpoint_err_clip[np.isnan(zpoint_err_clip)] = 1e-5

                    # return value [zp_wa[0]] and error  [zp_wa[1]]
                    zp_wa =  weighted_avg_and_std(np.array(zpoint_clip),np.sqrt(1/zpoint_err_clip))

                    logging.info('\n%s-band zeropoint: %.3f +/- %.3f \n' % (str(use_filter),zp_wa[0],zp_wa[1]))

                    # Adding fwhm and Zeropoint to headerinfo
                    headinfo['fwhm'] = (round(mean_fwhm,3), 'fwhm w/ autophot')
                    headinfo['zp']   = (round(zp_wa[0],3), 'zp w/ autophot')

                    fits.writeto(fpath,image.astype(np.single),
                                 headinfo,
                                 overwrite = True,
                                 output_verify = 'silentfix+ignore')



                    syntax['zeropoint'] = zp
                    syntax['zeropoint_err'] = zp_err


                except Exception as e:
                    logging.exception(e)
                    logging.critical('Zeropoint not Found')
                    zp_wa = [np.nan,np.nan]
                    pass

                syntax['zp'] =zp_wa[0]
                syntax['zp_err'] =zp_wa[1]

                zp     = {'zp_'+str(use_filter):zp_wa[0]}
                zp_err = {'zp_'+str(use_filter)+'_err':zp_wa[1]}

                headinfo['ZP'] = (zp_wa[0],'ZP by AUTOPHOT')

                # Observed magnitude
                c[str(use_filter)] = mag(c['count_rate_star'],zp_wa[0])

                # Error in observed magnitude
                c[str(use_filter)+'_err'] = np.sqrt(c_SNR_err**2 + zp_wa[1]**2)





# =============================================================================
# Plot of image with sources used and target
# =============================================================================
                fig_source_check = plt.figure(figsize = set_size(500,aspect = 1))

                try:
                    # from astropy.visualization.mpl_normalize import ImageNormalize
                    from astropy.visualization import  ZScaleInterval

                    x_pix_sources,y_pix_sources = w1.all_world2pix(c.ra.values,c.dec.values,1)

                    # norm = ImageNormalize( stretch = SquaredStretch())

                    vmin,vmax = (ZScaleInterval(nsamples = 600)).get_limits(image)

                    ax = fig_source_check.add_subplot(111)

                    ax.imshow(image,
                              vmin = vmin,
                              vmax = vmax,
                              interpolation = 'nearest',
                              origin = 'lower',
                              aspect = 'equal',
                              cmap = 'Greys')

                    ax.scatter(np.array(c.x_pix),np.array(c.y_pix),
                               marker = 's',
                               facecolor = 'none',
                               color = 'blue',
                               s = 15,
                               label = 'Centroiding [%d]' % len(c.x_pix),
                               zorder = 2)

                    ax.scatter(x_pix_sources,y_pix_sources,
                               marker = '^',
                               facecolor = 'none',
                               edgecolor = 'green',
                               s = 15,
                               label = 'Catalog sources [%d]' % len(x_pix_sources),
                               zorder = 1
                               )

                    if syntax['target_name'] != 'None':
                        tname = syntax['target_name']
                    else:
                        tname = 'Center of Field'

                    ax.scatter([target_x_pix],[target_y_pix],
                               marker = 'D',
                               s = 25,
                               facecolor = 'None',
                               edgecolor = 'gold',
                               label = 'Target: %s' % tname)

                    if not do_ap:
                        ax.scatter(psf_heights.x_pix,psf_heights.y_pix,
                                   marker = 'o',
                                   color = 'red',
                                   s = 15,
                                   facecolor = 'None',
                                   label = 'PSF Sources [%d]' % len(psf_heights),
                                   zorder = 3)
                    if syntax['use_local_stars']:

                        local_radius_circle = plt.Circle( ( target_x_pix, target_y_pix ), syntax['local_radius'],
                                                         color = 'red',
                                                         ls = '--',
                                                         label = 'Local Radius [%d px]' % syntax['local_radius'],
                                                         fill=False)
                        ax.add_patch( local_radius_circle)
                        ax.set_xlim(0,image.shape[1])
                        ax.set_ylim(0,image.shape[0])



                    ax.set_xlabel('X PIXEL')
                    ax.set_ylabel('Y PIXEL')

                    plt.legend(fancybox=True,
                               ncol = 3,
                               bbox_to_anchor=(0, 1.15, 1, 0),
                               loc = 'upper center',
                               frameon=False)

                    if syntax['save_source_plot']:
                        fig_source_check.savefig(cur_dir + '/' +'source_check_'+str(base.split('.')[0])+'.pdf',
                                                 format = 'pdf',bbox_inches='tight')

                    plt.close(fig_source_check)


                except Exception as e:
                    logging.exception(e)


    # =============================================================================
    #     Plotting Zeropoint hisograms w/ clipping
    # =============================================================================
                try:
                    import matplotlib.gridspec as gridspec
                    from scipy.optimize import curve_fit


                    def gauss(x,a,x0,sigma):
                        return a*np.exp(-(x-x0)**2/(2*sigma**2))

                    fig_zeropoint = plt.figure(figsize = set_size(540,aspect = 1))

                    gs = gridspec.GridSpec(2, 2)

                    ax1 = fig_zeropoint.add_subplot(gs[:-1, :-1])
                    ax2 = fig_zeropoint.add_subplot(gs[-1, :-1])
                    ax3 = fig_zeropoint.add_subplot(gs[:, -1])

                    markers, caps, bars = ax1.errorbar(zpoint,zp_inst_mag,xerr = zpoint_err,
                                 label = 'Before clipping',
                                 marker = 'o',
                                 linestyle="None",
                                 color = 'r',
                                 capsize=1,
                                 capthick=1)

                    [bar.set_alpha(0.3) for bar in bars]
                    [cap.set_alpha(0.3) for cap in caps]


                    ax1.set_ylabel('Instrumental magnitude')
                    ax1.set_xlabel('Zeropoint Magnitude')

                    ax1.invert_yaxis()

                    markers, caps, bars = ax2.errorbar(zpoint_clip,zp_inst_mag_clip,
                                 xerr = zpoint_err_clip,
                                 label = 'After clipping [$%s sigma$]' %  str(int(syntax['zp_sigma'])),
                                 marker = 'o',
                                 linestyle="None",
                                 color = 'blue',
                                 capsize=1,
                                 capthick=1)

                    [bar.set_alpha(0.3) for bar in bars]
                    [cap.set_alpha(0.3) for cap in caps]



                    ax2.set_ylabel('Instrumental Magnitude')
                    ax2.set_xlabel('Zeropoint Magnitude')
                    ax2.invert_yaxis()

                    n, bins, patches = ax3.hist(zpoint_clip,
                             bins = 'auto',
                             color = 'green',
                             label = 'Zeropoint Distribution',
                             density = True)

                    mean = zp_wa[0]
                    sigma = 0.1

                    bins_fix = [(bins[i-1] + bins[i]) / 2  for i in range(1,len(bins))]

                    try:
                        popt,pcov = curve_fit(gauss,bins_fix,n, p0=[np.max(n),mean,sigma])
                        r = np.arange(np.min(bins_fix)-0.1,np.max(bins_fix)+0.1,0.01)
                        ax3.plot(r,gauss(r,*popt),'r',label='Gaussian fit')
                        ax3.xlim(np.min(bins_fix)-0.1,np.max(bins_fix)+0.1)
                    except:
                        pass

                    ax3.set_xlabel('Zeropoint Magnitude')
                    ax3.set_ylabel('Probability')

                    for axes in [ax1,ax2,ax3]:
                        axes.legend(loc = 'upper left',fancybox=True,frameon = False)

                    if syntax['save_zp_plot']:
                        fig_zeropoint.savefig(cur_dir + '/' +'zp_'+str(base.split('.')[0])+'.pdf',format = 'pdf')

                    plt.close(fig_zeropoint)

                except Exception as e:
                    logging.exception(e)
                    plt.close(fig_zeropoint)


# =============================================================================
# Limiting Magnitude
# =============================================================================
            if syntax['do_mag_lim']:

                from scipy.stats import binned_statistic

                # Create copy of image to avoid anything being written to original image
                image_copy = image.copy()
                try:

                    # Bin size in magnitudes
                    b_size = 0.5

                    lim_err =  sigma_mag_err(syntax['lim_SNR'])

                    # Sort values by filter
                    c_mag_lim = c.sort_values(by = [str('cat_'+use_filter)])

                    # x values: autophot magnitudes
                    x = c_mag_lim[str(use_filter)].values
                    x_err = c_mag_lim[str(use_filter)+'_err'].values

                    # y values: absolute differnece between catalog magnitude and autophots
                    y =  c_mag_lim[str(use_filter)].values - c_mag_lim[str('cat_'+use_filter)].values
                    y_err =  np.sqrt(c_mag_lim[str(use_filter)+'_err'].values**2 + c_mag_lim[str('cat_'+use_filter)+'_err'].values**2)

                    # remove nans
                    idx = (np.isnan(x)) | (x<=0.0)

                    x = x[~idx]
                    y = y[~idx]

                    if len(x)  <= 1 :
                        logging.warning('Magnitude diagram not found')
                        mag_limit = np.nan

                    else:

                        fig_magnitude = plt.figure(figsize = set_size(500,0.5))

                        grid = GridSpec(1, 2 ,wspace=0.05, hspace=0,width_ratios = [1,0.25])

                        ax1 = fig_magnitude.add_subplot(grid[0 , 0])

                        ax2 = fig_magnitude.add_subplot(grid[0, 1],sharey = ax1)

                        s, edges, _ = binned_statistic(x,y,
                                                       statistic='median',
                                                       bins=np.linspace(np.nanmin(x),np.nanmax(x),int((np.nanmax(x)-np.nanmin(x))/b_size)))
                        bin_centers = np.array(edges[:-1]+np.diff(edges)/2)

                        ax1.hlines(s,edges[:-1],edges[1:], color="black")

                        ax1.scatter(bin_centers, s,
                                    c="green",
                                    marker = 's',
                                    label = 'Median Bins',
                                    zorder = 10)

                        markers, caps, bars = ax1.errorbar(x,y,
                                                        xerr = x_err,yerr = y_err,
                                                        color = 'red',
                                                        ecolor = 'black',
                                                        marker = 'o',
                                                        ls = '',
                                                        label = r'$\Delta \ Mag$',
                                                        zorder = 90)
                        [bar.set_alpha(0.3) for bar in bars]
                        [cap.set_alpha(0.3) for cap in caps]

                        ax1.axhline(lim_err,
                                    label = r'$SNR_{err}$ cuttof [%d$\sigma$]' % syntax['lim_SNR'],
                                    linestyle = '--',
                                    color = 'black')

                        ax2.axhline(lim_err,
                                    linestyle = '--',
                                    color = 'black')

                        ax1.axhline(-1*lim_err,linestyle = '--',color = 'black')
                        ax2.axhline(-1*lim_err,linestyle = '--',color = 'black')

                        ax1.set_ylim(-0.95,0.95)


                        ax1.set_ylabel(r'Magnitude difference')
                        ax1.set_xlabel(r'%s band catalog magnitudes' % use_filter)
                        '''
                        Find magnitude where all over where all proceeding magnitude bins are greater than the error SNR cutoff
                        '''
                        for i in range(len(s)):
                            t = s[i]
                            if abs(t)>lim_err:
                                for j in range(i+1,len(s)):
                                    if abs(s[j]) < lim_err:
                                        break
                                else:
                                    ax1.axhline(round(t,3),
                                                linestyle = ':',
                                                label ='$M_{Lim} / Guess$',
                                                color = 'blue')
                                    break
                        # Set magnitude limit with inall i'th value here
                        mag_limit = bin_centers[i]

                        # Soltuion to finx density issue
                        weights = np.ones_like(y)/float(len(y))

                        # print(y)


                        n, bins, patches = ax2.hist(y,
                                                    weights = weights,
                                                    bins = int(np.ceil(len(y)/3)),
                                                    facecolor = 'green',
                                                    label = 'Zeropoint Distribution',
                                                    density = False,
                                                    orientation = 'horizontal')

                        ax2.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
                        ax2.set_xlabel('Probability')
                        ax2.yaxis.tick_right()


                        ax1.set_xlim(np.nanmin(x)-0.5,np.nanmax(x))

                        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

                        ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
                        ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
                        ax2.set_ylabel('')

                        ax1.legend(fancybox=True,
                                   ncol = 3,
                                   bbox_to_anchor=(0, 1.01, 1, 0),
                                   loc = 'lower center',
                                   frameon=False
                                    )


                        ax1.axhspan(lim_err,1, alpha=0.3, color='gray')

                        ax1.axhspan(-lim_err,-1, alpha=0.3, color='gray')

                        ax1.text(0.5,lim_err+0.2,
                            'Over Luminous',
                            va = 'bottom',
                            ha = 'center',
                            transform=ax1.get_yaxis_transform(),
                            color = 'black',
                            rotation = 0)

                        ax1.text(0.5,-lim_err-0.2,
                            'Under Luminous',
                            va = 'top',
                            ha = 'center',
                            transform=ax1.get_yaxis_transform(),
                            color = 'black',
                            rotation = 0)



                        if syntax['save_mag_lim_plot']:
                            fig_magnitude.savefig(os.path.join(cur_dir,'mag_lim_'+str(base.split('.')[0])+'.pdf'),
                                                  # bbox_inches='tight',
                                                  )

                        plt.close(fig_magnitude)


                except Exception as e:
                    logging.exception(e)
                    mag_limit = np.nan
                    # test_mag = np.nan
                    try:
                        plt.close(fig_magnitude)
                    except:
                        pass
                    pass

                logging.info('Approx. limiting magnitude: %s ' % str(round(mag_limit,3)))

                if target_x_pix < 0 or target_x_pix> image.shape[1] or target_y_pix < 0 or target_y_pix> image.shape[0] :
                    raise Exception ( ' *** EXITING - Target pixel coordinates outside of image [%s , %s] *** ' % (int(target_x_pix), int(target_y_pix)))

                elif do_ap or syntax['do_ap_phot']:
                    # test_mag = mag_limit  - syntax['zp']
                    # SNR_mock_target = [np.nan]

                    '''
                    if ap_phot selected - sets magnitude limit to nan <- will fix
                    '''
                    mag_limit = np.nan


                image_copy = image.copy()
                if np.isnan(mag_limit):
                    mag_limit = 20

# =============================================================================
# Do photometry on all sources
# =============================================================================
            if syntax['do_all_phot']:
                logging.info(' --- Perform photometry on all sources in field ---')

                _,df_all,_= find.fwhm(image,syntax,sigma_lvl = syntax['do_all_phot_sigma'],fwhm = mean_fwhm)

                ra_all,dec_all = w1.all_pix2world(df_all.x_pix.values,df_all.y_pix.values,1 )

                df_all['RA'] = ra_all
                df_all['DEC'] = dec_all

                photfile = 'phot_filter_%s_sigma_%d_%s.csv' % (use_filter,syntax['do_all_phot_sigma'],str(base.split('.')[0]))

                if do_ap:


                    positions  = list(zip(df_all.x_pix.values,df_all.y_pix.values))

                    target,target_bkg = ap_phot(positions,
                                                image,
                                                radius = syntax['ap_size']    * mean_fwhm,
                                                r_in   = syntax['r_in_size']  * mean_fwhm,
                                                r_out  = syntax['r_out_size'] * mean_fwhm)

                    source_flux = (target/exp_time)
                    source_bkg = (target_bkg/exp_time)

                    SNR_sources = SNR(source_flux,source_bkg,exp_time,0,syntax['ap_size']* mean_fwhm,gain,0)
                    df_all['snr'] = SNR_sources
                    df_all[use_filter] = mag(source_flux,zp_wa[0] ) + ap_corr

                    mag_err = sigma_mag_err(SNR_sources)
                    df_all[use_filter+'_err'] =  np.sqrt(mag_err**2 + zp_wa[1]**2)



                else:
                    positions = df_all[['x_pix','y_pix']]

                    psf_sources,_ = psf.fit(image,
                                            positions,
                                            r_table,
                                            syntax,
                                            mean_fwhm)

                    psf_sources_phot,_ = psf.do(psf_sources,
                                                r_table,
                                                syntax,
                                                mean_fwhm)

                    sources_flux = np.array(psf_sources_phot.psf_counts/exp_time)

                    sources_err = np.array(psf_sources_phot.psf_counts_err/exp_time)

                    SNR_sources = np.array(sources_flux/sources_err)

                    mag_err = sigma_mag_err(SNR_sources)

                    ra_all,dec_all = w1.all_pix2world(psf_sources_phot.x_pix.values,psf_sources_phot.y_pix.values,1 )

                    df_all = pd.DataFrame([])
                    df_all['RA'] = ra_all
                    df_all['DEC'] = dec_all

                    df_all['snr'] = SNR_sources
                    df_all['flux_inst'] = psf_sources['H_psf']
                    df_all[use_filter] = mag(sources_flux,zp_wa[0] )

                    mag_err = sigma_mag_err(SNR_sources)
                    df_all[use_filter+'_err'] =  np.sqrt(mag_err**2 + zp_wa[1]**2)



                try:
                    df_all.to_csv(syntax['write_dir']+photfile,index = False)
                    logging.info('Photometry of all sources saved as: %s' % str(base.split('.')[0])+'.csv')
                except Exception as e:


                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    logging.info(exc_type, fname1, exc_tb.tb_lineno,e)
                    # sys.stdout = default_stdout
                    logging.info('Warning - Table could not be saved to csv')
                    # flog.close()
                    pass

# =============================================================================
# Get Template
# =============================================================================
            try:
                PSF_model = psf.fit(image,
                                    c,
                                    r_table,
                                    syntax,
                                    mean_fwhm,
                                    return_psf_model = True)
            except:
                PSF_model = None

            # print(PSF_model)
            # print(PSF_model.shape)
            subtraction_ready = False
            if syntax['do_subtraction']:


                try:
                    # Pan_starrs template images use some old WCS keywrods, astropy can handle them just for cleaniness
                    warnings.filterwarnings('ignore')

                    ra = target_coords.ra.degree
                    dec = target_coords.dec.degree

                    # Get pixel scale - output in deg
                    image_scale = np.max(wcs.utils.proj_plane_pixel_scales(w1))

                    # size of image in arcseonds
                    size = round(image_scale * 3600 * np.nanmax(image.shape))

                    # Aligning images with astroalign https://github.com/toros-astro/astroalign

                    template_found = False

                    if use_filter in ['g','r','i','z','u']:
                        use_filter_template = use_filter+'p'
                    else:
                        use_filter_template = use_filter




                    if syntax['use_user_template']:
                        logging.info('Using user template')
                        if not os.path.exists(syntax['fits_dir'] + '/templates'):
                            logging.info('Templates folder not found')
                        else:
                            logging.info('Found user templates')


                            if not os.path.exists(syntax['fits_dir'] + '/templates/' + use_filter_template + '_template'):
                                logging.info('cannot find User filter template')
                            else:
                                list_dir = os.listdir(syntax['fits_dir'] + '/templates/' + use_filter_template + '_template')
                                fits_list_dir = [i for i in list_dir if i.split('.')[-1] in ['fits','fts','fit']]

                                if len(fits_list_dir) >1:
                                    logging.info(syntax['fits_dir'] + '/templates/' + use_filter_template + '_template')
                                    logging.info(os.listdir(syntax['fits_dir'] + '/templates/' + use_filter_template + '_template'))
                                    logging.info('Check template folder - too many files in fodler')
#                                else:
                                fpath_template = syntax['fits_dir'] + '/templates/' + use_filter_template+ '_template/' + fits_list_dir[0]
#                                    subtraction_ready = True
                                template_found = True

                    if syntax['get_template'] and not template_found:
                        logging.info('Searching for template ...')
#
#                        # Template retrival from 2mass - shite because it's too faint
                        if syntax['catalog'] == '2mass':
                            try:
                                from astroquery.skyview import SkyView

                                # https://astroquery.readthedocs.io/en/latest/skyview/skyview.html
                                hdu  = SkyView.get_images(target_coords,survey = ['2MASS-'+use_filter_template.upper()],coordinates = 'ICRS',radius = size * u.arcsec)
                                fits.writeto(fpath.replace(fname_ext,'_template')+'no_rot'+fname_ext, hdu[0][0].data.astype(np.single), headinfo, overwrite=True,output_verify = 'silentfix+ignore')

                            except Exception as e:

                                    logging.exception(e)


                        # Template retrival from panstarrs
                        if syntax['catalog'] == 'pan_starrs' or syntax['catalog'] == 'skymapper':

                            if os.path.isfile(syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/'+'template_'+use_filter+'_retrieved'+fname_ext):
                                logging.info('Found previously retrived template:'+str(syntax['fits_dir'].split('/')[-1])+'/templates/')
                                template_found = True
                                fpath_template = syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/'+'template_'+use_filter_template+'_retrieved'+fname_ext
                            else:

                                pan_starrs_pscale = 0.25 # arcsec per pxiel

                                logging.info('Searching for template on PanSTARRS')

                                fitsurl = get_pstars(float(ra), float(dec), size=int(size/pan_starrs_pscale), filters=use_filter)

                                with fits.open(fitsurl[0],ignore_missing_end = True,lazy_load_hdus = True) as hdu:
                                    try:

                                        hdu.verify('silentfix+ignore')
                                        headinfo_template = hdu[0].header
                                        template_found  = True


                                        # save templates into original folder under the name template
                                        pathlib.Path(syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/').mkdir(parents = True, exist_ok=True)
                                        fits.writeto(syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/'+'template_'+use_filter_template+'_retrieved'+fname_ext,
                                                     hdu[0].data.astype(np.single),
                                                     headinfo_template,
                                                     overwrite=True,
                                                     output_verify = 'silentfix+ignore')
                                        if os.path.isfile(syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/'+'template_'+use_filter_template+'_retrieved'+fname_ext):
                                            logging.info('Retrived template saved in: '+str(syntax['fits_dir'].split('/')[-1])+'/templates/')
                                            fpath_template = syntax['fits_dir']+'/'+'templates/'+ use_filter_template + '_template/'+'template_'+use_filter_template+'_retrieved'+fname_ext
                                            template_found = True

                                    except Exception as e:
                                        logging.exception(e)




                    if not template_found:
                        logging.info('Template not found')
                    else:
                        with fits.open(fpath_template,ignore_missing_end = True,lazy_load_hdus = True) as hdu:
                            headinfo_template = hdu[0].header

                            try:
                                if syntax['use_astroalign']:
                                    try:
                                        import astroalign as aa



                                        # aa.MIN_MATCHES_FRACTION = 0.5
                                        # aa.NUM_NEAREST_NEIGHBORS = 5
                                        # aa.PIXEL_TOL = 4

                                        logging.info('Aligning via Astro Align')
                                        aligned_image, footprint = aa.register(hdu[0].data.astype(float),
                                                                                image.astype(float))
                                        aligned_image[np.isnan(aligned_image)] = 1e-30

                                    except Exception as e:
                                        logging.info('ASTRO ALIGN failed: %s' % e)
                                        syntax['use_reproject_interp'] = True

                                if syntax['use_reproject_interp']:
                                    try:
                                        logging.info('Aligning via WCS')

                                        from reproject import reproject_interp
                                        aligned_image, footprint = reproject_interp(hdu[0], headinfo)
                                        aligned_image[(np.isnan(aligned_image)) | (aligned_image==0) ] = 1e-30
                                    except Exception as e:
                                        logging.info('reproject_interp failed: %s' % e)
                                        raise Exception

                                fits.writeto(fpath.replace(fname_ext,'_template')+fname_ext, aligned_image.astype(np.single), headinfo_template, overwrite=True,output_verify = 'silentfix+ignore')
                                fpath_template = fpath.replace(fname_ext,'_template')+fname_ext
                                subtraction_ready = True

                            except Exception as e:
                                logging.exception(e)

                    warnings.filterwarnings("default")

                    if os.path.isfile(fpath.replace(fname_ext,'_template')+fname_ext):
                        logging.info('Template saved as: %s' %os.path.basename(fpath.replace(fname_ext,'_template')))

                except Exception as e:
                    logging.error('Error with Template aquisiton')
                    logging.exception(e)


# Check for template image

            if syntax['get_template']:
                if not os.path.isfile(fpath.replace(fname_ext,'_template')+fname_ext):
                    logging.info('Template file NOT found')
                else:
                    fpath_template = fpath.replace(fname_ext,'_template')+fname_ext
                    subtraction_ready = True

# =============================================================================
# Image subtraction using HOTPANTS
# =============================================================================

            if syntax['do_subtraction'] and subtraction_ready:
                logging.info('Performing image subtraction using HOTPANTS')

                fpath_sub = HOTterPANTS(fpath,fpath_template,syntax,psf = PSF_model)

# =============================================================================
# Perform photometry on target
# =============================================================================


            if syntax['do_phot']:

                dx = syntax['dx']
                dy = syntax['dy']

                if syntax['phot_on_sub'] and subtraction_ready:
                    image    = getimage(fpath_sub)
                    logging.info('Target photometry on subtracted image')
                else:
                    logging.info('Target photometry on original image')

                image_copy  = image.copy()

                target_x_pix_TNS, target_y_pix_TNS = w1.all_world2pix(syntax['target_ra'], syntax['target_dec'], 1)

                close_up = image_copy[int(target_y_pix_TNS - syntax['scale']): int(target_y_pix_TNS + syntax['scale']),
                                      int(target_x_pix_TNS - syntax['scale']): int(target_x_pix_TNS + syntax['scale'])]
                from autophot.packages.rm_bkg import rm_bkg

                close_up,bkg_surface = rm_bkg(close_up,syntax,close_up.shape[1]/2,close_up.shape[0]/2)


                x = np.arange(0,2*syntax['scale'])
                xx,yy= np.meshgrid(x,x)

                # try:

                #     pars = lmfit.Parameters()
                #     pars.add('A',value = np.nanmax(close_up),min = 1e-3)
                #     pars.add('x0',value = close_up.shape[1]/2,min = (close_up.shape[1]/2)-dx ,max = (close_up.shape[1]/2)+dx)
                #     pars.add('y0',value = close_up.shape[0]/2,min = (close_up.shape[0]/2)-dy ,max = (close_up.shape[0]/2)+dy)
                #     # pars.add('sky',value = 0)


                #     if syntax['use_moffat']:
                #         pars.add('alpha',value = syntax['image_params']['alpha'],
                #                  min = 0,
                #                  max = 25,
                #                  vary = False)
                #         pars.add('beta',value = syntax['image_params']['alpha'],
                #                  min = 0,
                #                  vary = syntax['vary_moff_beta'])


                #     else:
                #         pars.add('sigma',value = syntax['image_params']['sigma'],
                #                  min = 0,
                #                  max = gauss_fwhm2sigma(syntax['max_fit_fwhm']),
                #                  vary = False)

                #     if syntax['use_moffat']:
                #         def residual(p):
                #             p = p.valuesdict()
                #             return (close_up - moffat_2d((xx,yy),p['x0'],p['y0'],0,p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(close_up.shape)).flatten()

                #     else:
                #         def residual(p):
                #             p = p.valuesdict()
                #             return (close_up - gauss_2d((xx,yy),p['x0'],p['y0'],0,p['A'],dict(sigma=p['sigma'])).reshape(close_up.shape)).flatten()

                #     mini = lmfit.Minimizer(residual,
                #                            pars,nan_policy = 'omit')

                #     result = mini.minimize()

                #     target_x_pix_corr= result.params['x0'].value - syntax['scale']
                #     target_y_pix_corr= result.params['y0'].value - syntax['scale']

                #     target_x_pix =  target_x_pix_corr + target_x_pix_TNS
                #     target_y_pix =  target_y_pix_corr + target_y_pix_TNS

                #     close_up = image[int(target_y_pix - syntax['scale']): int(target_y_pix + syntax['scale']),
                #                       int(target_x_pix - syntax['scale']): int(target_x_pix + syntax['scale'])]

                # except Exception as e:

                #     logging.error('Could not fit gaussian profile to target - using TNS coords')
                #     logging.exception(e)

                #     target_x_pix = target_x_pix_TNS
                #     target_y_pix = target_y_pix_TNS

                #     target_x_pix_corr = syntax['scale']
                #     target_y_pix_corr = syntax['scale']

# =============================================================================
# Subtraction image
# =============================================================================
                if syntax['save_subtraction_quicklook'] and subtraction_ready:
                    fig_sub = plt.figure(figsize = set_size(240,aspect = 1))

                    ax1 = fig_sub.add_subplot(121)
                    ax2 = fig_sub.add_subplot(122)
                    plt.subplots_adjust(hspace=0.0,wspace=0.3)

                    image_sub_tmp = fits.getdata(fpath_sub)


                    vmin,vmax = (ZScaleInterval(nsamples = 1200)).get_limits(image_sub_tmp)

                    ax1.imshow(image_sub_tmp,
                               vmin = vmin,
                               vmax = vmax,
                               aspect ='auto',
                               interpolation = 'nearest',
                               origin = 'lower',
                               cmap = 'Greys')

                    # ax.set_zlim(vmin,vmax)

                    ax1.scatter([target_x_pix],[target_y_pix],marker = 'D',
                               facecolor = 'None',
                               color = 'GOLD',
                               linewidth = 0.8,
                               s = 25,
                               label = 'Target: %s' % tname)

                    im = ax2.imshow(close_up,
                                # vmin = vmin,
                                # vmax = vmax,
                               aspect ='auto',
                               # interpolation = 'nearest',
                               origin = 'lower',
                               cmap = 'Greys')

                    ax2.scatter(close_up.shape[0]/2,close_up.shape[0]/2,
                                marker = '+',
                               # facecolor = 'None',
                               color = 'GOLD',
                               linewidth = 0.8,
                               s = 25,
                               label = 'Target: %s' % tname)

                    ax1.set_title('Host Subtrated Image')
                    ax2.set_title('Transient Cutout')
                    ax1.set_xlabel('X PIXEL')
                    ax1.set_ylabel('Y PIXEL')

                    ax2.set_xlabel('X PIXEL')
                    ax2.set_ylabel('Y PIXEL')

                    ax1.legend(loc = 'best')

                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cb = fig_sub.colorbar(im, cax=cax)
                    cb.ax.set_ylabel('Counts', rotation=270,labelpad = 10)

                    cb.formatter.set_powerlimits((0, 0))
                    cb.ax.yaxis.set_offset_position('left')
                    cb.update_ticks()

                    fig_sub.savefig(cur_dir + '/' +os.path.basename(fpath.replace(fname_ext,'_subtraction_QUICKLOOK'))+'.pdf',
                                    bbox_inches='tight',)

                    plt.close()



# =============================================================================
#               Work on target location
# =============================================================================

                pars = lmfit.Parameters()
                pars.add('A',value = np.nanmax(close_up)*0.75,min = 1e-5)
                pars.add('x0',value = close_up.shape[1]/2,
                         min = close_up.shape[1]/2 - dx,max = close_up.shape[1]/2 +dx)
                pars.add('y0',value = close_up.shape[0]/2,
                         min = close_up.shape[0]/2 - dx,max = close_up.shape[0]/2 +dx)
                pars.add('sky',value = np.nanmedian(close_up))


                close_up = image[int(target_y_pix - syntax['scale']): int(target_y_pix + syntax['scale']),
                                 int(target_x_pix - syntax['scale']): int(target_x_pix + syntax['scale'])]

                if syntax['use_moffat']:
                    pars.add('alpha',value = syntax['image_params']['alpha'],
                             min = 0,
                             max = gauss_fwhm2sigma(syntax['max_fit_fwhm']) )
                    pars.add('beta',value = syntax['image_params']['beta'],
                             min = 0,
                             vary = syntax['vary_moff_beta']  )

                else:
                    pars.add('sigma',value = syntax['image_params']['sigma'],
                             min = 0,
                             max = gauss_fwhm2sigma(syntax['max_fit_fwhm']) )

                if syntax['use_moffat']:
                    def residual(p):
                        p = p.valuesdict()
                        return (close_up - moffat_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(close_up.shape)).flatten()

                else:
                    def residual(p):
                        p = p.valuesdict()
                        return (close_up - gauss_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(sigma=p['sigma'])).reshape(close_up.shape)).flatten()

                mini = lmfit.Minimizer(residual, pars,nan_policy = 'omit')
                result = mini.minimize(method = 'least_squares')

                if syntax['use_moffat']:

                    fitting_model = moffat_2d
                    fitting_model_fwhm = moffat_fwhm

                else:
                    fitting_model = gauss_2d
                    fitting_model_fwhm = gauss_sigma2fwhm


                if syntax['use_moffat']:
                    target_fwhm = fitting_model_fwhm(dict(alpha=result.params['alpha'],beta=result.params['beta']))
                else:
                    target_fwhm = fitting_model_fwhm(dict(sigma=result.params['sigma']))


                target_x_pix_corr =  result.params['x0'].value
                target_y_pix_corr =  result.params['y0'].value



                positions  = list(zip([target_x_pix_corr],[target_y_pix_corr]))

                target,target_bkg = ap_phot(positions,
                                             close_up,
                                             radius = syntax['ap_size']    * mean_fwhm,
                                             r_in   = syntax['r_in_size']  * mean_fwhm,
                                             r_out  = syntax['r_out_size'] * mean_fwhm)

                target = (target/exp_time)[0]
                target_bkg = (target_bkg/exp_time)[0]

                # print(target,target_bkg)

                if syntax['adjust_SN_loc']:
                    target_x_pix =  target_x_pix_corr  - 0.5*close_up.shape[1]  + target_x_pix_TNS
                    target_y_pix =  target_y_pix_corr  - 0.5*close_up.shape[0]  + target_y_pix_TNS
                else:
                    target_x_pix = target_x_pix_TNS
                    target_y_pix = target_y_pix_TNS

                SNR_target = SNR(target,target_bkg,exp_time,0,syntax['ap_size']*mean_fwhm,gain,0)

                if not do_ap and not syntax['do_ap_phot'] :

                    if mag(target,0) - approx_psf_mag < 0.25:
                        if not syntax['force_psf']:
                            logging.warning('PSF not applicable')
                            logging.warning('target mag [%.3f] <  PSF mag [%.3f]' % (mag(target,0),approx_psf_mag))
                            logging.info('set "force_psf" = True to fix')

                            do_ap = True
                            ap_corr = ap_corr_base



                if syntax['save_target_plot'] and (do_ap or (syntax['do_ap_on_sub'] and subtraction_ready)):

                    fig_target = plt.figure(figsize = set_size(240,aspect = 1))

                    ax = fig_target.add_subplot(111)


                    im = ax.imshow(close_up,
                                   interpolation = 'nearest',
                                   origin = 'lower',
                                   aspect = 'equal',
                                   )

                    ax.scatter(target_x_pix_corr,target_y_pix_corr,
                               label ='Centroid fit',
                               marker = '+',
                               s =20,
                               color = 'red')

                    circ1 = Circle((target_x_pix_corr,target_y_pix_corr),syntax['ap_size'] * mean_fwhm,
                                   label = 'Aperture',
                                   fill=False,
                                   color = 'red')
                    circ2 = Circle((target_x_pix_corr,target_y_pix_corr),syntax['r_in_size'] * mean_fwhm,
                                   label = 'Background Annulus',
                                   fill=False,
                                   color = 'red',
                                   linestyle = '--')
                    circ3 = Circle((target_x_pix_corr,target_y_pix_corr),syntax['r_out_size'] * mean_fwhm,
                                   # label = 'Outer Annulus',
                                   fill=False,
                                   color = 'red',
                                   linestyle = '--')

                    ax.add_patch(circ1)
                    ax.add_patch(circ2)
                    ax.add_patch(circ3)

                    ax.set_ylabel('Y PIXEL')
                    ax.set_xlabel('X PIXEL')

                    ax.legend(fancybox=True,
                              ncol = 2,
                              bbox_to_anchor=(0, 1.2, 1, 0),
                              loc = 'upper center',
                              frameon=False)

                    divider = make_axes_locatable(ax)

                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cb = fig_target.colorbar(im, cax=cax)
                    cb.ax.set_ylabel('Counts', rotation=270,labelpad = 10)

                    cb.formatter.set_powerlimits((0, 0))
                    cb.ax.yaxis.set_offset_position('left')
                    cb.update_ticks()

                    fig_target.savefig(cur_dir + '/' +'target_'+str(base.split('.')[0])+'.pdf',
                                       bbox_inches='tight'
                                       )

                    plt.close(fig_target)

                if syntax['do_psf_phot'] and not (syntax['do_ap_on_sub'] and subtraction_ready) and not do_ap:

                    logging.info('Fitting PSF to Target')

                    c_target = pd.DataFrame(data = [[target_x_pix,target_y_pix]],columns = ['x_pix','y_pix'])

                    c_psf_target1,_ = psf.fit(image,
                                              c_target,
                                              r_table,
                                              syntax,
                                              mean_fwhm,
                                              save_plot = syntax['save_target_plot'],
                                              show_plot = syntax['plot_target'] ,
                                              fname = str(base.replace(fname_ext,'')),
                                              return_fwhm = True)


                    c_psf_target,_ = psf.do(c_psf_target1,
                                            r_table,
                                            syntax,
                                            mean_fwhm)



                    target = np.array(c_psf_target.psf_counts/exp_time)[0]

                    target_x_pix = c_psf_target['x_fitted']
                    target_y_pix = c_psf_target['y_fitted']


                    target_bkg = np.array(c_psf_target.bkg/exp_time)[0]


                    target_err = c_psf_target.psf_counts_err/exp_time

                    SNR_target = np.array(target/target_err)[0]

                    target_fwhm = c_psf_target['target_fwhm'].values[0]
                else:
                    logging.info('Doing Aperture Photometry on Target')
                    target_err = 0

# =============================================================================
# Limiting Magnitude
# =============================================================================

                lmag_check = True

                try:
                    syntax['image_radius']

                except:
                    logging.warning('Image Radius not defined')
                    syntax['image_radius'] = 1.5 * mean_fwhm

                if syntax['get_lim_mag_prob']:

                    if SNR_target > 10 and abs(target_fwhm - mean_fwhm) < 0.5 and not syntax['force_lmag'] :

                            lmag = np.nan
                            output.update({'lmag':lmag})

                            logging.info('SNR = %.f - skipping limiting magnitude' % SNR_target)
                            lmag_check = False
                    else:

                        if abs(target_fwhm - mean_fwhm) > 0.5:
                            logging.warning('Discrepancy in FWHM of %.1f pixels' % abs(target_fwhm - mean_fwhm))


                        if SNR_target <10:
                            logging.warning('SNR = %.f - checking limiting magnitude' % SNR_target)

                        expand_scale = int(np.floor(2.5*syntax['image_radius']))+0.5

                        # cut out an Expanded area around target x and y pixel - needs to be even

                        while expand_scale < syntax['scale']:
                            expand_scale+=1

                            
                        if (expand_scale - syntax['scale'] )%2 !=0:
                            expand_scale+=1

                        # get close up at bigger scale to allow for injected sources to be well spaced
                        close_up_expand = image_copy[int(target_y_pix - expand_scale): int(target_y_pix + expand_scale),
                                                     int(target_x_pix - expand_scale): int(target_x_pix + expand_scale)]

                        try:
                             model = model_psf
                             r_table = r_table
                        except:
                            model = None
                            r_table = None

                        lmag_inst,syntax = limiting_magnitude_prob(syntax,close_up_expand,model,r_table)

                        lmag = lmag_inst + zp_wa[0]

                        output.update({'lmag':lmag})




                # =============================================================================
                # Error on target magnitude
                # =============================================================================

                SNR_error = sigma_mag_err(SNR_target)

                fit_error  = mag(target,0) - mag(target+target_err,0)

                target_mag_err = SNR_error + fit_error[0]


                location_offset = pix_dist(target_x_pix,target_x_pix_TNS,target_y_pix,target_y_pix_TNS)

                logging.info('Pixel Offset: %.3f' % location_offset)

                location_offset_dict = {'pixel_offset':location_offset}

                # =============================================================================
                # Output
                # =============================================================================
                # Don't include aperture correction unless aperature photometry is used
                if not do_ap:
                    ap_corr = 0
                corrections_included = ''

                mag_target = mag(target,zp_wa[0]) + ap_corr



                mag_inst={use_filter+'_inst':mag(target,0)}
                mag_inst_err={use_filter+'_inst_err':target_mag_err}

                zp={'zp_'+use_filter:zp_wa[0]}
                zp_err={'zp_'+use_filter+'_err':zp_wa[1]}

                fwhm_out={'fwhm':mean_fwhm,
                          'fwhm_err':mean_fwhm_err}

                target_fwhm_out = {'target_fwhm':target_fwhm}

                SNR_dict = {'SNR':SNR_target}

                time_exe = {'time':str(datetime.datetime.now())}

                # if mag_target[use_filter] == 0.0:
                    # logging.info(' Target not seen - SNR = 0 ')
                    # mag_target[use_filter] = [np.nan]

                target_locx = {'xpix':target_x_pix}
                target_locy = {'ypix':target_y_pix}


                # Print outbursts
                logging.info('\nTarget counts: %.3f +/- %.3f'% (target,target_err))
                logging.info('Target SNR: %.3f +/- %.3f' % (SNR_target,SNR_error))
                logging.info('Instrumental Magnitude: %.3f +/- %.3f' % (mag(target,0)[0],fit_error))
                logging.info('Zeropoint: %.3f +/- %.3f' % (zp_wa[0],zp_wa[1]))



                # Apply extinction airmass correction
                if syntax['apply_airmass_extinction']:
                    from autophot.packages.apply_airmass_extinction import apply_airmass_extinction

                    telescope_location = tele_syntax[telescope][inst_key][inst]['location']
                    syntax['location'] = telescope_location

                    airmass_correction = apply_airmass_extinction(syntax)

                    if airmass_correction == 'no_entry':
                        logging.info('No found airmass extinction data for site - skipping' )
                    else:

                        logging.info('Airmass Correction: %.3f' % airmass_correction)

                        output.update({'AIR':-1 * airmass_correction})

                        corrections_included+='_AIR'

                        # add [subtract] airmass correction magnitude
                        mag_target-=airmass_correction

                    # output.update({use_filter + '_AP':'ap'})

                mag_target_dict = {use_filter+corrections_included:mag_target}

                mag_err = np.sqrt(target_mag_err**2 + zp_wa[1]**2)

                mag_target_err_dict = {use_filter+corrections_included+'_err':mag_err}






                if do_ap:
                    output.update({'method':'ap'})
                else:
                    output.update({'method':'psf'})

                if not np.isnan(lmag):

                    from autophot.packages.functions import beta_value

                    beta = beta_value(syntax['lim_SNR'],syntax['maglim_std'],target*exp_time,mean = syntax['maglim_mean'])

                    logging.info('Detection Probability [beta] %.2f%%:'%  (beta*100))


                    detection_beta = {'beta':beta}

                else:
                    detection_beta = {'beta':1}

                if subtraction_ready:
                    output.update({'subtraction':True})
                else:
                    output.update({'subtraction':False})


                output.update(target_locx)
                output.update(target_locy)
                output.update(mag_inst)
                output.update(mag_inst_err)
                output.update(zp)
                output.update(zp_err)
                output.update(mag_target_dict)
                output.update(mag_target_err_dict)
                output.update(SNR_dict)
                output.update(time_exe)
                output.update(fwhm_out)
                output.update(target_fwhm_out)
                output.update(detection_beta)
                output.update(location_offset_dict)


                if not lmag_check:
                    logging.info('Limiting Magnitude: skipped')
                else:
                    logging.info('Limiting Magnitude: %.3f' % lmag)

                logging.info('Target Magnitude: %.3f +/- %.3f ' % (mag_target,mag_err))


                # =============================================================================
                # Print message to tell about source detection
                # =============================================================================

                if lmag < mag_target or np.isnan(mag_target):
                    lim_mag_check = False
                else:
                    lim_mag_check = True

                if abs(target_fwhm - mean_fwhm) < 0.5:

                    fwhm_check = True
                else:

                    fwhm_check = False


                # =============================================================================
                # Print final message
                # =============================================================================

                if fwhm_check and lim_mag_check:
                    logging.info('\n*** Transient well detected ***\n')

                elif not lim_mag_check :
                    logging.info('\n*** Image is magnitude limited ***\n')

                elif not fwhm_check:
                    logging.info('\n*** FWHM discrepancy: %.3f pixels ***\n' % abs(target_fwhm - mean_fwhm))

                '''
                Calibration file used in reduction

                - used in color calibration
                '''

                c.round(6).to_csv(syntax['write_dir']+'image_calib_'+str(base.split('.')[0])+'_filter_'+str(use_filter)+'.csv',index = False)


                output_file = os.path.join(cur_dir,'out.csv')

                target_output = pd.DataFrame(output,columns=output.keys(), index=[0])

                target_output.round(6).to_csv(output_file,index=False)

            logging.info('Time Taken [ %s ]: %ss' % (str(os.getpid()),round(time.time() - start)))

            logging.info('Sucess: %s :: PID %s \n'%(str(base),str(os.getpid())))

            console.close()

            return output,base

        # Parent try/except statement for loop
        except Exception as e:
            logging.exception(e)
            logging.critical('Failure: '+ str(base) + ' - PID: '+str(os.getpid()))
            # # console.close()
            # print(cur_dir)
            # fail_dir = os.path.join(cur_dir,'fail')


            # print(fail_dir)

            # pathlib.Path(fail_dir).mkdir(parents = True, exist_ok=True)

            # shutil.move(cur_dir,fail_dir)
            # print('Moving contents to fail folder')


            return None,fpath

    except Exception as e:
        logging.exception(e)
        logging.critical('Fatal Error')
        logging.info('Failure: '+str(base) + ' - PID: '+str(os.getpid()))

        # fail_dir = os.path.join(new_output_dir,'fail')


        # print(fail_dir)

        # pathlib.Path(fail_dir).mkdir(parents = True, exist_ok=True)


        # if os.path.exists(fail_dir):
        #     os.remove(fail_dir)

        # shutil.move(cur_dir,fail_dir)
        # print('Moving contents to fail folder')

        files = (file for file in os.listdir(os.getcwd())
             if os.path.isfile(os.path.join(os.getcwd(), file)))

        for file in files:
             if fpath == file:
                 break

             if os.path.basename(fpath).split('.')[0] in file:
                   shutil.move(os.path.join(os.getcwd(), file),
                    os.path.join(cur_dir, file))

        try:
            console.close()
            c.to_csv('table_ERROR_'+str(base.split('.')[0])+'_filter_'+str(use_filter)+'.csv',index=False)
        except:
            pass

        return None,fpath
