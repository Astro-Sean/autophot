#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:41:07 2022

@author: seanbrennan
"""

class check_wcs():
    
    def __init__(self,fpath,image,header,default_input   ):
        self.fpath = fpath
        self.header = header
        self.image  = image
        self.default_input = default_input
        
        
        
    def remove(self,delete_keys = True):


        import logging
        
        import warnings
        from astropy.utils.exceptions import AstropyWarning

        # Filter out Astropy warnings
        warnings.filterwarnings("ignore", category=AstropyWarning, append=True)
        
        
        logger = logging.getLogger(__name__)
        
        logger.info('Removing any pre-existing WCS keys ')
        
        try:
            if self.header['UPWCS']:
                # print('Removed UPWCS key')
                del self.header['UPWCS']
        
        except:
            pass
        


        keywords = ['CD1_1','CD1_2', 'CD2_1','CD2_2', 'CRVAL1','CRVAL2', 'CRPIX1','CRPIX2',
                    'CUNIT1','CUNIT2', 'CTYPE1','CTYPE1','CTYPE2', 'WCSAXES','EQUINOX', 'LONPOLE','LATPOLE',
                    'CDELT1','CDELT2', 'A_ORDER', 'A_0_0', 'A_0_1','A_0_2', 'A_1_0','A_1_1',
                    'A_2_0', 'B_ORDER', 'B_0_0','B_0_1', 'B_0_2','B_1_0', 'B_1_1','B_2_0',
                    'AP_ORDER', 'AP_0_0','AP_0_1', 'AP_0_2','AP_1_0', 'AP_1_1','AP_2_0',
                    'BP_ORDER', 'BP_0_0','BP_0_1', 'BP_0_2','BP_1_0', 'BP_1_1','BP_2_0',
                    'PROJP1','PROJP3', 'RADECSYS', 'PV1_1','PV1_2', 'PV2_1','PV2_2', 'LTV1','LTV2',
                    'LTM1_1','LTM2_2', 'PC1_1','PC1_2', 'PC2_1','PC2_2', 'RADESYS',
                    'PV1_0', 'PV1_1', 'PV1_2', 'PV1_3', 'PV1_4', 'PV1_5', 'PV1_6', 'PV1_7',
                    'PV1_8', 'PV1_9', 'PV1_10', 'PV1_11', 'PV1_12', 'PV1_13', 'PV1_14',
                    'PV1_15', 'PV1_16', 'PV1_17', 'PV1_18', 'PV1_19', 'PV1_20','PV1_21','PV1_22',
                    'TNX_0_0', 'TNX_1_0', 'TNX_0_1', 'TNX_2_0', 'TNX_1_1', 'TNX_0_2', 'TNX_3_0',
                    'TNX_2_1', 'TNX_1_2', 'TNX_0_3', 'TNX_4_0', 'TNX_3_1', 'TNX_2_2', 'TNX_1_3',
                    'TNX_0_4', 'TNX_5_0', 'TNX_4_1', 'TNX_3_2', 'TNX_2_3', 'TNX_1_4', 'TNX_0_5',
                    'TNX_6_0', 'TNX_5_1', 'TNX_4_2', 'TNX_3_3', 'TNX_2_4', 'TNX_1_5', 'TNX_0_6',
                    'PC001001','PC001002','PC002001','PC002002', 'A_1_1', 'A_1_2', 'A_1_3', 'A_2_0', 'A_2_1', 'A_2_2', 'A_2_3', 'A_3_0', 'A_3_1', 'A_3_2', 'A_3_3',
    'B_1_1', 'B_1_2', 'B_1_3', 'B_2_0', 'B_2_1', 'B_2_2', 'B_2_3', 'B_3_0', 'B_3_1', 'B_3_2', 'B_3_3',
    'PV1_0', 'PV1_1', 'PV1_2', 'PV1_3', 'PV1_4', 'PV1_5', 'PV1_6', 'PV1_7', 'PV1_8', 'PV1_9', 'PV1_10',
    'PV2_0', 'PV2_1', 'PV2_2', 'PV2_3', 'PV2_4', 'PV2_5', 'PV2_6', 'PV2_7', 'PV2_8', 'PV2_9', 'PV2_10',
    'SIP_A', 'SIP_B', 'SIP_C', 'SIP_D',
    'SIP_AP', 'SIP_BP', 'SIP_CP', 'SIP_DP'
]

    
        for i in keywords:
    
            # if i.replace(i[0],'_') in self.header:
            #     del self.header[i.replace(i[0],'_')]
            try:
                if delete_keys == True:
                    try:
                        del self.header[i]
                    except:
                        continue
                else:
                    self.header.rename_keyword(i,i.replace(i[0],'_'))
                    continue
    
            except Exception as e:
                logger.exception(e)
                pass
    
    
        return self.header
    
    
    
    def plate_solve(self,solvefieldExe=None,downsample=2,cpulimit = 30,xyList = None):


        import subprocess
        import os
        import numpy as np
        from functions import border_msg,get_image,get_header
        import signal
        import time
        import logging
        import warnings
        
        from astropy.io import fits
        from astropy import wcs
        
                
        import warnings
        from astropy.utils.exceptions import AstropyWarning

        # Filter out Astropy warnings
        warnings.filterwarnings("ignore", category=AstropyWarning, append=True)
        

        logging.info(border_msg('Solving for WCS with Astrometry.net'))
        
        if  xyList :
            logging.info('Using coorindate list')
            
        logger = logging.getLogger(__name__)
        
        pixel_scale = self.default_input['pixel_scale']
        target_ra = self.default_input['target_ra']
        target_dec = self.default_input['target_dec']
        search_radius = self.default_input['wcs']['search_radius']
           
    
        try:
            
            # get filename from filepath used to name new WCS fits file contained WCS header with values
            base = os.path.basename(self.fpath)
            base = os.path.splitext(base)[0]
          
    
            dirname = os.path.dirname(self.fpath)
    
            # new of output.fits file
            wcs_file = os.path.join(dirname,base.replace('sources_','') + ".wcs.fits")
    
            # location of executable [/Users/seanbrennan/AutoPhot_Development/AutoPHoT/astrometry.net/bin/solve-field]
            exe = str(solvefieldExe)
            
            
            if exe == str(None):
                raise Exception('Please enter "solve_field_exe_loc" to solve for WCS')
                # return np.nan
    
    
            # Guess image scale if f.o.v is not known
            if pixel_scale == None or not self.default_input['wcs']['guess_pixel_scale']:
                scale = [(str("--guess-scale"))]
                scale = []
                pass
    
            else:
                scale = [("--scale-units=" , 'arcsecperpix'),
                        ("--scale-low="    , str(pixel_scale*0.85)),
                        ("--scale-high="   , str(pixel_scale*1.25))]
    
  
            if target_ra != None and target_dec != None:
                try:
    
                    tar_args = [
                        ("--ra="     , str(target_ra)), # target location on sky, used to narrow down cone search
                                ("--dec="    , str(target_dec)),
                                ("--radius=" , str(search_radius)) # radius of search around target for matching sources to index deafult 0.5 deg
                                ]
                    scale = scale + tar_args
                    
                except:
                    pass
    
 
            if 'NAXIS1' in self.header and 'NAXIS2' in self.header:
                NAXIS1 = self.header['NAXIS1']
                NAXIS2 = self.header['NAXIS2']
            else:
                image = get_image(self.fpath)
                NAXIS1 = image.shape[1]
                NAXIS2 = image.shape[0]
    
            # downsample = 0
            include_args = [
    
                ('--no-remove-lines'),
                ("--overwrite"),
                ('--uniformize=' , str(1)),
                ("--downsample=" , str(downsample) ),  # Downsample image - good for large images
                ("--new-fits="   , str(None)), # Don't download new fits file with updated wcs
                ("--cpulimit="   , str(cpulimit)), # set time limit on subprocess
                ("--wcs="        , str(wcs_file)), # filepath of wcs fits header file
                ("--index-xyls=" , str(None)),# don't need all these files
                ("--axy="        , str(None)),
                ("--scamp="      , str(None)),
                ("--corr="       , str(None)), 
                ("--rdl="        , str(None)),
                ("--match="      , str(None)),
                ("--solved="     , str(None)),
                ("--height="     , str(NAXIS2)), #set image height and width
                ("--width="      , str(NAXIS1)),
                # ("--tweak-order="      , str(2)),
                # 
                # ("--parity="      , str(1)),
                # ("--nsigma="    , str(5)),
                # ("--depth="    , str(10)),
                # ("--sigma="    , str(sigma_BKG)),
                ("--no-plots"),
                # ("--no-tweak"),
# 
                ]
            
            
            if self.default_input['wcs']['remove_wcs']:
                
                tar_args = [('--no-verify') # radius of search around target for matching sources to index deafult 0.5 deg
                            ]
                scale = scale + tar_args
            # and scale commands
            include_args = include_args + scale
    
            # build command to run astrometry.net
            args= [str(exe) + ' ' + str(self.fpath) + ' ' ]
            for i in include_args:
                if isinstance(i,tuple):
                    for j in i:
                        args[0] += j
                else:
                    args[0] += i
                args[0] += ' '
                

    
            start = time.time()
            
            astrometry_log_fpath = os.path.join(dirname,'astrometry_'+base + '.txt')
            with open(astrometry_log_fpath, 'w+') as FNULL:
                
                FNULL.write(' '.join(args))
                # print(args)
     
                logger.info('ASTROMETRY started...' )
                pro = subprocess.Popen(args,
                                       shell=True,
            
                                       stdout=FNULL,
                                       stderr=FNULL,
                                       preexec_fn=os.setsid)
                # subprocess.run(args, shell=True,  text=True,stdout=FNULL,
                # stderr=FNULL,)
                # Timeout command - will only run command for this long - ~30s works fine
                try:
                    pro.wait(cpulimit)
                    
                except:
                    print(args)
                    return np.nan
    
    

    
            # check if file is there - if so return filepath if not return nan
            
            if os.path.isfile(wcs_file):
                # print(wcs_file)
                # astrometry creates this none file - delete it
                os.remove(os.path.join(os.getcwd(), 'None'))
                
                astrometry_outputs = ['.corr','.axy','.match',
                                      '.rdls','.solved','-indx.xyls']
                
                import glob
                for i in astrometry_outputs:
                    matching_files  = glob.glob(os.path.join(dirname,'*'+i))
                    for j in matching_files: 
                        os.remove(j)
            
    
               
    
    
            else:
                logger.warning("-> Could not solve WCS - Return NAN <-")
                logger.debug(args)
             
    
                return np.nan
            
      
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore")
                 newWCS = fits.open(wcs_file)[0].header
    
                 self.header.update(newWCS,relax = True)
                
                 fits.writeto(self.fpath,
                              self.image,
                              self.header,
                              overwrite = True,
                              output_verify = 'silentfix+ignore')
                 
                 logger.info('WCS values updated!' )
            
                 os.remove(wcs_file)
            
            logger.info('ASTROMETRY finished: %ss' % round(time.time() - start) )
            
            return self.header
    
        except Exception as e:
            logger.exception(e)
            return np.nan
    
    
            
            
            
