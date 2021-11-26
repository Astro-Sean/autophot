#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def search(image, 
           headinfo, 
           target_coords,  
           catalog_keywords, 
           image_filter,
           radius = 0.5,
           wdir = None,
           catalog = 'apass',
           include_IR_sequence_data = False,
           catalog_custom_fpath = None,
           target_name = None,
           target_ra = None,
           target_dec = None):
    '''
    
    :param image: DESCRIPTION
    :type image: TYPE
    :param headinfo: DESCRIPTION
    :type headinfo: TYPE
    :param target_coords: DESCRIPTION
    :type target_coords: TYPE
    :param catalog_keywords: DESCRIPTION
    :type catalog_keywords: TYPE
    :param image_filter: DESCRIPTION
    :type image_filter: TYPE
    :param radius: DESCRIPTION, defaults to 0.5
    :type radius: TYPE, optional
    :param wdir: DESCRIPTION, defaults to None
    :type wdir: TYPE, optional
    :param catalog: DESCRIPTION, defaults to 'apass'
    :type catalog: TYPE, optional
    :param include_IR_sequence_data: DESCRIPTION, defaults to False
    :type include_IR_sequence_data: TYPE, optional
    :param catalog_custom_fpath: DESCRIPTION, defaults to None
    :type catalog_custom_fpath: TYPE, optional
    :param target_name: DESCRIPTION, defaults to None
    :type target_name: TYPE, optional
    :param target_ra: DESCRIPTION, defaults to None
    :type target_ra: TYPE, optional
    :param target_dec: DESCRIPTION, defaults to None
    :type target_dec: TYPE, optional
    :raises Exception: DESCRIPTION
    :return: DESCRIPTION
    :rtype: TYPE

    '''
    

    import numpy as np
    import os,sys
    import requests
    import pathlib
    import shutil
    import os.path
    import warnings

    from functools import reduce
    import pandas as pd


    from astropy.table import Table
    from astropy.wcs import wcs
    from astroquery.vizier import Vizier
    from astropy.io.votable import parse_single_table
    from astropy.coordinates import Angle
    
    from autophot.packages.functions import pix_dist
    
    import logging
    logger = logging.getLogger(__name__)
    
    if wdir is None:
        # TODO: fix this exception
        raise Exception('WDIR needs to be set so I can find the catalog directory')

    try:

        # Get wxs information
        w1 = wcs.WCS(headinfo)

        # Radius around target
        radius  = float(radius)

        # Target name, if applicable
        target = target_name

        # Get workdirectory location,, create directory if needed
        dirname = os.path.join(wdir,'catalog_queries')
        pathlib.Path(dirname).mkdir(parents = True, exist_ok=True)


        # if target or it's ra/dec - set target name
        if target == None:
             if target_ra != None and target_dec != None:
                 target = 'target_ra_'+str(round(target_ra))+'_dec_'+str(round(target_dec))
                 logger.info('New target name: %s' %target)
             else:
                 #  if not just call target
                 target = 'target'

        # Search limitation with Pan Starrs rlimited to 0.5 deg
        if radius > 0.5 and catalog == 'pan_starrs' :
            logger.warning('Search Limitation with PanStarrs API -> Radius = 0.5 [deg] ')
            radius = 0.5

        # Choosen catalog for input.yml, create directory for catalog if needed
        pathlib.Path(os.path.join(dirname ,catalog)).mkdir(parents = True, exist_ok=True)

        # Folder for target, create directory if needed
        target_dir =   reduce(os.path.join,[dirname,catalog,target.lower()])
        pathlib.Path(target_dir).mkdir(parents = True, exist_ok=True)

        # Filename of fetched catalog for this target
        fname = str(target) + '_r_' + str(radius)

        #  If catalog set to cutsom
        if catalog == 'custom':

            target = target_name
            fname = str(target) + '_RAD_' + str(float(radius))

            if catalog_custom_fpath is None:
                logger.critical('Custon catalog selected but "catalog_custom_fpath" not defined')
                exit()
            else:
                fname  = catalog_custom_fpath

            chosen_catalog = pd.read_csv(fname)
            # return chosen_catalog



        # if catalog is found via it's filename - use this and return chosen_catalog
        if os.path.isfile(os.path.join(target_dir,fname+'.csv')):
            logger.info('Catalog found for %s\nCatalog: %s \nFile: %s' % (target,str(catalog).upper(),fname))
            chosen_catalog = Table.read(os.path.join(target_dir,fname+'.csv'),format = 'csv')
            chosen_catalog = chosen_catalog.to_pandas().fillna(np.nan)
        
        else:
            # If no previously catalog found - look for one
            logger.info('Searching for new catalog [%s] for %s ' % ( catalog, target))

            if catalog in ['gaia']:
                # TODO is this needed?

                import astropy.units as u
                from astroquery.gaia import Gaia
                import warnings
                warnings.filterwarnings('ignore')

                width = u.Quantity(radius, u.deg)
                height = u.Quantity(radius, u.deg)

                chosen_catalog = Gaia.query_object_async(coordinate=target_coords, width=width, height=height)

                chosen_catalog = chosen_catalog.to_pandas()
                chosen_catalog.to_csv(fname+'.csv', sep=',',index = False)


            if catalog in ['apass','2mass','sdss']:

                # No row limit
                Vizier.ROW_LIMIT = -1
                catalog_search = Vizier.query_region(target_coords,
                                                     radius = Angle(radius,'deg'),
                                                     catalog = catalog)

                # Select first catalog from list
                chosen_catalog = catalog_search[0].to_pandas()
                
                # Clean the chosen_catalog from sdss
                # TODO: make sure mode 2 is correct
                if catalog == 'sdss':
                    chosen_catalog = chosen_catalog[chosen_catalog['mode']== 2]
                    
            # some catalogs need specific download path using 'requests'
            if catalog in ['pan_starrs','skymapper']:

                mindet=1

                if catalog == 'pan_starrs':

                    server=('https://archive.stsci.edu/'+'panstarrs/search.php')
                    params = {'RA': target_coords.ra.degree, 'DEC': target_coords.dec.degree,
                              'SR': radius, 'max_records': 10000,
                              'outputformat': 'VOTable',
                              'ndetections': ('>%d' % mindet)}

                if catalog == 'skymapper':

                    server=('http://skymapper.anu.edu.au/sm-cone/public/query?')
                    params = {'RA': target_coords.ra.degree, 'DEC': target_coords.dec.degree,
                              'SR': radius,
                              'RESPONSEFORMAT': 'VOTABLE'}

                with open('temp.xml', "wb") as f:

                    logger.info('Downloading sequence stars from %s'  % catalog )
                    response = requests.get(server,params = params)
                    f.write(response.content)

                # Parse local file into astropy.table object
                chosen_catalog = parse_single_table('temp.xml')

                # Delete temporary file
                os.remove('temp.xml')

                # Convert table to chosen_catalogframe
                chosen_catalog_table = chosen_catalog.to_table(use_names_over_ids=True)
                chosen_catalog = chosen_catalog_table.to_pandas()

                # invalid entries in panstarrs are -999 - change to nans
                if catalog == 'pan_starrs':
                    chosen_catalog = chosen_catalog.replace(-999,np.nan)
                    
          

            # No sources in field - temporary fix - will add "check different catalog"
            if len(chosen_catalog) == 0:
                logging.critical('Catalog: %s : does not cover field' %  catalog)
                sys.exit()
            
            # If you have 
            if include_IR_sequence_data and catalog != '2mass':
                Vizier.ROW_LIMIT = -1
                catalog_search = Vizier.query_region(target_coords,
                                                     radius = Angle(radius,'deg'),
                                                     catalog = '2mass')

                # Select first catalog from list
                chosen_catalog_2mass = catalog_search[0].to_pandas()
                chosen_catalog_2mass.rename(columns={"Jmag": "J",
                                           "e_Jmag": "J_err",
                                           "Hmag": "H",
                                           "e_Hmag": "H_err",
                                           "Kmag": "K",
                                           "e_Kmag": "K_err",
                                           'RAJ2000':catalog_keywords['RA'],
                                           'DEJ2000':catalog_keywords['DEC']},
                                  inplace = True)
                
                chosen_catalog = pd.concat([chosen_catalog,chosen_catalog_2mass])
                

            # Save to csv and move to 'catalog_queries'
            chosen_catalog = chosen_catalog.fillna(np.nan)
            chosen_catalog.to_csv(fname+'.csv',index = False,na_rep = np.nan)

            shutil.move(os.path.join(os.getcwd(), fname+'.csv'),
                        os.path.join(target_dir,  fname+'.csv'))


        # Add in x and y pixel locatins under wcs given by file
        x_pix,y_pix = w1.wcs_world2pix(chosen_catalog[catalog_keywords['RA']], chosen_catalog[catalog_keywords['DEC']],1)

        chosen_catalog.insert(loc = 5, column = 'x_pix', value = x_pix)
        chosen_catalog.insert(loc = 6, column = 'y_pix', value = y_pix)


        logger.info('Catalog length: %d' % len(chosen_catalog))

        warnings.filterwarnings("default")
        
        

    except Exception as e:
        logger.info('Catalog retireval failed:\n->%s\n Returning None' %  e)
        chosen_catalog = None

    return chosen_catalog





def match(image,
          headinfo, 
          target_coords, 
          catalog_keywords,
          image_filter,
          chosen_catalog,
          fwhm,
          local_radius = 1000,
          target_x_pix = None,
          target_y_pix = None,
          default_dmag = None,
          mask_sources = False,
          mask_sources_XY_R = None,
           use_moffat = True,
           use_local_stars = False,
           default_moff_beta = 4.765,
          vary_moff_beta = False,
          bkg_level = 3,
          scale = 25,
          sat_lvl = 65536,
          max_fit_fwhm = 30,
          max_catalog_sources = 300,
          fitting_method = 'least_squares',
          matching_source_FWHM = False,
          matching_source_FWHM_limit = 3,
          catalog_matching_limit = 25,
          include_IR_sequence_data = False,
          remove_boundary_sources = True,
          pix_bound = 25,
          plot_catalog_nondetections = False):
    
    
    
    print('\nMatching Catalog sources to image\n')
    
    import warnings
    import numpy as np

    import pandas as pd
    import lmfit
    from photutils import DAOStarFinder
    from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma

    from autophot.packages.functions import moffat_2d,moffat_fwhm
    from astropy.stats import sigma_clipped_stats


    from autophot.packages.functions import pix_dist


    import logging

    # with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logger = logging.getLogger(__name__)
    
    RA_list = []
    DEC_list = []
    x_new_source = []
    y_new_source = []
    x_new_cen    = []
    y_new_cen    = []
    cp_dist      = []
    dist2target_list = []
    cat_idx      = []
    
    # list for (non) detected sources
    non_detections = []
    detections     = []
    fwhm_list = []
    
    # Add all available filter information, not just the image filter
    image_filtermagnitude = {}
    image_filtermagnitude_err = {}
    
    if default_dmag is None:
        # TODO: Set default DMAGSpass
        pass

    for key,val in default_dmag.items():
        
        if key != image_filter:
        
            image_filtermagnitude[key] = []
            image_filtermagnitude_err[key+'_err'] = []
        
    image_filtermagnitude[image_filter]= []
    image_filtermagnitude_err[image_filter+'_err'] = []
    
    if include_IR_sequence_data:
        for i in ['J','H','K']:
            catalog_keywords[i] = i
            catalog_keywords[i+'_err'] = i+'_err'

    # Remove values that don't have matching value in selected value
    chosen_catalog = chosen_catalog[~np.isnan(chosen_catalog[catalog_keywords[image_filter]])]

    dist_to_target = pix_dist(target_x_pix,chosen_catalog['x_pix'],target_y_pix,chosen_catalog['y_pix'])
    
    too_close_to_target = dist_to_target < 3*fwhm
    
    chosen_catalog = chosen_catalog[~too_close_to_target]
    
    if np.sum(too_close_to_target)!=0:
        print('Removed %d sources too close to target' % np.sum(too_close_to_target))
        
    # Remove faint sources
    len_before = len(chosen_catalog)
    chosen_catalog = chosen_catalog[chosen_catalog[catalog_keywords[image_filter]] < catalog_matching_limit]
    logger.info('Removed %d sources fainter than cutoff [%d mag]' % (len_before-len(chosen_catalog),catalog_matching_limit))

    

    
    if remove_boundary_sources:

        len_before = len(chosen_catalog)

        chosen_catalog = chosen_catalog[chosen_catalog['x_pix'] < image.shape[1] - pix_bound ]
        chosen_catalog = chosen_catalog[chosen_catalog['x_pix'] > pix_bound ]
        chosen_catalog = chosen_catalog[chosen_catalog['y_pix'] < image.shape[0] - pix_bound ]
        chosen_catalog = chosen_catalog[chosen_catalog['y_pix'] > pix_bound ]
        
        print('Removed %d sources too close to boundary or off image' % (len_before-len(chosen_catalog)))


    # Sort by the brighest sources first
    chosen_catalog.sort_values(by=[catalog_keywords[image_filter]],
                            ascending=True,
                            inplace = True,
                            na_position='last')



    # Grid of close-up mathcing scale of image
    x = np.arange(0,2*scale)
    xx,yy= np.meshgrid(x,x)

    k = 0

    dx = 2*fwhm
    dy = 2*fwhm


    off_image_idx = (chosen_catalog.x_pix<=0) | (chosen_catalog.x_pix>=image.shape[1]) | (chosen_catalog.y_pix<=0)  | (chosen_catalog.y_pix>=image.shape[1])
    print('%d off image catalog sources removed' % np.sum(off_image_idx))

    chosen_catalog = chosen_catalog[~off_image_idx]

    if use_local_stars:

        too_far_dist = pix_dist(chosen_catalog.x_pix,target_x_pix,chosen_catalog.y_pix,target_y_pix)
        too_far_idx = too_far_dist > local_radius

        print('%d sources outisde selected radius removed' % np.sum(too_far_idx))

        chosen_catalog = chosen_catalog[~too_far_idx]


    if mask_sources:

        remove_idx = []

        for X_mask,Y_mask,R_mask in mask_sources_XY_R:


            dist_within_mask = pix_dist(chosen_catalog.x_pix,X_mask,chosen_catalog.y_pix,Y_mask)

            inside_mask = dist_within_mask < R_mask

            remove_idx.append(inside_mask)


        sources_within_mask = abs(np.array(sum(remove_idx))-1)


        s = pd.Series(sources_within_mask.astype(bool), name='bools')
        
        chosen_catalog = chosen_catalog[s.values]

        logger.info('Removed %d sources within masked regions' % (np.sum(sources_within_mask)))


    if use_moffat:

        logging.info('Using Moffat Profile for fitting')

        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:

        logging.info('Using Gaussian Profile for fitting')

        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm

    useable_sources = 0
    # offimage_source = 0
    not_detected = 0
    saturated_source = 0
    broken_cutout = 0
    broken = 0
    too_far = 0
    
    logging.info('Catalog Length: %d' % len(chosen_catalog))
    
    if plot_catalog_nondetections:
        logging.info('\nIncluding non detections in catalog analysis')
        
        
    
    
    try:

        for i in range(len(chosen_catalog.index.values)):


            if useable_sources >= max_catalog_sources:
                break

            idx = np.array(chosen_catalog.index.values)[i]

            message = '\rMatching catalog to image: %d / %d :: Useful sources %d / %d '% (float(i)+1,
                                                                                        len(chosen_catalog.index),
                                                                                        useable_sources+1,
                                                                                       len(chosen_catalog.index),
)
            print(message,end = '')

            # catalog pixel coordinates of source take as an approximate location
            x = chosen_catalog.x_pix[idx]
            y = chosen_catalog.y_pix[idx]
            
            RA_list.append(chosen_catalog[catalog_keywords['RA']][idx])
            DEC_list.append(chosen_catalog[catalog_keywords['DEC']][idx])

            x_new_source.append(x)
            y_new_source.append(y)

            image_filtermagnitude[image_filter].append(chosen_catalog[catalog_keywords[image_filter]][idx])
            image_filtermagnitude_err[image_filter+'_err'].append(chosen_catalog[catalog_keywords[image_filter+'_err']][idx])

            for key,val in default_dmag.items():
                
                if key != image_filter:
                    try:
                        # print(chosen_catalog[catalog_keywords[key]][idx])
                        image_filtermagnitude[key].append(chosen_catalog[catalog_keywords[key]][idx])
                        image_filtermagnitude_err[key+'_err'].append(chosen_catalog[catalog_keywords[key+'_err']][idx])
                        
                        # print(image_filtermagnitude[key])
                    except:
                        
                        image_filtermagnitude[key].append(np.nan)
                        image_filtermagnitude_err[key+'_err'].append(np.nan)
                        pass

            # Add index key for original catalog file comparision and matching
            cat_idx.append(int(idx))

            try:


                 # Create cutout image of size (2*scale,2*scale)
                 close_up = image[int(y-scale): int(y + scale),
                                  int(x-scale): int(x + scale)]


                 # Cutout not possible - too close to edge or invalue pixel chosen_catalog i.e. nans of infs
                 if close_up.shape != (2*scale,2*scale):

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     broken_cutout +=1

                     continue

                 # Preset pixel error popup skip this source
                 if np.nanmax(close_up) >= sat_lvl or np.isnan(np.min(close_up)):

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     saturated_source +=1

                     continue

                 # Get close up image properties
                 mean, median, std = sigma_clipped_stats(close_up,
                                                         sigma = bkg_level,
                                                         maxiters = 10)

                 try:
             

                    daofind = DAOStarFinder(fwhm      = fwhm,
                                            threshold = bkg_level*std,
                                            sharplo   =  0.2,sharphi = 1.0,
                                            roundlo   = -1.0,roundhi = 1.0
                                            )
    
                    sources = daofind(close_up - median)
    
                    # If no source is found - skip
                    if sources == None:
                        sources = []
    
                    else:
                        sources = sources.to_pandas()



                 except Exception as e:


                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     logger.exception(e)

                     broken+=1

                     continue

                 if len(sources) == 0 and not plot_catalog_nondetections:

                     not_detected+=1

                     non_detections.append(chosen_catalog[catalog_keywords[image_filter]].loc[[idx]].values[0])

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     continue
                 
                 if len(sources) == 0:
                     
                     xc_guess = close_up.shape[1]/2
                     yc_guess = close_up.shape[0]/2
                     
                 

                 # If more than one source detected in close up
                 # assume source closest to center is desired source
                 elif len(sources) > 1:


                   r_vals = pix_dist(scale,np.array(sources['xcentroid']),
                                     scale,np.array(sources['ycentroid']))

                   r_idx = np.argmin(r_vals)
                   
                   xc_guess = np.array(sources['xcentroid'])[r_idx]
                   yc_guess = np.array(sources['ycentroid'])[r_idx]
                   
                 else:

                     # Approximate location of source
                     xc_guess = np.array(sources['xcentroid'])[0]
                     yc_guess = np.array(sources['ycentroid'])[0]

                 

                 pars = lmfit.Parameters()
                 pars.add('A',value = np.nanmax(close_up),
                         min = 1e-6)

                 pars.add('x0',value = close_up.shape[1]/2,
                         min = close_up.shape[1]/2 - dx,
                         max = close_up.shape[1]/2 + dx)

                 pars.add('y0',value = close_up.shape[0]/2,
                         min = close_up.shape[0]/2 - dy,
                         max = close_up.shape[0]/2 + dy)

                 pars.add('sky',value = np.nanmedian(close_up))

                 if use_moffat:

                   pars.add('alpha',
                            value = 3,
                            min = 0,
                            max = 30)
                   pars.add('beta',
                            value = default_moff_beta,
                            min = 0,
                            vary = vary_moff_beta)

                 else:

                  pars.add('sigma',
                            value = 3,
                           min = 0,
                           max = gauss_fwhm2sigma(max_fit_fwhm) )

                 if use_moffat:

                     def residual(p):
                         p = p.valuesdict()
                         return (close_up - moffat_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(alpha=p['alpha'],beta=p['beta'])).reshape(close_up.shape)).flatten()
                 else:

                     def residual(p):
                         p = p.valuesdict()
                         return (close_up - gauss_2d((xx,yy),p['x0'],p['y0'],p['sky'],p['A'],dict(sigma=p['sigma'])).reshape(close_up.shape)).flatten()


                 mini = lmfit.Minimizer(residual,
                                       pars,
                                       nan_policy = 'omit')
                 result = mini.minimize(method = fitting_method)


                 xcen = result.params['x0'].value
                 ycen = result.params['y0'].value

                 S    = result.params['sky'].value
                 H    = result.params['A'].value

                 if use_moffat:
                     source_image_params = dict(alpha=result.params['alpha'].value,beta=result.params['beta'].value)
                 else:
                     source_image_params = dict(sigma=result.params['sigma'])


                 fwhm_fit = fitting_model_fwhm(source_image_params)


                 k+=1

                # Add new source location accounting for difference
                # in fitted location / expected location
                 centroid_x = xcen - scale + x
                 centroid_y = ycen - scale + y


                 dist2target = pix_dist(target_x_pix,centroid_x,
                                       target_y_pix,centroid_y)

                 detections.append(chosen_catalog[catalog_keywords[image_filter]].loc[[idx]].values[0])

                 useable_sources +=1

                 x_new_cen.append(centroid_x)
                 y_new_cen.append(centroid_y)

                 cp_dist.append(np.sqrt( (xcen - scale)**2 + (ycen - scale)**2) )

                 dist2target_list.append(dist2target)

                 fwhm_list.append(fwhm_fit)



            except Exception as e:

                x_new_cen.append(np.nan)
                y_new_cen.append(np.nan)
                cp_dist.append(np.nan)
                dist2target_list.append(np.nan)
                fwhm_list.append(np.nan)

                logger.exception(e)

                continue

        print('  .. done')

        print('\nBroken cutouts: %d' % broken_cutout )
        print('Not in correct location: %d' % too_far)
        print('Not detected: %d' % not_detected)
        print('Saturated: %d' % saturated_source)
        print('Error: %d\n' % broken )




        # =============================================================================
        # Prepare output
        # =============================================================================

        frame_chosen_catalog = [
                      np.array([int(i) for i in cat_idx]),
                      np.array(RA_list),
                      np.array(DEC_list),
                      np.array(x_new_source),
                      np.array(y_new_source),
                      np.array(x_new_cen),
                      np.array(y_new_cen),
                      np.array(cp_dist),
                      np.array(dist2target_list),
                      np.array(fwhm_list),
                      np.array(image_filtermagnitude[image_filter]),
                      np.array(image_filtermagnitude_err[image_filter+'_err'])
                      

                      ]
        
        frame_cols = [
                    'cat_idx',
                     'ra',
                     'dec',
                     'x_pix_cat',
                     'y_pix_cat',
                     'x_pix',
                     'y_pix',
                     'cp_dist',
                     'dist2target',
                     'fwhm',
                     'cat_'+image_filter,
                     'cat_'+image_filter+'_err'


                      ]
        
        chosen_catalog_new_frame = pd.DataFrame(frame_chosen_catalog).T
        chosen_catalog_new_frame.columns = frame_cols
        
        for f in image_filtermagnitude:
            if  (np.isnan(image_filtermagnitude[f]).all() ) | (f == image_filter) :
                    continue
   
            # print(image_filtermagnitude[f])
            chosen_catalog_new_frame['cat_'+f] = image_filtermagnitude[f]
            chosen_catalog_new_frame['cat_'+f+'_err'] = image_filtermagnitude_err[f+'_err']
                
        
        chosen_catalog_new_frame = chosen_catalog_new_frame.dropna(subset=['x_pix', 'y_pix'])
        chosen_catalog_new_frame = chosen_catalog_new_frame.dropna(how='all') 
        
        if matching_source_FWHM:

            extended_mask = abs(chosen_catalog_new_frame['fwhm'] - fwhm) < matching_source_FWHM_limit

            print('\nCatalog sources removed: %d/%d' % (sum(extended_mask),len(chosen_catalog_new_frame)))

            chosen_catalog_new_frame = chosen_catalog_new_frame[extended_mask]


        warnings.filterwarnings("default")

    except Exception as e:
        logger.exception(e)


    return chosen_catalog_new_frame