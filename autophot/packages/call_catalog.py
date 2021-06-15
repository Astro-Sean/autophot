#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:40:29 2018

@author: seanbrennan
"""

def search(image, headinfo, target_coords, syntax, catalog_syntax, filter_):

    import warnings

    if not syntax['catalog_warnings'] or syntax['master_warnings']:
            warnings.filterwarnings("ignore")

    import numpy as np
    import os,sys
    import requests
    import pathlib
    import shutil
    import os.path

    from functools import reduce
    import pandas as pd

    # from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma

    # from autophot.packages.functions import moffat_2d,moffat_fwhm

    from astropy.table import Table
    from astropy.wcs import wcs
    from astroquery.vizier import Vizier
    from astropy.io.votable import parse_single_table
    from astropy.coordinates import Angle

    # from autophot.packages.functions import pix_dist
    import logging
    logger = logging.getLogger(__name__)

    try:

        # Get wxs information
        w1 = wcs.WCS(headinfo)

        # Radius around target
        radius  = float(syntax['radius'])

        # Target name, if applicable
        target = syntax['target_name']

        # Get workdirectory location,, create directory if needed
        dirname = os.path.join(syntax['wdir'],'catalog_queries')
        pathlib.Path(dirname).mkdir(parents = True, exist_ok=True)


        # if target or it's ra/dec - set target name
        if target == None:
             if syntax['target_ra'] != None and syntax['target_dec'] != None:
                 target = 'target_ra_'+str(round(syntax['target_ra']))+'_dec_'+str(round(syntax['target_dec']))
                 logger.info('New target name: %s' %target)
             else:
                 #  if not just call target
                 target = 'target'

        # Search limitation with Pan Starrs rlimited to 0.5 deg
        if radius > 0.5 and syntax['catalog'] == 'pan_starrs' :
                    logger.warning('Search Limitation with PanStarrs API -> Radius = 0.5 [deg] ')
                    radius = 0.5

        # Choosen catalog for input.yml, create directory for catalog if needed
        catalog_dir = syntax['catalog']
        pathlib.Path(os.path.join(dirname ,catalog_dir)).mkdir(parents = True, exist_ok=True)

        # Folder for target, create directory if needed
        target_dir =   reduce(os.path.join,[dirname,catalog_dir,target.lower()])
        pathlib.Path(target_dir).mkdir(parents = True, exist_ok=True)

        # Filename of fetchec catalog
        fname = str(target) + '_r_' + str(radius)

        # Can force to use certain catalog - untested 03-10-19
        if syntax['force_catalog_csv']:

            logger.info('Using '+syntax['force_catalog_csv_name']+' as catalog')
            fname = str(syntax['force_catalog_csv_name']) + '_r_' + str(radius)


         #  If catalog set to cutsom
        if syntax['catalog'] == 'custom':

            target = syntax['target_name']
            fname = str(target) + '_RAD_' + str(float(syntax['radius']))

            if not syntax['catalog_custom_fpath']:
                logger.critical('Custoim catalog selected but "catalog_custom_fpath" not defined')
                exit()
            else:
                fname = syntax['catalog_custom_fpath']

            data = pd.read_csv(fname)



        # if catalog is found via it's filename - use this and return data
        if os.path.isfile(os.path.join(target_dir,fname+'.csv')):
            logger.info('Catalog found for Target: %s\nCatalog: %s \nFile: %s' % (target,str(catalog_dir).upper(),fname))
            data = Table.read(os.path.join(target_dir,fname+'.csv'),format = 'csv')
            data = data.to_pandas()

        else:
            # If no previously catalog found - look for one
            logger.info('Searching for new catalog: %s ' % syntax['catalog'])

            if syntax['catalog'] in ['gaia']:

                import astropy.units as u
                from astroquery.gaia import Gaia
                import warnings
                warnings.filterwarnings('ignore')

                width = u.Quantity(radius, u.deg)
                height = u.Quantity(radius, u.deg)

                data = Gaia.query_object_async(coordinate=target_coords, width=width, height=height)

                data = data.to_pandas()
                data.to_csv(fname+'.csv', sep=',',index = False)

                # Move file to new location - 'catalog queries'
                shutil.move(os.path.join(os.getcwd(), fname+'.csv'),
                            os.path.join(target_dir, fname+'.csv'))

                warnings.filterwarnings('default')

            if syntax['catalog'] in ['apass','2mass','sdss']:

                # No row limit
                Vizier.ROW_LIMIT = -1
                catalog_search = Vizier.query_region(target_coords,
                                                     radius = Angle(radius,'deg'),
                                                     catalog = syntax['catalog'])

                # Select first catalog from list
                data = catalog_search[0].to_pandas()
                data.to_csv(fname+'.csv', sep=',',index = False)


                # Move file to new location - 'catalog queries'
                shutil.move(os.path.join(os.getcwd(), fname+'.csv'),
                            os.path.join(target_dir, fname+'.csv'))


            # some catalogs need specific download path using 'requests'
            if syntax['catalog'] in ['pan_starrs','skymapper']:

                mindet=1

                if syntax['catalog'] == 'pan_starrs':

                    server=('https://archive.stsci.edu/'+'panstarrs/search.php')
                    params = {'RA': target_coords.ra.degree, 'DEC': target_coords.dec.degree,
                              'SR': radius, 'max_records': 10000,
                              'outputformat': 'VOTable',
                              'ndetections': ('>%d' % mindet)}

                if syntax['catalog'] == 'skymapper':

                    server=('http://skymapper.anu.edu.au/sm-cone/public/query?')
                    params = {'RA': target_coords.ra.degree, 'DEC': target_coords.dec.degree,
                              'SR': radius,
                              'RESPONSEFORMAT': 'VOTABLE'}

                with open('temp.xml', "wb") as f:

                    logger.info('Downloading from %s'  % syntax['catalog'] )
                    response = requests.get(server,params = params)
                    f.write(response.content)

                # Parse local file into astropy.table object
                data = parse_single_table('temp.xml')

                # Delete temporary file
                os.remove('temp.xml')

                # Convert table to dataframe
                data_table = data.to_table(use_names_over_ids=True)
                data = data_table.to_pandas()

                # invalid entries in panstarrs are -999 - change to nans
                if syntax['catalog'] == 'pan_starrs':
                    data = data.replace(-999,np.nan)

                # No sources in field - temporary fix - will add "check different catalog"
                if len(data) == 0:
                    logging.critical('Catalog: %s : does not cover field' %  syntax['catalog'])
                    sys.exit()

                # Save to csv and move to 'catalog_queries'
                data.to_csv(fname+'.csv',index = False)

                shutil.move(os.path.join(os.getcwd(), fname+'.csv'),
                            os.path.join(target_dir,  fname+'.csv'))


        # Add in x and y pixel locatins under wcs
        x_pix,y_pix = w1.wcs_world2pix(data[catalog_syntax['RA']], data[catalog_syntax['DEC']],1)

        data.insert(loc = 5, column = 'x_pix', value = x_pix)
        data.insert(loc = 6, column = 'y_pix', value = y_pix)

        # Remove boundary sources
        data = data[data.x_pix < image.shape[1] - syntax['pix_bound']]
        data = data[data.x_pix > syntax['pix_bound']]
        data = data[data.y_pix < image.shape[0] - syntax['pix_bound']]
        data = data[data.y_pix > syntax['pix_bound']]

        logger.info('Catalog length: %d' % len(data))

        warnings.filterwarnings("default")

    except Exception as e:
        logger.exception(e)
        data = None

    return data


import sys
def update_pct(w_str):
    w_str = str(w_str)
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(" " * len(w_str))
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(w_str)
    sys.stdout.flush()

def match(image, headinfo, target_coords, syntax, catalog_syntax, filter_,data, fwhm):

    print('\nCatalog matching\n')
    import warnings

    if not syntax['catalog_warnings'] or syntax['master_warnings']:
        warnings.filterwarnings("ignore")

    import numpy as np
    import matplotlib.pyplot as plt

    import pandas as pd
    import lmfit
    from autophot.packages.functions import gauss_sigma2fwhm,gauss_2d,gauss_fwhm2sigma

    from autophot.packages.functions import moffat_2d,moffat_fwhm
    from astropy.stats import sigma_clipped_stats


    from autophot.packages.functions import pix_dist

    import logging

    logger = logging.getLogger(__name__)

    x_new_source = []
    y_new_source = []
    x_new_cen    = []
    y_new_cen    = []
    cp_dist      = []
    dist2target_list = []
    cat_idx      = []
    non_detections = []
    detections     = []
    fwhm_list = []
    filter_magnitude = []
    filter_magnitude_err = []

    # Remove values that don't have matching value in selected value
    data_update = data[~np.isnan(data[catalog_syntax[filter_]])]
    len_before = len(data_update)
    data_update = data_update[data_update[catalog_syntax[filter_]] < syntax['catalog_matching_limit']]
    print('Remove %d sources fainter that cutoff [%d mag]' % (len_before-len(data_update),syntax['catalog_matching_limit']))


    if syntax['remove_boundary_sources']:

        # with_boundary = len(sources)

        data_update = data_update[data_update['x_pix'] < image.shape[1] - syntax['pix_bound'] ]
        data_update = data_update[data_update['x_pix'] > syntax['pix_bound'] ]
        data_update = data_update[data_update['y_pix'] < image.shape[0] - syntax['pix_bound'] ]
        data_update = data_update[data_update['y_pix'] > syntax['pix_bound'] ]

    # Look at most accuarte measurements first based on errors

    data_update.sort_values(by=[catalog_syntax[filter_]],
                            ascending=True,
                            inplace = True,
                            na_position='last')




    if syntax['use_daofind']:

        print('Using DAOFIND for catalog matching')

        from photutils import DAOStarFinder

    elif syntax['use_imageseg']:

        print('Using Image Segmentation for scatalog matching')

        from photutils import detect_threshold
        from astropy.convolution import Gaussian2DKernel
        from astropy.stats import gaussian_fwhm_to_sigma
        from photutils import detect_sources
        from photutils import deblend_sources
        from photutils import SourceCatalog

    else:

        print('No search method selected - Using DAOFIND for catalog matching')

        from photutils import DAOStarFinder
        syntax['use_daofind'] = True


    # Grid of close-up mathcing scale of image
    x = np.arange(0,2*syntax['scale'])
    xx,yy= np.meshgrid(x,x)

    k = 0

    # Wiggle room for catalog matching
    # TODO: check if thiss is still needed
    # dx = syntax['catalog_matching_dx']
    # dy = syntax['catalog_matching_dy']

    dx = 2.*syntax['fwhm']
    dy = 2.5*syntax['fwhm']



    # non_local_stars = 0
    # inside_mask_star = 0

    off_image_idx = (data_update.x_pix<=0) | (data_update.x_pix>=image.shape[1]) | (data_update.y_pix<=0)  | (data_update.y_pix>=image.shape[1])
    print('%d off image catalog sources removed' % np.sum(off_image_idx))

    data_update = data_update[~off_image_idx]


    if syntax['use_local_stars']:

        too_far_dist = pix_dist(data_update.x_pix,syntax['target_x_pix'],data_update.y_pix,syntax['target_y_pix'])
        # print(too_far_dist)
        too_far_idx = too_far_dist > syntax['local_radius']

        print('%d sources outisde selected radius removed' % np.sum(too_far_idx))

        data_update = data_update[~too_far_idx]


    if syntax['mask_sources']:

        remove_idx = []

        for X_mask,Y_mask,R_mask in syntax['mask_sources_XY_R']:


            dist_within_mask = pix_dist(data_update.x_pix,X_mask,data_update.y_pix,Y_mask)

            inside_mask = dist_within_mask < R_mask
            # inside_mask = inside_mask

            remove_idx.append(inside_mask)

        # print(remove_idx)
        sources_within_mask = abs(np.array(sum(remove_idx))-1)
        # print(sources_within_mask)

        s = pd.Series(sources_within_mask.astype(bool), name='bools')
        # print(s)
        # sources_within_mask[sources_within_mask>=1] = 1

        # sources = sources[~sources_within_mask]


        data_update = data_update[s.values]

        logger.info('Removed %d sources within masked regions' % (np.sum(sources_within_mask)))


    if syntax['use_moffat']:

        logging.info('Using Moffat Profile for fitting')

        fitting_model = moffat_2d
        fitting_model_fwhm = moffat_fwhm

    else:

        logging.info('Using Gaussian Profile for fitting')

        fitting_model = gauss_2d
        fitting_model_fwhm = gauss_sigma2fwhm



    try:
        useable_sources = 0
        # offimage_source = 0
        not_detected = 0
        saturated_source = 0
        broken_cutout = 0
        broken = 0
        too_far = 0
        for i in range(len(data_update.index.values)):

            if useable_sources >= syntax['max_catalog_sources']:
                break

            idx = np.array(data_update.index.values)[i]

            message = '\rMatching catalog to image: %d / %d :: Useful sources %d / %d '% (float(i),
                                                                                        len(data_update.index),
                                                                                        useable_sources,
                                                                                        len(data_update.index),
                                                                                        )
            print(message,end = '')

            # catalog pixel coordinates of source take as an approximate location
            x = data_update.x_pix[idx]
            y = data_update.y_pix[idx]

            x_new_source.append(x)
            y_new_source.append(y)

            filter_magnitude.append(data_update[catalog_syntax[filter_]][idx])
            filter_magnitude_err.append(data_update[catalog_syntax[filter_+'_err']][idx])

            # Add index key for original catalog file comparision and matching
            cat_idx.append(int(idx))

            try:


                 # Create cutout image of size (2*syntax['scale'],2*syntax['scale'])
                 close_up = image[int(y-syntax['scale']): int(y + syntax['scale']),
                                  int(x-syntax['scale']): int(x + syntax['scale'])]


                 # Cutout not possible - too close to edge or invalue pixel data i.e. nans of infs
                 if close_up.shape != (2*syntax['scale'],2*syntax['scale']):

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     broken_cutout +=1

                     continue

                 # Preset pixel error popup skip this source
                 if np.nanmax(close_up) >= syntax['sat_lvl']  or np.isnan(np.min(close_up)):

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     saturated_source +=1

                     continue

                 # Get close up image properties
                 mean, median, std = sigma_clipped_stats(close_up,
                                                         sigma = syntax['bkg_level'],
                                                         maxiters = syntax['iters'])

                 try:
                     if syntax['use_daofind']:

                        daofind = DAOStarFinder(fwhm      = fwhm,
                                                threshold = syntax['bkg_level']*std,
                                                sharplo   =  0.2,sharphi = 1.0,
                                                roundlo   = -1.0,roundhi = 1.0
                                                )

                        sources = daofind(close_up - median)

                        # If no source is found - skip
                        if sources == None:
                            sources = []

                        else:
                            sources = sources.to_pandas()

                     elif syntax['use_imageseg']:

                        threshold = detect_threshold(close_up, nsigma=syntax['bkg_level'])

                        sigma = fwhm * gaussian_fwhm_to_sigma

                        kernel = Gaussian2DKernel(sigma,
                                             # x_size=close_up.shape[0],
                                             # y_size=close_up.shape[1]
                                             )
                        # kernel.normalize()

                        segm = detect_sources(close_up,
                                              threshold,
                                              npixels=5,
                                              filter_kernel=kernel
                                              )

                        segm_deblend = deblend_sources(close_up,
                                                       segm,
                                                       npixels=5,
                                                       filter_kernel=kernel
                                                       )

                        props = SourceCatalog(close_up, segm_deblend )

                        sources = props.to_table().to_pandas()

                 except Exception as e:


                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     logger.exception(e)

                     broken+=1

                     continue

                 if len(sources) == 0 :

                     not_detected+=1

                     non_detections.append(data_update[catalog_syntax[filter_]].loc[[idx]].values[0])

                     x_new_cen.append(np.nan)
                     y_new_cen.append(np.nan)
                     cp_dist.append(np.nan)
                     dist2target_list.append(np.nan)
                     fwhm_list.append(np.nan)

                     continue


                 # Approximate location of source
                 xc_guess = np.array(sources['xcentroid'])[0]
                 yc_guess = np.array(sources['ycentroid'])[0]

                 # If more than one source detected in close up
                 # assume source closest to center is desired source
                 if len(sources) > 1:

                    # if syntax['match_catalog_locate_dist']:

                   r_vals = pix_dist(syntax['scale'],np.array(sources['xcentroid']),
                                     syntax['scale'],np.array(sources['ycentroid']))

                   r_idx = np.argmin(r_vals)

                   # if closest source is too far away from predicted loction - ignore
                   if r_vals[r_idx] > syntax['match_dist']:

                      too_far +=1

                      x_new_cen.append(np.nan)
                      y_new_cen.append(np.nan)
                      cp_dist.append(np.nan)
                      dist2target_list.append(np.nan)
                      fwhm_list.append(np.nan)

                      continue
                    # else:

                    #     # Look for the brightest sources in the field
                    #    # r_idx = np.argmin(sources.mag)
                    #    pass

                   xc_guess = np.array(sources['xcentroid'])[r_idx]
                   yc_guess = np.array(sources['ycentroid'])[r_idx]


                 try:

                     pars = lmfit.Parameters()
                     pars.add('A',value = np.nanmax(close_up),min = 0)

                     pars.add('x0',value = close_up.shape[1]/2,
                              min = close_up.shape[1]/2 - dx,
                              max = close_up.shape[1]/2 + dx)

                     pars.add('y0',value = close_up.shape[0]/2,
                              min = close_up.shape[0]/2 - dy,
                              max = close_up.shape[0]/2 + dy)

                     pars.add('sky',value = np.nanmedian(close_up))

                     if syntax['use_moffat']:

                        pars.add('alpha',
                                 value = 3,
                                 min = 0,
                                 max = 30)
                        pars.add('beta',
                                 value = syntax['default_moff_beta'],
                                min = 0,
                                vary = syntax['vary_moff_beta'])

                     else:

                       pars.add('sigma',
                                 value = 3,
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


                     mini = lmfit.Minimizer(residual,
                                            pars,
                                            nan_policy = 'omit')
                     result = mini.minimize(method = 'least_squares')


                     xcen = result.params['x0'].value
                     ycen = result.params['y0'].value

                     S    = result.params['sky'].value
                     H    = result.params['A'].value

                     if syntax['use_moffat']:
                        source_image_params = dict(alpha=result.params['alpha'].value,beta=result.params['beta'].value)
                     else:
                        source_image_params = dict(sigma=result.params['sigma'])


                     fwhm_fit = fitting_model_fwhm(source_image_params)


                     k+=1

                     # Add new source location accounting for difference
                     # in fitted location / expected location
                     centroid_x = xcen - syntax['scale'] + x
                     centroid_y = ycen - syntax['scale'] + y


                     dist2target = pix_dist(syntax['target_x_pix'],centroid_x,
                                            syntax['target_y_pix'],centroid_y)

                     # if dist2target <=3 * syntax['fwhm']:
                     #     print('Catalog Matching pixked up source near target, ignoring')
                     #     x_new_cen.append(np.nan)
                     #     y_new_cen.append(np.nan)
                     #     cp_dist.append(np.nan)
                     #     dist2target_list.append(np.nan)
                     #     fwhm_list.append(np.nan)
                     #     continue


                     detections.append(data_update[catalog_syntax[filter_]].loc[[idx]].values[0])

                     useable_sources +=1

                     x_new_cen.append(centroid_x)
                     y_new_cen.append(centroid_y)

                     cp_dist.append(np.sqrt( (xcen - syntax['scale'])**2 + (ycen - syntax['scale'])**2) )

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


                 if syntax['source_plot']:

                    if len(sources) == 1:

                         fig = plt.figure(figsize = (6,6))
                         ax = fig.add_subplot(111)
                         ax.imshow(close_up)

                         ax.set_title('Source @ x = '+'{0:.3f}'.format(xcen +  x - syntax['scale'])+' : y = '
                                     +'{0:.3f}'.format(ycen +  y - syntax['scale']))

                         small_ap = plt.Circle((xcen,ycen), syntax['ap_size']*headinfo['FWHM'], color='r',fill = False,label = 'Aperture')
                         big_ap = plt.Circle((xcen,ycen), syntax['inf_ap_size']*headinfo['FWHM'], color='b',fill = False,label = 'Aperture Correction')

                         ax.add_artist(small_ap)
                         ax.add_artist(big_ap)

                         ax.plot([],[],' ', label = 'Sky ='+'{0:.3f}'.format(S)+'Height ='+'{0:.3f}'.format(H))
                         ax.scatter(syntax['scale'],syntax['scale'],marker = '+',s = 100 , color = 'r',linewidths=0.01,label = 'Catalog')
                         ax.scatter(xc_guess,yc_guess,marker = '+',s = 100 , color = 'b',linewidths=0.01,label= 'Source detection [closest object to catalog]' )
                         ax.scatter(xcen,ycen,marker = '+',s = 100 , color = 'green',linewidths=0.01,label= 'Least square fit' )

                         ax.legend(loc = 'upper right')


                         import os

                         save_loc = os.path.join(syntax['write_dir'],'matched_sources')

                         os.makedirs(save_loc, exist_ok=True)

                         fig.savefig(os.path.join(save_loc,'catalog_match_%d.png' % i),
                               bbox_inches='tight')

                         plt.close(fig)


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
        # fit this
        if syntax['show_nondetect_plot']:

            non_detections = np.array(non_detections)[np.isfinite(non_detections)]
            detections = np.array(detections)[np.isfinite(detections)]


            if len(non_detections) == 0:
                logger.debug('All sources detected')

            fig = plt.figure(figsize=(6,8))
            ax = fig.add_subplot(111)
            ax.hist(non_detections,bins = 'auto',
                         align = 'mid',
                         color = 'green',
                         histtype = 'step',
                         label = 'Non-Detection')

            ax.hist(detections,
                         bins = 'auto',
                         align = 'mid',
                         color = 'red',
                         histtype = 'step',
                         label = 'Detection')

            ax.set_title('Non - Detections')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('Binned Occurance')
            ax.legend(loc = 'best')



# =============================================================================
#
# =============================================================================

        frame_data = [
                      np.array([int(i) for i in cat_idx]),
                      np.array(data_update[catalog_syntax['RA']]),
                      np.array(data_update[catalog_syntax['DEC']]),
                      np.array(x_new_source),
                      np.array(y_new_source),
                      np.array(x_new_cen),
                      np.array(y_new_cen),
                      np.array(cp_dist),
                      np.array(dist2target_list),
                      np.array(fwhm_list),
                      np.array(filter_magnitude),
                      np.array(filter_magnitude_err)
                      ]

        frame_cols = [
                    'cat_idx',
                      'ra',
                      'dec',
                      'x_pix_source',
                      'y_pix_source',
                      'x_pix',
                      'y_pix',
                      'cp_dist',
                      'dist2target',
                      'fwhm',
                      'cat_'+filter_,
                      'cat_'+filter_+'_err',

                      ]

        data_new_frame = pd.DataFrame(frame_data).T
        data_new_frame.columns = frame_cols
        
        # print(data_new_frame['x_pix'].values)
        # print(data_new_frame['y_pix'].values)
        
        # nans = (np.isnan(data_new_frame['x_pix'].values))| (np.isnan(data_new_frame['y_pix'].values)) 
        
        # print(nans)

        data_new_frame = data_new_frame.dropna(subset=['x_pix', 'y_pix'])

        if syntax['matching_source_FWHM']:

            extended_mask = abs(data_new_frame['fwhm'] - fwhm) < syntax['matching_source_FWHM_limt']

            print('\nCatalog sources removed: %d/%d' % (sum(extended_mask),len(data_new_frame)))

            data_new_frame = data_new_frame[extended_mask]


        warnings.filterwarnings("default")

    except Exception as e:
        logger.exception(e)


    return data_new_frame,syntax