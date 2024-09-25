
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:47:32 2022

@author: seanbrennan
"""


class catalog():
    
    def __init__(self, input_yaml):
            
        self.input_yaml = input_yaml
            
        
    def fetch_refcat2_field(self, ra,dec, credentials,nsources = 500,sr = 0.5):
        
        try:
  
       
            # Import necessary modules
            from mastcasjobs import MastCasJobs
            import random
            import string
            import logging
            # Logger setup
            logger = logging.getLogger(__name__)
            # Generate a unique name for the catalog
            name = 'autophot_{}'.format(
                ''.join(random.choices(string.ascii_uppercase, k=5)))
            
            # name = 'autophot_refcat_search'            
            # Print a message indicating the fetching process
            logging.info(f'Fetching ATLAS-RefCat2 catalog from MAST over {sr:.1f} field-of-view centered at ra = {ra:.1f} dec =  {dec:.1f}')
            
            # Define the columns to be retrieved from the database
            table = ['RA','Dec','g','dg','r','dr',
                     'i','di','z', 'dz','J','dJ','H','dH',
                     'K','dK']

            # Create the SQL query to fetch data from the database
            q = '''
            SELECT TOP {max} {columns}
            INTO MyDB.{name}
            FROM fGetNearbyObjEq({ra},{dec},{sr}) as n
            INNER JOIN refcat2 AS r ON (n.objid=r.objid)
            WHERE r.dr < 0.1
            ORDER BY n.distance
            '''.format(
                max=nsources,
                columns='r.{}'.format(',r.'.join(table)),
                name=name,
                ra=ra,
                dec=dec,
                sr=sr
            )
            # Print the SQL query for debugging purposes
            logging.info(q)

            # Create a job to execute the SQL query
            job = MastCasJobs(context="HLSP_ATLAS_REFCAT2", **credentials)
            job.drop_table_if_exists(name)
            jobid = job.submit(q, task_name=('refcat catalog search {:.5f} {:.5f}'.format(ra, dec)))
            
            # Monitor the status of the job
            status = job.monitor(jobid)
            
            # Check if the job status indicates an error and raise an exception if so
            if status[0] in (3, 4):
                raise Exception('status={}, {}'.format(status[0], status[1]))
            
            # Retrieve the result table in CSV format and drop the temporary table
            tab = job.get_table(name, format='CSV')
            job.drop_table_if_exists(name)
            
            # Convert the result table to a pandas DataFrame
            tab = tab.to_pandas()
        
            # Remove any row where any value is zero
            tab = tab[~(tab == 0).any(axis=1)]
            tab.reset_index(inplace=True)


            
        except Exception as e:
            import sys,os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno,e)
            return None
            


        return tab
    
    # =============================================================================
    #     
    # =============================================================================
                
    def download(self, target_coords, catalogName, radius=15, target_name=None,
             catalog_custom_fpath=None, include_IR_sequence_data=True):
        """
        Download and process catalog data for a given target.
    
        Parameters:
        - target_coords: SkyCoord object with the RA and DEC of the target.
        - catalogName: Name of the catalog to fetch ('refcat', 'gaia', 'apass', '2mass', 'sdss', 'skymapper', 'pan_starrs', 'custom').
        - radius: Search radius around the target in degrees.
        - target_name: Optional name for the target.
        - catalog_custom_fpath: File path to a custom catalog (used if catalogName is 'custom').
        - include_IR_sequence_data: Boolean to include IR sequence data from 2MASS.
    
        Returns:
        - DataFrame containing the catalog data, or None if an error occurs.
        """
        import os
        import sys
        import logging
        import requests
        import shutil
        import warnings
        import pathlib
    
        import numpy as np
        import pandas as pd
        from functools import reduce
        
        from functions import autophot_yaml
        
        from astropy.table import Table
        from astroquery.vizier import Vizier
        from astropy.io.votable import parse_single_table
        from astropy.coordinates import Angle, SkyCoord
        from astropy import units as u
    
        # Logger setup
        logger = logging.getLogger(__name__)
    
        try:
    
    
            # Set default target name if not provided
            if not target_name:
                target_name = self.input_yaml.get('target_name', 'target')
    
            # Set default custom catalog path if not provided
            if not catalog_custom_fpath:
                catalog_custom_fpath = self.input_yaml['catalog'].get('catalog_custom_fpath', None)
    
            # Working directory
            wdir = self.input_yaml.get('wdir')
            if not wdir:
                raise ValueError('Working directory (wdir) is not set in input YAML.')
    
            # Define paths and directories
            filepath = os.path.dirname(os.path.abspath(__file__))
            catalog_autophot_input_yml = 'catalog.yml'
            catalog_keywords = autophot_yaml(os.path.join(filepath, 'databases', catalog_autophot_input_yml), catalogName).load()
    
            # Create directories for storing catalog data
            dirname = os.path.join(wdir, 'catalog_queries')
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            catalog_dir = os.path.join(dirname, catalogName)
            pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
            target_dir = reduce(os.path.join, [dirname, catalogName, target_name.lower()])
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
    
            # Generate file name for the catalog
            fname = f"{target_name}_r_{radius:.1f}arcmins_{catalogName}".replace(' ', '')
            
            # Convert from arcmins to degrees
            radius = radius / 60
    
            # Handle different catalog sources
            if catalogName == 'custom':
                if not catalog_custom_fpath:
                    logger.critical('Custom catalog selected but "catalog_custom_fpath" is not defined.')
                    return None
                selectedCatalog = pd.read_csv(catalog_custom_fpath)
            elif os.path.isfile(os.path.join(target_dir, f"{fname}.csv")):
                logger.info(f'\n> Existing {catalogName.upper()} catalog found for {target_name}\n')
                
                selectedCatalog = Table.read(os.path.join(target_dir, f"{fname}.csv"), format='csv').to_pandas().fillna(np.nan)
            else:
                selectedCatalog = []
    
                if catalogName == 'refcat':
                    credentials = {
                        'userid': self.input_yaml['catalog'].get('MASTcasjobs_wsid'),
                        'password': self.input_yaml['catalog'].get('MASTcasjobs_pwd')
                    }
                    if not credentials['userid'] or not credentials['password']:
                        logger.critical('Refcat catalog selected but credentials are missing.')
                        sys.exit()
                        
                        
                        
                    selectedCatalog = self.fetch_refcat2_field(ra=target_coords.ra.degree, dec=target_coords.dec.degree, 
                                                               credentials=credentials, nsources=500, sr=radius)
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(os.path.join(os.getcwd(), f"{fname}.csv"), os.path.join(target_dir, f"{fname}.csv"))
    
                elif catalogName == 'gaia':
                    from astroquery.gaia import Gaia
                    warnings.filterwarnings('ignore')
                    width = u.Quantity(radius, u.deg)
                    height = u.Quantity(radius, u.deg)
                    selectedCatalog = Gaia.query_object_async(coordinate=target_coords, width=width, height=height).to_pandas()
                    selectedCatalog.to_csv(f"{fname}.csv", sep=',', index=False)
    
                elif catalogName in ['apass', '2mass', 'sdss']:
                    Vizier.ROW_LIMIT = -1
                    catalog_search = Vizier.query_region(target_coords, radius=Angle(radius, 'deg'), catalog=catalogName)
                    if len(catalog_search)<1:
                        selectedCatalog = None
                    else:
                        selectedCatalog = catalog_search[0].to_pandas()
                        if catalogName == 'sdss':
                            selectedCatalog = selectedCatalog[selectedCatalog['mode'] == 1]
                        selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                        shutil.move(os.path.join(os.getcwd(), f"{fname}.csv"), os.path.join(target_dir, f"{fname}.csv"))
    
                elif catalogName == 'skymapper':
                    server = 'http://skymapper.anu.edu.au/sm-cone/public/query?'
                    params = {'RA': target_coords.ra.degree, 'DEC': target_coords.dec.degree, 'SR': radius, 'RESPONSEFORMAT': 'VOTABLE'}
                    with open('temp.vot', "wb") as f:
                        logger.info('> Downloading sequence stars from %s', catalogName)
                        response = requests.get(server, params=params)
                        f.write(response.content)
                    selectedCatalog = parse_single_table('temp.vot').to_table(use_names_over_ids=True).to_pandas()
                    os.remove('temp.vot')
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(os.path.join(os.getcwd(), f"{fname}.csv"), os.path.join(target_dir, f"{fname}.csv"))
    
                elif catalogName == 'pan_starrs':
                    mindet = 1
                    server = 'https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean?'
                    params = {'ra': target_coords.ra.degree, 'dec': target_coords.dec.degree, 'radius': radius, 'format': 'csv', 'nDetections.gte': f'{mindet}'}
                    with open('temp.vot', "wb") as f:
                        logger.info('> Downloading sequence stars from %s', catalogName)
                        response = requests.get(server, params=params)
                        f.write(response.content)
                        
                        
                    selectedCatalog = pd.read_csv('temp.vot')
                    selectedCatalog = selectedCatalog.replace(-999, np.nan)
                    columns = ["raMean", "decMean", "raMeanErr", "decMeanErr", "gMeanPSFMag", "gMeanPSFMagErr", "rMeanPSFMag", "rMeanPSFMagErr", "iMeanPSFMag", "iMeanPSFMagErr", "zMeanPSFMag", "zMeanPSFMagErr", "yMeanPSFMag", "yMeanPSFMagErr"]
                    selectedCatalog = selectedCatalog[columns]
                    coords = SkyCoord(ra=selectedCatalog['raMean'].values * u.deg, dec=selectedCatalog['decMean'].values * u.deg)
                    distances = target_coords.separation(coords)
                    selectedCatalog['distance'] = distances.arcsecond
                    selectedCatalog = selectedCatalog.sort_values(by='distance').head(300)
                    os.remove('temp.vot')
                    selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                    shutil.move(os.path.join(os.getcwd(), f"{fname}.csv"), os.path.join(target_dir, f"{fname}.csv"))
    
                else:
                    logger.critical('Catalog %s is not recognized.', catalogName)
                    sys.exit()
    
                # # Optionally include IR sequence data from 2MASS
                # if include_IR_sequence_data and catalogName != '2mass':
                #     Vizier.ROW_LIMIT = -1
                #     catalog_search = Vizier.query_region(target_coords, radius=Angle(radius, 'deg'), catalog='2mass')
                #     selectedCatalog_2mass = catalog_search[0].to_pandas()
                #     selectedCatalog_2mass.rename(columns={"Jmag": "J", "e_Jmag": "J_err", "Hmag": "H", "e_Hmag": "H_err", "Kmag": "K", "e_Kmag": "K_err", 'RAJ2000': catalog_keywords['RA'], 'DEJ2000': catalog_keywords['DEC']}, inplace=True)
                #     selectedCatalog = pd.concat([selectedCatalog, selectedCatalog_2mass])
                #     selectedCatalog.to_csv(f"{fname}.csv", index=False, na_rep=np.nan)
                #     shutil.move(os.path.join(os.getcwd(), f"{fname}.csv"), os.path.join(target_dir, f"{fname}.csv"))
    
                logger.info('> %s catalog contains %d sources', catalogName.upper(), len(selectedCatalog))
                warnings.filterwarnings("default")
    

        except Exception as e:
            logger.error('\n> Catalog retrieval failed! \n')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error('%s %s %d %s', exc_type, fname, exc_tb.tb_lineno, e)
            selectedCatalog = None

        return selectedCatalog
    
            
    # =============================================================================
    #     
    # =============================================================================
    def clean(self, selectedCatalog, catalogName = None, usefilter=None, magCutoff=21, border=11, image_wcs=None,
          fwhm=5, get_local_sources=False, full_clean=True, update_names_only = False):
        """
        Clean the catalog of sources by applying various filters and removing unwanted sources.
    
        Parameters:
        - selectedCatalog: DataFrame containing the sources with initial measurements.
        - usefilter: List of filters to apply (default is None, which uses the filter specified in self.input_yaml).
        - magCutoff: Magnitude cutoff for filtering sources (default is 21).
        - border: Border size to exclude sources near the edge of the image (default is 11 pixels).
        - image_wcs: World Coordinate System transformation to convert RA/DEC to pixel coordinates.
        - fwhm: Full width at half maximum, used for determining crowding (default is 5).
        - get_local_sources: Whether to retrieve local sources (default is False).
        - full_clean: Whether to apply full cleaning procedures (default is True).
    
        Returns:
        - DataFrame with cleaned catalog.
        """
        import os
        import logging
        import numpy as np
        import pandas as pd
        from functions import autophot_yaml, border_msg, pix_dist
        
    
        # Initialize logger
        logger = logging.getLogger(__name__)
    

    
        try:
            # Load catalog configuration
            filepath = os.path.dirname(os.path.abspath(__file__))
            catalog_autophot_input_yml = 'catalog.yml'
            
            if catalogName is None:
                catalogName = self.input_yaml['catalog']['use_catalog']
            

            # try:
            catalog_keywords = autophot_yaml(os.path.join(filepath, 'databases', catalog_autophot_input_yml),
                                             catalogName).load()
            
            # print(catalog_keywords)
            # Prepare final catalog
            ouputCatalog = pd.DataFrame({
                'RA': selectedCatalog[catalog_keywords['RA']].values,
                'DEC': selectedCatalog[catalog_keywords['DEC']].values,

            })
            
            # Convert RA/DEC to pixel coordinates
            if image_wcs:
                
                
                x_pix, y_pix = image_wcs.all_world2pix(selectedCatalog[catalog_keywords['RA']].values,
                                                       selectedCatalog[catalog_keywords['DEC']].values, 0)

                    
                ouputCatalog ['x_pix'] = x_pix
                ouputCatalog ['y_pix'] = y_pix

                
 
            
            try:
                if not self.input_yaml['HST_mode']:
                    ouputCatalog [self.input_yaml['imageFilter']] = selectedCatalog[catalog_keywords[self.input_yaml['imageFilter']]].values
                    ouputCatalog [self.input_yaml['imageFilter'] + '_err'] = selectedCatalog[catalog_keywords[self.input_yaml['imageFilter'] + '_err']].values
            except:
                pass
                # logger.info(f"{catalogName} does not contain {self.input_yaml['imageFilter']} band magnitudes")
                
            # Attempt to get all available filter information in the catalog
            baseFilepath = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))
            baseDatabase = os.path.join(baseFilepath, 'databases')
            filters_yml = 'filters.yml'
            availableFilters = autophot_yaml(os.path.join(baseDatabase, filters_yml)).load()
    
            for filter_x in availableFilters['default_dmag'].keys():
                if filter_x in catalog_keywords:
                    try:
                        ouputCatalog [ filter_x] = selectedCatalog[catalog_keywords[filter_x]].values
                        ouputCatalog [ filter_x + '_err'] = selectedCatalog[catalog_keywords[filter_x + '_err']].values
                    except KeyError:
                        pass  # Skip if filter information is not available
                        
                        
            if update_names_only:
                return ouputCatalog 
            
            
            if full_clean:
                logger.info(border_msg('Cleaning sequence data'))


    
            # Determine filters to use
            if usefilter is None:
                usefilter = [self.input_yaml['imageFilter']]
                magCutoff = [magCutoff]
    
         
            # Apply magnitude cutoff filter
            if not full_clean:
                tooFaint = np.array([False] * len(ouputCatalog))
                for i, usefilter_i in enumerate(usefilter):
                    if catalog_keywords[usefilter_i] in ouputCatalog .columns:
                        
                        
                        cutoff_i = magCutoff[i]
                        tooFaint_i = ouputCatalog [usefilter_i].values > cutoff_i
                        if sum(tooFaint_i) > 0:
                            logger.info('Ignoring %d source fainter than %s = %.1f mag' % (sum(tooFaint_i), usefilter_i, cutoff_i))
                            tooFaint[tooFaint_i] = True
                if len(tooFaint)>0:
                    logger.info(f'{sum(~tooFaint)} remaining that meet crtieria')
                    ouputCatalog  = ouputCatalog [~tooFaint]
            
            # Full cleaning procedures
            if full_clean:
                
                try:
                    # Remove sources with no filter information
                    hasFilterinfo = np.isfinite(ouputCatalog[self.input_yaml['imageFilter']].values)
                    if sum(~hasFilterinfo) > 0:
                        logger.info('Excluding %d sources with no %s band information' % (sum(~hasFilterinfo), self.input_yaml['imageFilter']))
                        selectedCatalog = selectedCatalog[hasFilterinfo]

                except: pass
    
                # Remove sources that are too close to their neighbors
                if not (image_wcs is None):

                    # Select sources closest to target if there are more than 100 sources
                    if len(ouputCatalog ) > 100:
                        logger.info('Selecting 100 sources closest to target location')
                        dist = pix_dist(float(self.input_yaml['target_x_pix']), ouputCatalog ['x_pix'].values,
                                        float(self.input_yaml['target_y_pix']), ouputCatalog ['y_pix'].values)
                        ouputCatalog ['dist_2_target'] = dist
                        ouputCatalog .sort_values(by='dist_2_target', inplace=True)
                        ouputCatalog  = ouputCatalog .head(100)
    
           
    
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, e)
    
        return ouputCatalog 


    
# =============================================================================
#     
# =============================================================================
    def recenter(self, selectedCatalog, image, boxsize=None):
        """
        Recenter sources in an image.
        
        Parameters:
        - selectedCatalog: DataFrame containing the source catalog with initial pixel coordinates.
        - image: 2D array representing the image where sources are located.
        - boxsize: Size of the box around each source for centering, in pixels. If None, it defaults to 3 times the FWHM from the input YAML.
        
        Returns:
        - Updated DataFrame with recentered pixel coordinates.
        """
        import logging
        from functions import border_msg, pix_dist
        from photutils.centroids import centroid_sources, centroid_com
        import numpy as np
        
        logger = logging.getLogger(__name__)
        
        try:
            # Log information about the recentering process
            if len(selectedCatalog) > 1:
                logger.info(border_msg('Recentering %d sources' % len(selectedCatalog)))
            else:
                logger.info(border_msg('Recentering Target'))
        
            # Determine the box size if not provided
            if boxsize is None and 'fwhm' in self.input_yaml:
                fwhm = self.input_yaml['fwhm']
                boxsize = int(np.ceil(fwhm)) * 3
            else:
                boxsize = 10
        
            # Ensure box size is an odd number and at least 3
            boxsize = int(boxsize if boxsize % 2 != 0 else boxsize + 1)
            boxsize = max(boxsize, 3)
        
            border = boxsize * 3
            logger.info(f'Boxsize: {boxsize:.0f} px')
        
            # Apply a border mask if there are multiple sources
            if len(selectedCatalog) > 1:
                width, height = image.shape[1], image.shape[0]
                mask_x = (selectedCatalog['x_pix'] > border) & (selectedCatalog['x_pix'] < width - border)
                mask_y = (selectedCatalog['y_pix'] > border) & (selectedCatalog['y_pix'] < height - border)
                mask = mask_x & mask_y
        
                logger.info('Recentering %d sources within border' % sum(mask))
                selectedCatalog = selectedCatalog[mask]
        
            # Recenter the sources
            old_x, old_y = selectedCatalog['x_pix'].values, selectedCatalog['y_pix'].values
        
            # Handle NaN values in image by replacing with a small number
            mask = ~np.isfinite(image)
            image[mask] = 1e-30
        
            x, y = centroid_sources(image, old_x, old_y, box_size=boxsize, centroid_func=centroid_com, mask=mask)
        
            # Update pixel coordinates in the catalog
            selectedCatalog['x_pix'] = x
            selectedCatalog['y_pix'] = y
        
            # Reapply border mask after recentering
            if len(selectedCatalog) > 1:
                width, height = image.shape[1], image.shape[0]
                mask_x = (selectedCatalog['x_pix'] >= border) & (selectedCatalog['x_pix'] < width - border)
                mask_y = (selectedCatalog['y_pix'] >= border) & (selectedCatalog['y_pix'] < height - border)
                mask = mask_x & mask_y
        
                if sum(~mask) > 0:
                    logger.info('Failed to recenter %d sources - ignoring' % sum(~mask))
                    selectedCatalog = selectedCatalog[mask]
        
            # Calculate and log the median pixel correction
            average_offset = np.nanmedian(pix_dist(x, old_x, y, old_y))
            logger.info(f'> Median pixel correction: {average_offset:.1f} px')
        
            # Drop rows with NaN values in 'x_pix' or 'y_pix'
            selectedCatalog.dropna(subset=['x_pix', 'y_pix'], inplace=True)
        
        except Exception as e:
            logger.error('Error occurred during recentering: %s', e)
            import traceback
            logger.error(traceback.format_exc())
        
        return selectedCatalog

    # =============================================================================
    #     
    # =============================================================================
    # Updated function to check if source exists using 5 arcsecond tolerance

    def find_source(self, ra, dec, catalog, tolerance=5.0):
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import numpy as np
        
        # Convert the RA, Dec of the input source to a SkyCoord object
        target_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        
        # Prepare an empty list to store matching indices
        matching_indices = []
        
        # Loop over each entry in the catalog
        for i in range(len(catalog)):
            # Get the RA and DEC of the current catalog entry
            catalog_coord = SkyCoord(ra=catalog['RA'][i]*u.degree, dec=catalog['DEC'][i]*u.degree)
            
            # Calculate the angular separation between the target and the catalog entry
            separation = target_coord.separation(catalog_coord)
            
            # If the separation is within the tolerance (in arcseconds), store the index
            if separation.arcsecond <= tolerance:
                matching_indices.append(i)
        
        # Return the filtered catalog based on the matching indices
        return catalog[np.array(matching_indices)]


    
    def build_complete_catalog(self, target_coords,target_name = None, 
                               catalog_list = ['refcat','sdss','pan_starrs','apass','2mass'], radius = 2,max_seperation = 3):
        
        
        import os
        import pandas as pd
        import numpy as np
        import pathlib
        import numpy as np
        import pandas as pd
        from functools import reduce
        
        from functions import autophot_yaml
        
        import os
        import logging
        import numpy as np
        import pandas as pd
        from functions import autophot_yaml, border_msg, pix_dist
        
    
        # Initialize logger
        logger = logging.getLogger(__name__)
        
        
        catalog_list_str = ','.join([i.upper() for i in catalog_list])
        logger.info(border_msg(f"Building Custom catalog using from {catalog_list_str}"))
        # Set default target name if not provided
        if not target_name:
            target_name = self.input_yaml.get('target_name', 'target')


        catalog_custom_fpath = self.input_yaml['catalog'].get('catalog_custom_fpath', None)
                
        # Generate file name for the catalog
        fname = f"{target_name}_r_{radius}arcmins_CUSTOM.csv".replace(' ', '')
        
        wdir = self.input_yaml.get('wdir')
        if not wdir:
            raise ValueError('Working directory (wdir) is not set in input YAML.')
        dirname = os.path.join(wdir, 'catalog_queries')
        dirname  = os.path.join(dirname,'custom')
        
        # Create directories for storing catalog data
        dirname = os.path.join(wdir, 'catalog_queries')
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        catalog_dir = os.path.join(dirname, 'custom_builds')
        pathlib.Path(catalog_dir).mkdir(parents=True, exist_ok=True)
        # target_dir = reduce(os.path.join, [dirname, , target_name.lower()])
        # pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        
        fpath = os.path.join(catalog_dir ,fname)
        
        
        # if os.path.exists(fpath):
        #     logger.info('> Found exisiting custom built catalog\n')
        #     output_catalog = pd.read_csv(fpath)
        #     return output_catalog
        
        filter_list = ['u','g', 'r', 'i', 'z','U','B', 'V', 'R', 'I', 'Z', 'J', 'H', 'K']

        # Create a new list by appending '_err' to each element
        updated_filter_list = []
        for filter in filter_list:
            updated_filter_list.append(filter)
            updated_filter_list.append(f"{filter}_err")
            
            
        # Tolerance in degrees (5 arcseconds)
        tolerance_deg = max_seperation  / 3600  # Equals to 0.0013889 degrees
        

        # cols = ['RA','DEC'].extend( updated_filter_list)
        cols = ['RA','DEC'] + updated_filter_list  
        output_catalog = pd.DataFrame(columns = cols)

            
        for catalogName in catalog_list:
            
            
            logger.info(f'Getting {catalogName} catalog ... ')
            
            catalog_i = self.download(target_coords = target_coords, catalogName = catalogName,radius=radius)
            if catalog_i is None: continue
            catalog_i = self.clean(catalog_i,catalogName=catalogName,update_names_only = True)
            
            # Loop over each entry in the current catalog
            for index, entry in catalog_i.iterrows():
                ra = entry['RA']
                dec = entry['DEC']
                
                # Check if the source exists in the output catalog
                existing_source = self.find_source(ra, dec, output_catalog,tolerance=tolerance_deg)
                
                if existing_source.empty:
                    # Source does not exist, add a new row
                    new_row = {col: entry.get(col, np.nan) for col in cols}
                    new_row_df = pd.DataFrame([new_row])
                    
                    output_catalog = pd.concat([output_catalog, new_row_df], ignore_index=True)
                    
                else:
                    # Source exists, update the row with missing filter data
                    index = existing_source.index[0]
                    for filter_name in updated_filter_list:
                        if filter_name in entry and pd.isna(output_catalog.at[index, filter_name]):
                            output_catalog.at[index, filter_name] = entry[filter_name]
            
        # Final output catalog is ready
        output_catalog.to_csv(fpath, index=False, float_format='%.6f')

        return output_catalog
    
    
    
    # ============================================================================= 
    #     
    # =============================================================================
        
    def measure(self, selectedCatalog, image):
        """
        Measure the flux and signal-to-noise ratio (SNR) of sources in an image using aperture photometry.
    
        Parameters:
        - selectedCatalog: DataFrame containing the sources with initial coordinates.
        - image: 2D array representing the image where sources are located.
    
        Returns:
        - Updated DataFrame with measured flux, instrumental magnitude, and SNR for each source.
        """
        import logging
        from aperture import aperture
        from functions import border_msg, mag, SNR
        from numpy import isnan
    
        # Initialize logger
        logger = logging.getLogger(__name__)
    
        # Log the start of the measurement process
        logger.info(border_msg(f'Measuring {len(selectedCatalog)} sources in the field using aperture photometry'))
    
        # Initialize the aperture photometry object with the provided input YAML and image
        initialAperture = aperture(input_yaml=self.input_yaml, image=image)
    
        # Measure the sources using aperture photometry
        selectedCatalog = initialAperture.measure(sources=selectedCatalog)
    
        # Rename the flux column to reflect the aperture measurement
        selectedCatalog.rename(columns={'flux': 'flux_AP'}, inplace=True)
    
        # Calculate the instrumental magnitude based on the measured flux
        instMag = mag(selectedCatalog['flux_AP'])
        selectedCatalog['inst_' + self.input_yaml['imageFilter'] + '_AP'] = round(instMag, 3)
    
        # Calculate the SNR for each source
        sourceSNR = SNR(selectedCatalog['maxPixel'], selectedCatalog['noiseSky'])
        selectedCatalog['SNR'] = round(sourceSNR, 1)
    
        # Log the number of sources for which the flux was measured
        logger.info('Instrumental magnitude of %d sources measured' % sum(~isnan(selectedCatalog['flux_AP'])))
    
        return selectedCatalog
