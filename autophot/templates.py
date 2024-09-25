
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:27:05 2022

@author: seanbrennan
"""


def download_panstarrs_template(ra, dec, size, template_folder, f="r"):
    """
    Downloads a Pan-STARRS image template for a given RA, Dec, and size in the specified filter.

    Parameters:
    ra (float): Right Ascension of the target in degrees.
    dec (float): Declination of the target in degrees.
    size (int): Pixel size of the image.
    template_folder (str): Folder to save the downloaded template.
    f (str, optional): Filter to use for the image ('g', 'r', 'i', 'z'). Defaults to 'r'.

    Returns:
    str: File path of the downloaded template file, or None if download fails.
    """

    # Import necessary libraries
    import numpy as np
    from astropy.table import Table
    from astropy.io import fits
    import requests
    import pandas as pd
    import os,sys
    import pathlib
    import astropy.wcs as wcs
    from astropy.utils.exceptions import AstropyWarning
    import warnings

    # Filter out Astropy warnings
    warnings.filterwarnings('ignore', category=AstropyWarning)

    # Check if filter is valid
    if f.strip() not in ['g', 'r', 'i', 'z']:
        print(f'{f}-band not in PanSTARRS catalog [griz]')
        return

    print(f'Searching for {f}-band image from Pan-STARRS server...\n')

    try:
        # Define the URL for the Pan-STARRS service
        format = 'fits'
        delimiter = ','
        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = (f"{service}?ra={ra}&dec={dec}&size={size}&format={format}&sep={delimiter}&filters={f}")

        # Start a session to download the image metadata
        with requests.Session() as s:
            myfile = s.get(url)
            text = np.array([line.decode('utf-8') for line in myfile.iter_lines()])

        # Parse the metadata into a DataFrame and Astropy Table
        text = [text[i].split(',') for i in range(len(text))]
        df = pd.DataFrame(text)
        df.columns = df.loc[0].values
        table = Table.from_pandas(df.reindex(df.index.drop(0)).reset_index(drop=True))

        # Construct the URL for the FITS image download
        urlbase = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?ra={ra}&dec={dec}"
                   f"&size={size}&format={format}&filters={f}")
        flist = [f.find(x) for x in table['filter']]
        table = table[np.argsort(flist)]

        urls = [f"{urlbase}&red={filename}" for filename in table['filename']]

        # Define the path to save the template
        template_fname = f'panstarrs_{f}_band_template.fits'
        pathlib.Path(template_folder).mkdir(parents=True, exist_ok=True)
        template_sub_folder = os.path.join(template_folder, f'{f}p_template')
        pathlib.Path(template_sub_folder).mkdir(parents=True, exist_ok=True)
        template_fpath = os.path.join(template_sub_folder, template_fname)

        # Check if the file already exists
        if os.path.exists(template_fpath):
            print('File already exists... ignoring...')
            return

        # Check if there are URLs to download
        if len(urls) < 1:
            print(f'\tCannot find {f}-band image... skipping...\n')
            return None

        # Download the FITS file
        with fits.open(urls[0], ignore_missing_end=True, lazy_load_hdus=True, ignore_missing_simple=True) as hdu:
            try:
                hdu.verify('silentfix+ignore')

                # Extract header information and create a new header
                headinfo_template = hdu[0].header
                new_header = fits.PrimaryHDU().header
                new_header['Telescop'] = 'PS1'
                new_header['Instrume'] = 'GPC1'
                new_header['Filter'] = f
                new_header['Gain'] = headinfo_template['CELL.GAIN']
                new_header['MJD-OBS'] = headinfo_template['MJD-OBS']
                new_header['EXPTIME'] = headinfo_template['EXPTIME']

                # Update header with WCS information
                template_wcs = wcs.WCS(headinfo_template, relax=True)
                new_header.update(template_wcs.to_header(), relax=True)

                # Write the FITS file with the new header
                fits.writeto(template_fpath, hdu[0].data, new_header, overwrite=True, output_verify='silentfix+ignore')

            except Exception as e:
                print(f'Issue with {urls[0]}\n{e}')

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname1, exc_tb.tb_lineno, e)
        return None

    return template_fpath

# =============================================================================
# 
# =============================================================================

class templates:
    """
    A class to handle the creation of source masks for image processing.

    Attributes:
    ----------
    input_yaml : str
        Path to the input YAML configuration file.
    """

    def __init__(self, input_yaml):
        """
        Initializes the Templates class with the given input YAML file.

        Parameters:
        ----------
        input_yaml : str
            Path to the input YAML configuration file.
        """
        self.input_yaml = input_yaml
        
    def apply_mask_to_fits(self,fits_file, mask_file):
        from astropy.io import fits
        import numpy as np
        # Open the FITS file and mask file
        with fits.open(fits_file, mode='update') as hdul_data, fits.open(mask_file) as hdul_mask:
            data = hdul_data[0].data  # Assuming data is in the primary HDU
            mask = hdul_mask[0].data  # Assuming mask is in the primary HDU
            
            # Ensure the mask is boolean
            mask = mask.astype(bool)

            # Apply the mask: mask out regions where mask is True
            data[mask] = np.nan  # Here, setting masked values to 0

            # Write changes back to the FITS file
            hdul_data.flush()

    def create_source_mask(self, dataframe, shape, radius=7, nsources=10):
        """
        Creates a source mask for the image based on the positions of sources.

        Parameters:
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing source information with 'x_pix' and 'y_pix' columns for positions.
        shape : tuple
            Shape of the image mask (height, width).
        radius : int, optional
            Radius of the circular aperture for each source (default is 7).
        nsources : int, optional
            Number of sources to consider for the mask (default is 10).

        Returns:
        -------
        numpy.ndarray
            Inverse mask with sources masked out.
        """
        # Import necessary libraries
        import numpy as np
        from photutils.aperture import CircularAperture

        # Initialize the mask with zeros (no sources masked)
        mask = np.zeros(shape, dtype=int)

        # Limit the DataFrame to the top `nsources` sources
        dataframe = dataframe.head(nsources)
        print(f'Creating mask with {len(dataframe)} sources')

        # Create a mask using circular apertures centered at each source's pixel coordinates
        for idx, source in dataframe.iterrows():
            center_x = source['x_pix']
            center_y = source['y_pix']

            # Define a circular aperture mask for the current source
            mask_circular_aperture = CircularAperture((center_x, center_y), r=radius)

            # Convert the aperture mask to an image and add it to the cumulative mask
            mask += mask_circular_aperture.to_mask().to_image(shape=mask.shape).astype(int)

        # Ensure mask values are binary (0 or 1)
        mask[mask > 1] = 1

        # Create the inverse mask (1 where there are no sources, 0 where there are sources)
        inv_mask = np.ones(mask.shape) - mask

        return inv_mask

    # =============================================================================
    #         
    # =============================================================================
    
    def does_box_overlap(self, source, ignore_position, padding=25):
        """
        Checks if a given source's bounding box overlaps with a specific position, considering padding.
    
        Parameters:
        ----------
        source : dict
            A dictionary containing the bounding box information with keys 'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax'.
        ignore_position : tuple
            A tuple (x, y) representing the position to check against the bounding box.
        padding : int, optional
            Extra padding to consider around the bounding box boundaries (default is 10).
    
        Returns:
        -------
        bool
            True if the ignore_position is within the adjusted bounding box boundaries, False otherwise.
        """
        
        # Use the bounding box coordinates directly
        left = source['bbox_xmin']
        right = source['bbox_xmax']
        top = source['bbox_ymin']
        bottom = source['bbox_ymax']
        
        # Adjust boundaries by padding
        left -= padding
        right += padding
        top -= padding
        bottom += padding
        
        # Extract x, y coordinates from the ignore_position
        x, y = ignore_position
    
        # Check if the ignore_position is within the adjusted bounding box
        if left <= x <= right and top <= y <= bottom:
            return True
        
        return False


    # =============================================================================
    # 
    # =============================================================================
    
    def find_non_uniform_center(self, img):
        """
        Find the center of the non-uniform region in an image where rows or columns are uniform.
        
        Parameters:
        ----------
        img : np.ndarray
            2D numpy array representing the image.
        
        Returns:
        -------
        tuple
            (center_y, center_x, top_row, bottom_row, left_col, right_col)
            Coordinates of the center and bounding box of the non-uniform region.
        """
        import numpy as np
        
        # Identify non-uniform rows: rows that do not have all elements equal
        non_uniform_rows = ~np.all(img == img[:, 0][:, np.newaxis], axis=1)
        
        # Identify non-uniform columns: columns that do not have all elements equal
        non_uniform_cols = ~np.all(img == img[0, :], axis=0)
    
        # Calculate the top and bottom row indices of the non-uniform region
        top_row = np.where(non_uniform_rows)[0].min()
        bottom_row = np.where(non_uniform_rows)[0].max()
        
        # Calculate the left and right column indices of the non-uniform region
        left_col = np.where(non_uniform_cols)[0].min()
        right_col = np.where(non_uniform_cols)[0].max()
    
        # Calculate the center of the bounding box for the non-uniform region
        center_y = (top_row + bottom_row) / 2
        center_x = (left_col + right_col) / 2
    
        return center_y, center_x, top_row, bottom_row, left_col, right_col

    
    # =============================================================================
    # 
    # =============================================================================

    def find_bright_sources(self, header, usefilter=['J'], magCutoff=[14],catalogName = 'refcat'):
        """
        Finds bright sources in the specified filter(s) within the image using the provided header information.
    
        Parameters:
        ----------
        header : astropy.io.fits.Header
            FITS header containing WCS information of the image.
        usefilter : list of str, optional
            List of filters to use when identifying bright sources (default is ['J']).
        magCutoff : list of float, optional
            Magnitude cutoff(s) for each filter to select bright sources (default is [14]).
    
        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the pixel positions ('x_pix', 'y_pix') of bright sources.
            Returns None if an exception occurs.
        """
        import astropy.wcs as wcs
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        import os
        import logging
        from functions import border_msg
        from catalog import catalog
    
        try:
            
            # catalogName = 'refcat'
            # Set up logging
            logger = logging.getLogger(__name__)
            logger.info(border_msg(f'Masking bright sources from {catalogName} catalog'))
    
            # Retrieve target coordinates from YAML configuration
            target_ra = self.input_yaml['target_ra']
            target_dec = self.input_yaml['target_dec']
            target_coords = SkyCoord(target_ra, target_dec, unit=(u.deg, u.deg))
    
            # Initialize WCS from the FITS header
            imageWCS = wcs.WCS(header, relax=True, fix=False)
    
            # Initialize catalog object to retrieve source data
            sequenceData = catalog(input_yaml=self.input_yaml)
    
            # Download catalog data for the target region and filter by specified radius
            bright_sources_catalog = sequenceData.download(target_coords, catalogName=catalogName)
    
            # Clean and filter the catalog data based on the specified filters and magnitude cutoffs
            bright_sources_catalog = sequenceData.clean(
                selectedCatalog=bright_sources_catalog,
                catalogName = catalogName,
                image_wcs=imageWCS,
                get_local_sources=False,
                border=0,
                full_clean=False,
                usefilter=usefilter,
                magCutoff=magCutoff
            )
    
            # Extract pixel positions of the bright sources
            bright_sources_catalog = bright_sources_catalog[['x_pix', 'y_pix']]
    
            logger.info(f'Found {len(bright_sources_catalog )} for masking')
    
        except Exception as e:
            # Handle exceptions and print error details
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
            return None
    
        return bright_sources_catalog

    
    
# =============================================================================
#     
# =============================================================================
    def create_image_mask(self, data, sat_lvl=2**16, fwhm=5, npixels=11, padding=10, snr_limit=3000, 
                          create_source_mask=True, ignore_position=[], remove_large_sources=False, bright_sources=None):
        """
        Creates an image mask to filter out sources based on various criteria.
    
        Parameters:
        ----------
        data : np.ndarray
            The 2D array of image data.
        sat_lvl : int, optional
            Saturation level to filter out saturated sources (default is 65536).
        fwhm : int, optional
            Full Width at Half Maximum (FWHM) for Gaussian kernel (default is 5).
        npixels : int, optional
            Minimum number of connected pixels to detect a source (default is 11).
        padding : float, optional
            Padding factor to apply around detected sources (default is 1.5).
        snr_limit : float, optional
            Signal-to-noise ratio limit for source detection (default is 300).
        create_source_mask : bool, optional
            Whether to create a mask for detected sources (default is True).
        ignore_position : list of tuples, optional
            List of positions to ignore when creating the mask (default is empty list).
        remove_large_sources : bool, optional
            Whether to remove large sources from the mask (default is False).
        bright_sources : pandas.DataFrame, optional
            DataFrame containing the positions of bright sources (default is None).
    
        Returns:
        -------
        np.ndarray
            A binary mask of the same shape as `data`, with detected sources masked out.
        """
        import numpy as np
        from photutils.segmentation import detect_sources, SourceCatalog, make_2dgaussian_kernel, deblend_sources
        from astropy.convolution import convolve
        from photutils.aperture import RectangularAperture
        from astropy.stats import sigma_clipped_stats, sigma_clip
    
    
    
        from functions import pix_dist
        try:
            # Compute basic statistics of the image data using sigma-clipping
            image_mean, image_median, image_std = sigma_clipped_stats(data, sigma=3.0, cenfunc=np.nanmedian, stdfunc='mad_std')
            
            # Ensure FWHM is an even integer
            fwhm = int(fwhm)
            fwhm = fwhm if fwhm % 2 == 0 else fwhm + 1
            
            
            npixels = int(np.pi * (fwhm / 2)**2)
            # Define detection threshold for source detection
            threshold = 3 * image_std + image_median
            
            # Create a Gaussian kernel and convolve it with the data
            kernel = make_2dgaussian_kernel(fwhm, size=5*fwhm + 1)
            convolved_data = convolve(data, kernel)
            
            # Detect sources in the convolved data
            segment_map = detect_sources(data, threshold, npixels=npixels)
            
            # Deblend overlapping sources
            # segm_deblend = deblend_sources(data, segment_map, npixels=npixels, progress_bar=False)
            
            # Create a source catalog from the deblended segmentation map
            cat = SourceCatalog(data,segment_map, localbkg_width=15*fwhm)
            
            # Convert the catalog to a pandas DataFrame for easier manipulation
            tbl = cat.to_table().to_pandas()
            
            # Initialize the mask with zeros
            mask = np.zeros_like(data, dtype=int)
            
            # Handle bright sources if provided
            if not (bright_sources is None):
                counter = 0
                for _, source in tbl.iterrows():
                    counter+=1
                    try:
                        # Calculate center, width, and height of the source bounding box
                        center_x = (source['bbox_xmin'] + source['bbox_xmax']) / 2
                        center_y = (source['bbox_ymin'] + source['bbox_ymax']) / 2
                        width = source['bbox_xmax'] - source['bbox_xmin']
                        height = source['bbox_ymax'] - source['bbox_ymin']
                        
                        # width = 5 * fwhm
                        # height = 5*fwhm
                        
                        
                        if not any(pix_dist(bright_sources['x_pix'].values, center_x, bright_sources['y_pix'].values, center_y) < 3*fwhm): continue
                    
                        # # Check if source overlaps with any bright sources
                        # if any([self.does_box_overlap(source, pos,padding=padding) for pos in zip(bright_sources['x_pix'].values, bright_sources['y_pix'].values)]):
                        #     pass
                        # else:
                        #     continue
                        
                        if any([self.does_box_overlap(source, pos,padding=padding) for pos in ignore_position]): continue 
                        # if any([pix_dist(center_x, pos[0], center_y, pos[1])< 3*fwhm  for pos in ignore_position]): continue
                        # # # Check if source overlaps with any ignore positions
                        # if any([self.does_box_overlap(source, pos,padding=padding) for pos in ignore_position]): 
                        #     continue 
                        
                        
                        # if height * width > fwhm*10 : continue
                        # Define a rectangular aperture for masking
                        mask_rectangular_aperture = RectangularAperture(
                            (center_x, center_y),
                            w=width + padding,
                            h=height+ padding,
                            theta=0  # Angle in degrees
                        )
                        
                        # label = 
                        
                        mask_i = np.zeros(mask.shape, dtype=bool)
                        mask_i[segment_map.data == counter] = True
                        
                        # Add the rectangular aperture to the mask
                        # mask += mask_rectangular_aperture.to_mask().to_image(shape=mask.shape).astype(int)
                        mask+=mask_i
                        
                
                    except Exception as e:
                        # Handle exceptions for individual source processing
                        import sys, os
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(f"Error: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
                        
                # print(f"> Masked {counter} bright sources")
                
            # Remove saturated sources based on the saturation level
            saturated_sources = tbl['max_value'] > sat_lvl 
            
            # Remove large sources using sigma-clipping on the area
            clipped_data = sigma_clip(tbl['area'], sigma=5, maxiters=10)
            large_sources = clipped_data.mask
                
            if not remove_large_sources:
                large_sources = np.array([False] * len(tbl))
                
            # Remove sources with negative peaks
            negative_peak_sources = tbl['max_value'] < 0
            
            tbl = tbl[(saturated_sources) | (negative_peak_sources) | (large_sources)]

            # Create a mask using rectangular apertures for the filtered sources
            for _, source in tbl.iterrows():
                break
                try:
                    # Calculate center, width, and height of the source bounding box
                    center_x = (source['bbox_xmin'] + source['bbox_xmax']) / 2
                    center_y = (source['bbox_ymin'] + source['bbox_ymax']) / 2
                    width = source['bbox_xmax'] - source['bbox_xmin']
                    height = source['bbox_ymax'] - source['bbox_ymin']
                    
                    # Skip if source overlaps with any ignore positions
                    if any([self.does_box_overlap(source, pos,padding=5*fwhm) for pos in ignore_position]): continue 
                    
                    mask_i = np.zeros(mask.shape, dtype=bool)
                    mask_i[segment_map.data == counter] = True
                    
                    # Add the rectangular aperture to the mask
                    # mask += mask_rectangular_aperture.to_mask().to_image(shape=mask.shape).astype(int)
                    mask+=mask_i
            
                except Exception as e:
                    # Handle exceptions for individual source processing
                    import sys, os
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f"Error: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
            
            # Ensure the mask is binary (0 or 1)
            mask[mask > 1] = 1
                
        except Exception as e:
            # General exception handling
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {exc_type} in {fname} at line {exc_tb.tb_lineno}: {e}")
    
        return mask

    


# =============================================================================
#         
# =============================================================================
        
    def getTemplate(self):
        """
        Finds the appropriate template file based on the input YAML configuration.
    
        Returns:
        -------
        str or None
            The path to the template file if found, otherwise None.
        """
        import os
        import logging
        from functions import border_msg
        
        try:
            # Set up logger
            logger = logging.getLogger(__name__)
            
            # Log the start of the template file search
            logger.info(border_msg('Finding template file'))
            
            # Determine the filter template to use based on the input YAML configuration
            use_filter = self.input_yaml.get('imageFilter')
            if not use_filter:
                logger.error("Image filter not specified in the input YAML.")
                return None
            
            use_filter_template = use_filter + 'p' if use_filter in ['g', 'r', 'i', 'z', 'u'] else use_filter
            
            # Construct the expected path to the template folder
            expected_template_folder = os.path.join(self.input_yaml.get('fits_dir', ''), 'templates')
            expected_filter_template_folder = os.path.join(expected_template_folder, use_filter_template + '_template')
            
            if not os.path.exists(expected_filter_template_folder):
                logger.error(f"Template folder does not exist: {expected_filter_template_folder}")
                return None
            
            # List all files in the expected template folder
            list_dir = os.listdir(expected_filter_template_folder)
            
            # Filter the list to include only relevant FITS files, excluding PSF model files
            fits_list_dir = [i for i in list_dir if i.lower().endswith(('fits', 'fts', 'fit')) and 'PSF_model_' not in i]
            
            # Further filter the list to include only files with '_template' in the name
            fits_list_dir = [i for i in fits_list_dir if '_template' in i]
            
            if not fits_list_dir:
                logger.error(f"No template files found in folder: {expected_filter_template_folder}")
                return None
            
            # Construct the full path to the template file
            templateFpath = os.path.join(expected_filter_template_folder, fits_list_dir[0])
            
            # Log the template file path
            logger.info(f'Template filepath: {templateFpath}')
            
        except Exception as e:
            # Log any exceptions that occur
            logger.exception(f"Error finding template file: {e}")
            return None
        
        # Return the path to the template file
        return templateFpath

    # =============================================================================
    #     
    # =============================================================================
        
    def align(self, scienceFpath, templateFpath, imageCatalog=None, useWCS=False):
        """
        Aligns a science image with a template image using source matching or WCS alignment.
    
        Parameters:
        ----------
        scienceFpath : str
            File path to the science image.
        templateFpath : str
            File path to the template image.
        imageCatalog : pandas.DataFrame, optional
            Catalog of sources detected in the science image (default is None).
        useWCS : bool, optional
            Flag to determine if WCS alignment should be used (default is False).
    
        Returns:
        -------
        tuple
            A tuple containing:
            - new_templateFpath (str): The path to the newly aligned template file.
            - template_lst (np.ndarray): Array of template source positions, or None if WCS alignment is used.
            - science_lst (np.ndarray): Array of science source positions, or None if WCS alignment is used.
        """
        import logging
        import os
        import numpy as np
        import pandas as pd
        import astropy.units as u
        import astropy.wcs as wcs
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        from astropy.utils.exceptions import AstropyWarning
        from functions import border_msg
        import astroalign as aa
        import warnings
        from reproject import reproject_interp
        from aafitrans import find_transform
    
        # Suppress Astropy warnings
        warnings.filterwarnings('ignore', category=AstropyWarning)
        
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Aligning science and reference images'))
        
        # File paths and names
        templateDir = os.path.dirname(templateFpath)
        scienceDir = os.path.dirname(scienceFpath)
        templateName = os.path.basename(templateFpath)
        new_templateFpath = os.path.join(scienceDir, templateName)
        
        # Open and read science image
        with fits.open(scienceFpath, ignore_missing_end=True, lazy_load_hdus=True) as hdu:
            scienceHeader = hdu[0].header
            scienceImage = hdu[0].data.astype(float)
            imageWCS = wcs.WCS(scienceHeader, relax=True)
        
        # Open and read template image
        with fits.open(templateFpath, ignore_missing_end=True, lazy_load_hdus=True) as hdu:
            templateHeader = hdu[0].header
            templateImage = hdu[0].data.astype(float)
    
        # Align using AstroAlign if imageCatalog is provided and useWCS is False
        if imageCatalog is not None and not useWCS:
            try:
                logger.info('Aligning via AstroAlign using source matching')
                
                # Load template calibration data
                template_calib_data_loc = os.path.join(templateDir, 'all_sources.csv')
                logger.info(f'Template calibration file: {template_calib_data_loc}')
                template_calib_data = pd.read_csv(template_calib_data_loc)
                coords_template_i = SkyCoord(template_calib_data['RA'], template_calib_data['DEC'], unit=(u.deg, u.deg))
                
                matchingDist = np.ceil(self.input_yaml['pixel_scale'] * self.input_yaml['fwhm']*2)
                logger.info(f'Matching sources within {matchingDist:.1f} arcseconds')
                
                science_lst = []
                template_lst = []
                
                # Match sources between the science image and template
                for _, row in imageCatalog.iterrows():
                    coords_science_i = SkyCoord(row['RA'], row['DEC'], unit=(u.deg, u.deg))
                    sep = coords_science_i.separation(coords_template_i)
                    matching_sources = template_calib_data[sep.arcsecond < matchingDist]

                    if len(matching_sources) == 1:
                        science_lst.append((row['x_pix'], row['y_pix']))
                        template_lst.append((float(matching_sources['x_pix'].iloc[0]), float(matching_sources['y_pix'].iloc[0])))
                
                if len(science_lst) == 0:
                    raise ValueError('No matching sources could be found for alignment - skipping')
                
                matching_limit = 3
                if len(science_lst) < matching_limit:
                    raise ValueError(f'Less than {matching_limit} common sources found [{len(science_lst)}] for alignment - skipping')
                else:
                    logger.info(f'Found {len(science_lst)} matching sources')
                
                science_lst = np.array(science_lst)
                template_lst = np.array(template_lst)

                tform, (matched_source_xy, matched_target_xy) = find_transform(template_lst, science_lst,
                                                                                max_control_points=50,
                                                                                # ttype="sim",
                                                                               pixel_tolerance=2,
                                                                               min_matches=4,
                                                                               num_nearest_neighbors=5,
                                                                               kdtree_search_radius=0.02,
                                                                               n_samples=4,
                                                                               get_best_fit=True,
                                                                               seed=None)

                templateImage, footprint = aa.apply_transform(tform, templateImage, scienceImage)
                
                # Handle masked and NaN values
                templateImage[footprint.astype(bool)] = 1e-30
                templateImage[np.isnan(templateImage)] = 1e-30
                
                logger.info('\n> Images successfully aligned with AstroAlign ')
                templateHeader.update(imageWCS.to_header(), relax=True)
                useWCS = False
                
                # Save the aligned template image
                fits.writeto(new_templateFpath, templateImage, templateHeader, overwrite=True, output_verify='silentfix+ignore')
                
                return new_templateFpath, matched_source_xy, matched_target_xy
                
            except Exception as e:
                logger.error(f"Alignment with AstroAlign failed: {e}")
                useWCS = True
    
        # Align using WCS if required
        if useWCS:
            logger.info('Aligning via WCS with reproject_interp')
            try:
                with fits.open(templateFpath, ignore_missing_end=True, lazy_load_hdus=True) as hdu:
                    templateImage, footprint = reproject_interp(hdu[0], output_projection=scienceHeader)
                    templateImage[~footprint.astype(bool)] = 1e-30
                    templateImage[np.isnan(templateImage)] = 1e-30
                    templateHeader.update(imageWCS.to_header(), relax=True)
                
                # Save the aligned template image
                fits.writeto(new_templateFpath, templateImage, templateHeader, overwrite=True, output_verify='silentfix+ignore')
                
                
                logger.info('\n> Images successfully aligned with reproject_interp ')
            except Exception as e:
                logger.error(f"Alignment with WCS failed: {e}")
                return None, None, None
    
        return new_templateFpath, None, None

    # =============================================================================
    #             
    # =============================================================================
        
    def process_fits_file(self,data,header,coords):
 
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D

    
        """
        Process the FITS file by finding the largest available rectangle and updating the WCS using Cutout2D.
        Also plots the original and final image with Z-scale.
        :param fits_file: Path to the input FITS file.
        """

        # Get the WCS information
        wcs = WCS(header)

        # Define the center and size of the cutout based on the largest rectangle
        cutout_center = ((coords[2] + coords[0]) // 2, (coords[3] + coords[1]) // 2)
        cutout_shape = (coords[2] - coords[0] + 1, coords[3] - coords[1] + 1)

        # Create the cutout
        cutout = Cutout2D(data, position=cutout_center, size=cutout_shape, wcs=wcs)

        header.update(cutout.wcs.to_header())
        
        return cutout.data,header
    
    
    
    # Function to find the largest rectangle in a histogram row
    def largest_histogram_rectangle(self,heights):
        """
        Finds the largest rectangle in a histogram (row of heights).
        :param heights: List of heights representing the current row.
        :return: (max_area, (left, height, right)) for the largest rectangle.
        """
        stack = []
        max_area = 0
        best_coords = (0, 0, 0)  # (left, height, right)
    
        for i in range(len(heights) + 1):
            while stack and (i == len(heights) or heights[i] < heights[stack[-1]]):
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    best_coords = (stack[-1] + 1 if stack else 0, h, i - 1)
            stack.append(i)
        
        return max_area, best_coords
    
    # Function to find largest rectangle
    def find_largest_available_area(self,image):
        """
        Finds the largest rectangle of non-NaN values in the 2D image array.
        :param image: 2D numpy array with NaN values marking unavailable areas.
        :return: tuple of (x_start, y_start, x_end, y_end) for the largest available area.
        """
        import numpy as np
        # Mask the NaN values, NaNs become False, others become True
        mask = ~np.isnan(image)
        
        # Get the dimensions of the image
        rows, cols = mask.shape
        
        # This will store the largest rectangle coordinates
        best_area = 0
        best_coords = (0, 0, 0, 0)
        
        # To store the height of continuous True values for each column
        height = np.zeros(cols, dtype=int)
        
        for r in range(rows):
            for c in range(cols):
                # Update height (continuous non-NaN values) for each column
                if mask[r, c]:
                    height[c] += 1
                else:
                    height[c] = 0
    
            # Find the largest rectangle for the current row
            area, coords = self.largest_histogram_rectangle(height)
            
            # Update if we found a better area
            if area > best_area:
                best_area = area
                best_coords = (r - coords[1], coords[0], r, coords[2])
        
        return best_coords
    
    
    # =============================================================================
    #     
    # =============================================================================

   
    def crop(self, scienceFpath, templateFpath):
        """
        Crops the science and template images around a specific target location, excluding uniform rows and columns.
    
        Parameters:
        ----------
        scienceFpath : str
            File path to the science image.
        templateFpath : str
            File path to the template image.
    
        Returns:
        -------
        tuple
            A tuple containing:
            - cropped_scienceFpath (str): The path to the cropped science image file.
            - cropped_templateFpath (str): The path to the cropped template image file.
        """
        import logging
        import numpy as np
        import os
        from astropy.io import fits
        from astropy.nddata.utils import Cutout2D
        import astropy.wcs as wcs
        from functions import border_msg, distance_to_uniform_row_col
        from astropy.stats import sigma_clipped_stats
    
        # Set up logger
        logger = logging.getLogger(__name__)
        logger.info(border_msg('Cropping science and reference images'))
    
        # Define file paths
        scienceDir = os.path.dirname(scienceFpath)
        scienceName = os.path.basename(scienceFpath)
        cropped_scienceFpath = os.path.join(scienceDir, 'cutout_' + scienceName)
        cropped_templateFpath = templateFpath  # Default to input template path
    
        # Target pixel coordinates
        target_x_pix = np.floor(self.input_yaml['target_x_pix'])
        target_y_pix = np.floor(self.input_yaml['target_y_pix'])
    
        # Open science image
        with fits.open(scienceFpath, ignore_missing_end=True, lazy_load_hdus=True) as hdu:
            scienceHeader = hdu[0].header
            scienceImage = hdu[0].data.astype(float)
    
        # Open template image
        with fits.open(templateFpath, ignore_missing_end=True, lazy_load_hdus=True) as hdu:
            templateHeader = hdu[0].header
            templateImage = hdu[0].data.astype(float)
    
        # Initialize WCS
        imageWCS = wcs.WCS(scienceHeader, relax=True)
    
        # Find the center of the non-uniform region
        center_y, center_x, top_row, bottom_row, left_col, right_col = self.find_non_uniform_center(templateImage)
    
        # Calculate the height and width of the cropped image
        height = bottom_row - top_row 
        width = right_col - left_col 
        position = (np.floor(center_x), np.floor(center_y))  # Note: Cutout2D expects (x, y) position
        
        height = (height if height%2==0 else height-1)
        width =  (width if width%2==0 else width-1)
        size = (height, width)
    
        logger.info(f'Non-uniform image center: {center_x:.1f}, {center_y:.1f}')
    
        # Calculate distances to uniform rows/columns
        distance_uniform_T = distance_to_uniform_row_col(templateImage, x=target_x_pix, y=target_y_pix)
        distance_uniform_I = distance_to_uniform_row_col(scienceImage, x=target_x_pix, y=target_y_pix)
    
        # Perform cropping if distances are finite
        if (np.isfinite(distance_uniform_T) or np.isfinite(distance_uniform_I) ):
            logger.info('Cropping images around non-uniform region')
    
            # Crop the science image
            scienceCutout = Cutout2D(
                data=scienceImage,
                position=position,
                size=size, 
                wcs=imageWCS,
                mode='partial',
                fill_value=1e-30
            )
            imageWCS = wcs.WCS(scienceCutout.wcs.to_header(), relax=True)
            scienceHeader.update(imageWCS.to_header(), relax=True)
            scienceImage = scienceCutout.data
    
            # Crop the template image
            templateCutout = Cutout2D(
                data=templateImage,
                position=position,
                size=size,
                mode='partial',
                fill_value=1e-30
            )
            templateImage = templateCutout.data
    
        else:
            logger.info('No cropping performed')
            cropped_scienceFpath = scienceFpath
            cropped_templateFpath = templateFpath
    
        # Mask and replace pixels
        idx = (templateImage == 1e-30) | (scienceImage == 1e-30)
        template_mean, template_median, template_std = sigma_clipped_stats(templateImage[~idx], sigma=3.0, cenfunc=np.nanmedian, stdfunc='mad_std')
        science_mean, science_median, science_std = sigma_clipped_stats(scienceImage[~idx], sigma=3.0, cenfunc=np.nanmedian, stdfunc='mad_std')
    
        templateImage[idx] = np.nan
        scienceImage[idx] = np.nan
        
        # When aligned and cuting the images you can be left with triangular ares - lets try to exclude them
        
        coords = self.find_largest_available_area(scienceImage)

        scienceImage_tmp,scienceHeader_tmp, =   self.process_fits_file(scienceImage, header = scienceHeader, coords = coords)     
        templateImage_tmp,templateHeader_tmp, =   self.process_fits_file( templateImage, header = templateHeader, coords = coords)     
        
        # header = get_header(scienceFpath)
        imageWCS_tmp = wcs.WCS(scienceHeader_tmp,relax = True,fix = False)

        target_x_pix, target_y_pix = imageWCS_tmp.all_world2pix(self.input_yaml['target_ra'],
                                                                self.input_yaml['target_dec'],
                                                            0)
        
        
        border = 2 * self.input_yaml['scale']
        width =    scienceImage_tmp.shape[1]
        height =    scienceImage_tmp.shape[0]
        
        inside_x = (target_x_pix >= border) & (target_x_pix < width - border)
        inside_y = (target_y_pix >= border) & (target_y_pix < height - border)
        
        if inside_x and inside_y:
            templateImage = templateImage_tmp
            templateHeader = templateHeader_tmp
            
            scienceImage = scienceImage_tmp
            scienceHeader =scienceHeader_tmp
            
            
        # Write the cropped images to files
        fits.writeto(
            cropped_templateFpath,
            templateImage,
            templateHeader,
            overwrite=True,
            output_verify='silentfix+ignore'
        )
        fits.writeto(
            cropped_scienceFpath,
            scienceImage,
            scienceHeader,
            overwrite=True,
            output_verify='silentfix+ignore'
        )
    
        return cropped_scienceFpath, cropped_templateFpath

              
# =============================================================================
#     
# =============================================================================

    def subtract(self, scienceFpath, templateFpath, method='sfft', stamps=None,
                 gain_matching_sources=None,allow_subpixel_shifts = False,
                 common_sources = []):
        """
        Subtracts the template image from the science image using either ZOGY or HOTPANTS.
        
        Parameters:
        - scienceFpath: Path to the science image file.
        - templateFpath: Path to the template image file.
        - useHotpants: Boolean flag to use HOTPANTS for subtraction if True, otherwise use ZOGY.
        - stamps: Optional; Path to stamp file for HOTPANTS.
        - gain_matching_sources: Optional; Not used in the current version.
        
        Returns:
        - subtractionFpath: Path to the saved subtraction result.
        - universal_mask: The combined mask for the science and template images.
        """
        
        import subprocess
        import os
        import sys
        import numpy as np
        import time
        from functions import get_image, get_header, border_msg, save_to_fits, get_image_stats
        from astropy.io import fits
        import logging, gc
        import warnings
        from glob import glob
        
        start = time.time()
        # Calculate the elapsed time

        # Garbage collection
        gc.collect()
        
        # Initialize logging
        logger = logging.getLogger(__name__)
        
        # Suppress Astropy warnings
        from astropy.utils.exceptions import AstropyWarning
        warnings.filterwarnings('ignore', category=AstropyWarning)
        
        logger.info(border_msg('Subtracting reference from science image'))
        
        try:
            # Load images and headers
            scienceImage = get_image(scienceFpath)
            templateImage = get_image(templateFpath)
            scienceHeader = get_header(scienceFpath)
            templateHeader = get_header(templateFpath)
            
            
            if allow_subpixel_shifts:
                from  image_registration import chi2_shift                
                from scipy.ndimage import shift
                dx,dy,edx2,edy2 = chi2_shift(scienceImage,templateImage, return_error=True, upsample_factor='auto')
                
                
                if abs(dx) < 1 and abs(dy) < 1  :
                    logger.info(f'\nShifting template: dx = {-1 * dx:.3f} px | dy = {-1 * dy:.3f} px...\n')
                    templateImage = shift(templateImage, (-1 * dy,-1 * dx),order=3,prefilter=True)
                
                       
                    fits.writeto(templateFpath,
                                  templateImage,
                                  templateHeader,
                                  overwrite = True,
                                  output_verify = 'silentfix+ignore')

            # Extract image parameters
            science_fwhm = scienceHeader['fwhm']
            template_fwhm = templateHeader['fwhm']
            m = 'AP'
            science_zeropoint = scienceHeader[f'zpoint_{m}']
            template_zeropoint = templateHeader[f'zpoint_{m}']
            science_gain = scienceHeader['gain'] or 1
            template_gain = templateHeader['gain'] or 1
            science_saturate = scienceHeader['saturate']
            template_saturate = templateHeader['saturate']
            
            target_location = [(self.input_yaml['target_x_pix'], self.input_yaml['target_y_pix'])]
            
            # Directories
            scienceDir = os.path.dirname(scienceFpath)
            templateDir = os.path.dirname(templateFpath)
            
            # PSF models
            science_psf = glob(os.path.join(scienceDir, 'PSF_model_*fits'))[0] if glob(os.path.join(scienceDir, 'PSF_model_*fits')) else None
            template_psf = glob(os.path.join(scienceDir, 'template_PSF_model_*fits'))[0] if glob(os.path.join(scienceDir, 'template_PSF_model_*fits')) else None
            
            # Output file path
            fname = os.path.basename(scienceFpath)
            subtractionFpath = os.path.join(scienceDir, 'diff_' + fname)
            
            logger.info('> Image FWHM: %.1f px' % science_fwhm)
            logger.info('> Template FWHM: %.1f px' % template_fwhm)
            logger.info('> Image Gain: %.1f' % science_gain)
            logger.info('> Template Gain: %.1f' % template_gain)
            logger.info('> Image Saturation: %.1f ADU' % science_saturate)
            logger.info('> Template Saturation: %.1f ADU' % template_saturate)
            logger.info('> Image Zeropoint: %.1f mag' % science_zeropoint)
            logger.info('> Template Zeropoint: %.1f mag' % template_zeropoint)
            
            # Generate masks
            science_mask = None
            template_mask = None
            
            bright_sources = self.find_bright_sources(header=scienceHeader)
            
            if template_mask is None:
                template_mask = ((abs(templateImage) < 1.1e-29) & (templateImage != 0)) | (~np.isfinite(templateImage))
                template_mask = template_mask.astype(int)
                template_seg_mask = self.create_image_mask(templateImage, 
                                                            sat_lvl=template_saturate,
                                                            fwhm=template_fwhm,
                                                            create_source_mask=False,
                                                            ignore_position=target_location,
                                                            remove_large_sources=False,
                                                            bright_sources=bright_sources,
                                                            padding = 5*template_fwhm,)
                template_mask += template_seg_mask.astype(int)
                template_mask = template_mask.astype(int)
            
            if science_mask is None:
                science_mask = ((abs(scienceImage) < 1.1e-29) & (scienceImage != 0)) | (~np.isfinite(scienceImage))
                science_mask = science_mask.astype(int)
                science_seg_mask = self.create_image_mask(scienceImage, 
                                                          sat_lvl=science_saturate,
                                                          fwhm=science_fwhm,
                                                          create_source_mask=False,
                                                          ignore_position=target_location,
                                                          remove_large_sources=False,
                                                          bright_sources=bright_sources,
                                                          padding = 5 * science_fwhm )
                science_mask += science_seg_mask.astype(int)
                science_mask = science_mask.astype(int)
            
            # Calculate image statistics
            scienceMean, scienceMedian, scienceSTD = get_image_stats(scienceImage[~science_mask.astype(bool)])
            templateMean, templateMedian, templateSTD = get_image_stats(templateImage[~template_mask.astype(bool)])
            
            # Combine masks
            universal_mask = science_mask + template_mask
            universal_mask[universal_mask > 1] = 1
            
            mask_loc = os.path.join(templateDir, 'universal_mask.fits')
            save_to_fits(universal_mask.astype(int), mask_loc)
            
            
            logger.info(f'Masking {(np.sum(universal_mask) / (universal_mask.shape[1] * universal_mask.shape[0]) ) :.3f}% of the image before template subtraction')
            
            
            
            
            if  method == 'zogy':
                try:
                    logger.info(border_msg('\nRunning Zogy...\n'))
                    from PyZOGY.subtract import run_subtraction
                    diff = run_subtraction(science_image=scienceFpath,
                                            reference_image=templateFpath,
                                            science_psf=science_psf,
                                            reference_psf=template_psf,
                                            science_mask=universal_mask,
                                            reference_mask=universal_mask,
                                            show=False,
                                            normalization="science",
                                            science_saturation=science_saturate,
                                            reference_saturation=template_saturate,
                                            reference_variance=templateSTD**2,
                                            science_variance=scienceSTD**2,
                                            max_iterations=30,
                                            use_pixels=False,
                                            size_cut=True)
                    diff = diff[0]
                    fits.writeto(subtractionFpath, diff, scienceHeader, overwrite=True, output_verify='silentfix+ignore')
                    
                   
                except Exception as e:
                    logger.info(border_msg(f'ZOGY failed for image subtraction - {e} - trying HOTPANTS'))
                   
                    
            elif method == 'sfft':    
                try:
                    # Logging the start of the SFFT process
                    logger.info(border_msg('Running SFFT ...'))
                    current_directory = os.path.dirname(os.path.abspath(__file__))
                    sfft_exe = os.path.join(current_directory, 'utils/run_sfft.py')
                    
             
                    base = os.path.splitext(os.path.basename(scienceFpath))[0]
                    sfft_log = os.path.join(scienceDir, 'sfft_' + base + '.txt')
                    
                    
                    from functions import apply_mask_image
                    apply_mask_image(image_path = scienceFpath, mask_path = mask_loc)
                    apply_mask_image(image_path = templateFpath, mask_path = mask_loc)
                    
        
                    with open(sfft_log, 'w') as FNULL:
                        # subprocess.run(' '.join(args), shell=True, check=True, text=True, stdout=FNULL, stderr=FNULL)
                        
                        args = f'python {sfft_exe} -sci {scienceFpath} -ref {templateFpath} -diff {subtractionFpath} -mask {mask_loc} '
                        subprocess.run(args, shell=True, check=True, text=True, stdout=FNULL, stderr=FNULL)
                            
                    gc.collect()
    
                    
                
                except Exception as e:
                    # Error handling: log the exception with details
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno, e)

               
            elif method == 'hotpants':
                logger.info(border_msg('Running HOTPANTS...'))
                base = os.path.splitext(os.path.basename(scienceFpath))[0]
                exe = self.input_yaml['template_subtraction']['hotpants_exe_loc']
                
                hotpants_fwhm = max(np.ceil(template_fwhm), np.ceil(science_fwhm))
                if hotpants_fwhm % 2 == 0:
                    hotpants_fwhm += 1
    
                r = max(2.5 * hotpants_fwhm, 13)
                rss = max(6 * hotpants_fwhm, 31)
                
                include_args = [
                    ('-inim', str(scienceFpath)),
                    ('-tmplim', str(templateFpath)),
                    ('-outim', str(subtractionFpath)),
                    ('-il', str(scienceMedian - 10 * scienceSTD)),
                    ('-tl', str(templateMedian - 10 * templateSTD)),
                    ('-tu', str(template_saturate)),
                    ('-iu', str(science_saturate)),
                    ('-imi', str(mask_loc)),
                    ('-tmi', str(mask_loc)),
                    ('-n', "i"),
                    ('-c', "t"),
                    ('-v', '2'),
                    ('-r', str(r)),
                    ('-rss', str(rss)),
                    ('-ko', '1'),
                    ('-bgo', '0')
                ]
                
                if stamps:
                    include_args.append(('-ssf', stamps))
                
                args = [str(exe)] + [f"{arg[0]} {arg[1]}" for arg in include_args]

                HOTPANTS_log = os.path.join(scienceDir, 'HOTPANTS_' + base + '.txt')
                with open(HOTPANTS_log, 'w') as FNULL:
                    subprocess.run(' '.join(args), shell=True, check=True, text=True, stdout=FNULL, stderr=FNULL)
                    print('\nARGUMENTS:\n', ' '.join([i.replace('-', '\n-') if not i.isnumeric() else i for i in args]), file=FNULL)
                
                logger.info('HOTPANTS finished: %ss' % round(time.time() - start))
            
            # Check if the subtraction file was created successfully
            if os.path.isfile(subtractionFpath):
                if os.path.getsize(subtractionFpath) == 0:
                    logger.info('File was created but nothing written')
                    return np.nan
                else:
                    logger.info('Difference image saved as %s' % os.path.basename(subtractionFpath))
                    subtractedHeader = get_header(subtractionFpath)
                    subtractionImage = get_image(subtractionFpath)
                    fits.writeto(subtractionFpath, subtractionImage, subtractedHeader, overwrite=True, output_verify='silentfix+ignore')
                            
                    end =  time.time() -start
            
                    logger.info(f'Subtraction done [{end:.1f}]s')
                    return subtractionFpath, universal_mask
                
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, e)
            return None, None
        
        
        # Garbage collection
        gc.collect()

        
        return subtractionFpath, universal_mask
