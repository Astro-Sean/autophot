#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:49:37 2022

@author: seanbrennan
"""

class prepare():
    
    def __init__(self, default_input):
        """
        Initialize the prepare class with a path to the default input YAML file.
        
        :param default_input: Path to the default input YAML file
        :type default_input: str
        """
        self.input_yaml = default_input

        
    def load():
        """
        Load the default input YAML file and return its content.

        - Imports necessary modules.
        - Determines the directory path of the current script.
        - Constructs the full path to the default input YAML file.
        - Loads the YAML file using the `autophot_yaml` function.
        - Logs the path of the loaded file using `border_msg` for formatting.

        :return: Loaded YAML content
        :rtype: dict
        """
        import os
        import logging
        from functools import reduce
        from functions import autophot_yaml, border_msg
        
        # Get location of this script
        filepath = os.path.dirname(os.path.abspath(__file__))

        # Name of default input YAML file - do not change
        default_input = 'default_input.yml'

        '''
        reduce package from functools
        - Applies a function of two arguments cumulatively to the items of
        iterable, from left to right, so as to reduce the iterable to a single value.
        '''

        # Construct the file path for the default input YAML file
        default_input_filepath = reduce(os.path.join, [filepath, 'databases', default_input])

        # Load default commands from the YAML file
        default_input_yaml = autophot_yaml(default_input_filepath, 'default_input').load()

        # Log the path of the loaded YAML file
        logging.info(border_msg(f'Default input loaded in from: {default_input_filepath}', body='-', corner='+'))

        return default_input_yaml

    
    
    def clean(self):
        '''
        Function to run on image dataset and set up a list of files for use in AutoPHOT.
        This function performs the following tasks:
    
        1. Searches through a given file path and creates a list of acceptable images to use. This script will look for files with the following extensions:
            - *.fist*
            - *.fit*
            - *.fts* 
            - *fits.fz*
    
        Science images must not include the following in their filepath as these filenames are used later on in AutoPHOT (and will end with the same extension as the input image) and may cause errors:
            - *subtraction*
            - *template*
            - *.wcs.* 
            - *footprint*
            - *PSF_model_*
            - *sources_*
    
        2. Run through this file list and check if the correct information is available in the *telescope.yml* file using the *checkteledata* function.
    
        3. If a Transient Name Server (TNS) bot is available, return the latest coordinates of a given target.
    
        4. Search through the file list and remove any file that has an *IMAGETYP* of *bias*, *zero*, *flat*, *WAVE*, or *LAMP* or an *OBS_MODE* of *spectroscopy*.
    
        5. Run the final file list through the AutoPHoT pipeline.
    
        See `here
        <https://github.com/Astro-Sean/autophot/blob/master/example_notebooks/basic_example.ipynb>`_
        for an example.
    
        :param input_yaml: AutoPHOT input dictionary
        :type input_yaml: dict
        :return: Creates a CSV file containing photometric data for transient in dataset
        :rtype: Dataframe saved to work directory
        '''
    
        import os
        import pathlib
        import numpy as np
        from functions import print_progress_bar
    
        files_removed = 0
    
        # Remove trailing slash from directory path if present
        if self.input_yaml['fits_dir'].endswith('/'):
            self.input_yaml['fits_dir'] = self.input_yaml['fits_dir'][:-1]
    
        # Check if template subtraction is not to be prepared
        if not self.input_yaml['template_subtraction']['prepare_templates']:
            flist = []
    
            # Set up new directory path for output
            new_dir = '_' + self.input_yaml['outdir_name']
            base_dir = os.path.basename(self.input_yaml['fits_dir']).replace(new_dir, '')
            work_loc = base_dir + new_dir
            work_fpath = os.path.join(self.input_yaml['fits_dir'], work_loc)
            pathlib.Path(os.path.dirname(work_fpath)).mkdir(parents=True, exist_ok=True)
            # os.chdir(os.path.dirname(work_fpath))  # (commented out: change working directory)
    
            # Search for acceptable FITS files in the directory
            for root, dirs, files in os.walk(self.input_yaml['fits_dir']):
                for fname in files:
                    if fname.endswith((".fits", '.fit', '.fts', 'fits.fz')):
                        if 'templates' not in root and 'template' not in self.input_yaml['fits_dir']:
                            if 'template' not in fname and 'template' not in self.input_yaml['fits_dir']:
                                if 'subtraction' not in fname:
                                    if '.wcs' not in fname:
                                        if 'PSF_model' not in fname:
                                            if 'footprint' not in fname:
                                                if 'sources_' not in fname:
                                                    flist.append(os.path.join(root, fname))
        else:
            flist = []
            # Search within the 'templates' directory
            new_dir = '_' + self.input_yaml['outdir_name']
            base_dir = os.path.basename(self.input_yaml['fits_dir']).replace(new_dir, '')
            work_loc = base_dir + new_dir
            template_loc = os.path.join(self.input_yaml['fits_dir'], 'templates')
            for root, dirs, files in os.walk(template_loc):
                for fname in files:
                    if fname.endswith((".fits", '.fit', '.fts', 'fits.fz')):
                        if 'PSF_model' not in fname and '.wcs' not in fname:
                            flist.append(os.path.join(root, fname))
    
        files_completed = False
    
        # If restart is enabled and templates are not being prepared
        if self.input_yaml['restart'] and not self.input_yaml['template_subtraction']['prepare_templates']:
            # Pick up where left off in the output folder
            flist_before = []
    
            # Clean up file paths
            for i in flist:
                path, file = os.path.split(i)
                file_nodots, file_ext = os.path.splitext(file)
                # Replace dots with underscores in file name
                file_nodots = file_nodots.replace('.', '_')
                file = file_nodots + file_ext
                clean_path = os.path.join(path, file).replace('_APT', '').replace(' ', '_').replace('_' + self.input_yaml['outdir_name'], '')
                clean_path_split = list(clean_path.split('/'))
                sub_dirs = list(dict.fromkeys([i.replace(file_ext, '') for i in clean_path_split]))
                clean_path = '/'.join(sub_dirs)
                clean_fpath = os.path.join(clean_path, file.replace('_APT', ''))
                flist_before.append(clean_fpath)
    
            len_before = len(flist)
    
            print('\nRestarting - checking for files already completed in:\n-..%s' % (self.input_yaml['fits_dir'] + '_' + self.input_yaml['outdir_name']).replace(' ', ''))
    
            flist_restart = []
            ending = '_' + self.input_yaml['outdir_name']
            output_folder = self.input_yaml['fits_dir'] + ending
    
            # Look in the output directory for completed files
            for root, dirs, files in os.walk(output_folder.replace(' ', '')):
                for fname in files:
                    if '_APT.f' in fname:
                        if os.path.isfile(os.path.join(root, fname)) and os.path.isfile(os.path.join(root, 'output.csv')):
                            dirpath_clean_up = os.path.join(root, fname).replace(ending, '')
                            path, file = os.path.split(dirpath_clean_up)
                            clean_path = path.split('/')
                            clean_path_new = '/'.join(clean_path) + '/' + file
                            flist_restart.append(clean_path_new.replace('_APT', '').replace(' ', '_').replace('_' + self.input_yaml['outdir_name'], ''))
    
            if len(flist_before) == 0:
                print('\n\t->No output files found - skipping ')
            else:
                flist_bool = [False if f in flist_restart else True for f in flist_before]
                flist = list(np.array(flist)[np.array(flist_bool)])
                len_after = len(flist)
                print('\nTotal Files: %d' % len_before)
                files_completed = len_before - len_after
                print('\nFiles already done: %d' % files_completed)
                files_removed += len_before - len_after
    
        return flist

    
    
    
    
    
    def check_files(self, flist):
        '''
        Checks the list of FITS files to ensure they meet required criteria by using
        the `fits_info` function from the `check` module.
    
        This function performs the following tasks:
        1. Calls the `fits_info` function, passing the input YAML dictionary and 
           the list of files to check.
        2. Returns a cleaned list of files that meet the criteria specified in 
           `fits_info`.
    
        :param flist: List of file paths to be checked
        :type flist: list of str
        :return: List of files that meet the required criteria
        :rtype: list of str
        '''
        
        from check import fits_info  # Import the `fits_info` function from the `check` module
        
        # Create an instance of `fits_info` and call the `check` method to filter the files
        cleanFlist = fits_info(input_yaml=self.input_yaml, flist=flist).check()
        
        return cleanFlist  # Return the filtered list of files

    
    
    def check_catalog(self):
        '''
        Checks the available filters in the catalog and includes additional filters if specified.
        
        This function performs the following tasks:
        
        1. Loads the catalog YAML file based on the configuration specified in the input YAML.
        2. If a custom catalog is selected:
           - Checks if a custom file path is provided.
           - Loads the custom catalog data from the specified path.
           - Determines which filters are available in the custom catalog.
        3. If not using a custom catalog:
           - Determines which filters are available based on the catalog YAML.
        4. If specified, includes additional IR sequence filters ('J', 'H', 'K') in the list of available filters.
        
        :return: List of available filters from the catalog
        :rtype: list of str
        '''
        
        import os
        from functions import autophot_yaml
        import pandas as pd
        
        # Retrieve the catalog type from the input YAML
        selected_catalog = self.input_yaml['catalog']['use_catalog']
        
        # Define the path to the catalog YAML file
        filepath = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))
        catalog_yml = 'catalog.yml'
        
        # Load the catalog YAML file based on the selected catalog type
        catalog_input_yaml = autophot_yaml(os.path.join(filepath+'/databases', catalog_yml), selected_catalog).load()
        
        
        if self.input_yaml['catalog']['build_catalog']:
            available_filters = [i for i, _ in catalog_input_yaml.items()]
            
        
        # If a custom catalog is selected
        elif selected_catalog == 'custom':
            
            target = self.input_yaml['target_name']
            fname = str(target) + '_RAD_' + str(float(self.input_yaml['catalog']['catalog_radius']))
            
            if not self.input_yaml['catalog']['catalog_custom_fpath']:
                # Print error message and exit if the custom file path is not defined
                print('Custom catalog selected but "catalog_custom_fpath" not defined')
                exit()
            else:
                # Use the custom file path specified in the input YAML
                fname = self.input_yaml['catalog']['catalog_custom_fpath']
            
            # Load the custom catalog data
            custom_table_data = pd.read_csv(fname)
            
            # Determine which filters are available in the custom catalog
            available_filters = [i for i, _ in catalog_input_yaml.items() if i in list(custom_table_data.columns)]
        else:
            # Determine which filters are available based on the catalog YAML
            available_filters = [i for i, _ in catalog_input_yaml.items()]
        
        # =============================================================================
        # Include IR sequence points if specified
        # =============================================================================
        
        if self.input_yaml['catalog']['include_IR_sequence_data']:
            # Add IR sequence filters to the available filters list
            available_filters += ['J', 'H', 'K']
            # Remove duplicates by converting to a set and then back to a list
            available_filters = list(set(available_filters))
        
        return available_filters  # Return the list of available filters

        
    
    def check_TNS(self):
        '''
        Checks for transient object information using the Transient Name Server (TNS) API.
        
        This function performs the following tasks:
        
        1. Creates a directory to store TNS object YAML files if it doesn't already exist.
        2. Checks if existing TNS information for the target is available locally:
           - If found, loads and prints the information.
           - If not found, attempts to retrieve the latest TNS data using the TNS API.
        3. Creates a YAML file with the retrieved TNS data if new data is fetched.
        4. Handles exceptions and errors, ensuring the function exits if there's a problem with reaching the server.
        5. If TNS information cannot be obtained and no RA/DEC coordinates are provided, prompts the user to decide whether to continue.
        
        :return: Dictionary containing TNS or user-provided coordinates
        :rtype: dict
        '''
        
        import os
        import sys
        import pathlib
        from functools import reduce
        from functions import autophot_yaml
        from tns import get_coords
    
        # Retrieve the target name from the input YAML
        target_name = self.input_yaml['target_name']
    
        # Create a directory to store TNS object files if it does not exist
        pathlib.Path(os.path.join(self.input_yaml['wdir'], 'tns_objects')).mkdir(parents=True, exist_ok=True)
    
        # Check if target name and TNS Bot ID are provided
        if self.input_yaml['target_name'] is not None and self.input_yaml['wcs']['TNS_BOT_ID'] is not None:
            # Construct the path for the transient YAML file
            transient_path = reduce(os.path.join, [self.input_yaml['wdir'], 'tns_objects', target_name + '.yml'])
    
            # Check if the TNS file already exists
            if os.path.isfile(transient_path):
                # Load existing TNS information from the YAML file
                TNS_response = autophot_yaml(transient_path, target_name).load()
                print('\nFound existing information for %s' % (TNS_response['objname']))
            else:
                try:
                    print('\nChecking TNS for %s information' % self.input_yaml['target_name'])
    
                    # Retrieve new TNS data using the TNS API
                    TNS_response = get_coords(
                        objname=self.input_yaml['target_name'],
                        TNS_BOT_ID=self.input_yaml['wcs']['TNS_BOT_ID'],
                        TNS_BOT_NAME=self.input_yaml['wcs']['TNS_BOT_NAME'],
                        TNS_BOT_API=self.input_yaml['wcs']['TNS_BOT_API']
                    )
                    
                    # Create a YAML file with the retrieved TNS information
                    autophot_yaml.create(transient_path, TNS_response)
                    
                    # Print a message confirming the TNS information was found
                    try:
                        print('\nFound TNS information for %s' % (TNS_response['name_prefix'] + TNS_response['objname']))
                    except KeyError:
                        print('\nFound TNS information for %s' % (TNS_response['objname']))
                except Exception as e:
                    # Handle and report any exceptions that occur
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname1, exc_tb.tb_lineno, e)
                    sys.exit("Can't reach Server - Check Internet Connection!")
    
        # If TNS information is not available and RA/DEC coordinates are provided
        elif self.input_yaml['target_ra'] is not None and self.input_yaml['target_dec'] is not None:
            TNS_response = {}
            TNS_response['ra'] = self.input_yaml['target_ra']
            TNS_response['dec'] = self.input_yaml['target_dec']
    
        else:
            # Prompt the user to continue if no TNS information and no RA/DEC coordinates are provided
            continue_response = (input('\n\nNo access to TNS and no RA/DEC given - do you wish to continue? [ y/[n] ]: ') or 'n')
            if continue_response != 'y':
                raise Exception('No target information given and user wants to quit')
            else:
                TNS_response = {}
    
        return TNS_response  # Return the dictionary with TNS or user-provided coordinates

    
    
    def check_filters(self, flist, available_filters):
        '''
        Checks and filters the list of image files based on their filter information.
        
        This function performs the following tasks:
        
        1. Loads telescope configuration from a YAML file.
        2. Iterates through the list of image files and checks their filter parameters.
        3. Compares these parameters with available filters and user-specified filters.
        4. Updates the list of files and filters based on the checks performed.
        
        :param flist: List of file paths to check
        :type flist: list of str
        :param available_filters: List of filters that are considered available
        :type available_filters: list of str
        :return: Tuple containing:
                 - Updated list of file paths that passed the filter check
                 - List of filters that were found to be available
        :rtype: tuple (list of str, list of str)
        '''
        
        import os
        import sys
        from functions import get_header, border_msg
        from functions import autophot_yaml, print_progress_bar
        
        # Initialize lists and counters
        filterUnavailable = []
        filterAvailable = []
        outFlist = []
        filesRemoved = 0
        filtersRemoved = 0
        filtersNotSelected = 0
        
        # Define keys for telescope and instrument
        teleKey = 'TELESCOP'
        instKey = 'INSTRUME'
        
        # Define key for filter information
        filterHeader = 'filter_key_0'
        
        # Define filter keys to avoid
        avoid_keys = ['clear', 'open']
        
        # Load telescope configuration from YAML file
        tele_autophot_input_yml = 'telescope.yml'
        tele_autophot_input = autophot_yaml(os.path.join(self.input_yaml['wdir'], tele_autophot_input_yml)).load()
        
        # Skip filter check if HST mode is enabled
        if self.input_yaml['HST_mode']:
            return flist, filterAvailable
        
        # Print initial message about the number of images being checked
        print(border_msg(f'Checking {len(flist)} images for correct filters parameters'))
        
        # Iterate through the list of files and process each one
        for name in print_progress_bar(flist):
            
            # Skip processing if template subtraction is enabled and 'PSF_model' is in the filename
            if self.input_yaml['template_subtraction']['prepare_templates'] and 'PSF_model' in name:
                return flist
            
            # Get header information from the file
            headinfo = get_header(name)
            
            # Retrieve telescope and instrument information from the header
            tele = headinfo[teleKey]
            inst = headinfo[instKey]
            
            # Get filter keys for the current telescope and instrument
            filter_keys = [i for i in list(tele_autophot_input[tele][instKey][inst]) if i.startswith('filter_key_')]
            
            # Initialize a flag to determine if the correct filter key is found
            found_correct_key = False
            
            for filter_header_key in filter_keys:
                # Check if the filter key exists in the header
                if tele_autophot_input[tele][instKey][inst][filter_header_key] not in list(headinfo.keys()):
                    continue
                
                # Check if the filter value should be avoided
                if headinfo[tele_autophot_input[tele][instKey][inst][filter_header_key]].lower() in avoid_keys:
                    open_filter = True
                    continue
                
                # Check if the filter value exists in the telescope configuration
                if headinfo[tele_autophot_input[tele][instKey][inst][filter_header_key]] in tele_autophot_input[tele][instKey][inst]:
                    found_correct_key = True
                    break
            
            # Try to retrieve the filter name from the header information
            try:
                fits_filter = headinfo[tele_autophot_input[tele][instKey][inst][filter_header_key]]
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname1, exc_tb.tb_lineno, e)
                print('***Filter header not found***')
                fits_filter = 'no_filter'
            
            # Try to retrieve the filter name from the configuration
            try:
                filter_name = tele_autophot_input[tele][instKey][inst][str(fits_filter)]
            except KeyError:
                filter_name = str(fits_filter)
            
            # Check if the filter is available and not to be removed
            if not filter_name in available_filters and not self.input_yaml['template_subtraction']['prepare_templates']:
                filesRemoved += 1
                filtersRemoved += 1
                filterUnavailable.append(filter_name)
                continue
            
            # Check if the filter should be selected based on user input
            if self.input_yaml['select_filter'] and not self.input_yaml['template_subtraction']['prepare_templates']:
                if str(tele_autophot_input[tele][instKey][inst][str(fits_filter)]) not in self.input_yaml['do_filter']:
                    filesRemoved += 1
                    filtersRemoved += 1
                    filterUnavailable.append(filter_name)
                    continue
            
            # Check if the filter is among the selected filters
            if self.input_yaml['selected_filters'][0] is not None:
                if filter_name not in self.input_yaml['selected_filters']:
                    filesRemoved += 1
                    filtersNotSelected += 1
                    continue
            
            # Add the file name and filter name to the respective lists if it passes all checks
            filterAvailable.append(filter_name)
            outFlist.append(name)
        
        # Print summaries of files removed and filters not selected
        if len(filterUnavailable) > 0:
            print(f'{len(filterUnavailable)} file(s) removed due to missing filters in available filters')
            
        if filtersNotSelected > 0:
            print(f'{filtersNotSelected} file(s) removed due to not selected by User')
        
        # Print the total number of files that passed the filter check
        print('Number of files: ', len(outFlist))
        
        return outFlist, filterAvailable  # Return the updated list of files and available filters

    
    
    
    def find_templates(self, requiredFilters=None):
        '''
        Searches for template files in the specified directory and returns a list of found template files.
        
        This function performs the following tasks:
        
        1. Determines the directory where template files should be located.
        2. Searches for templates based on the required filters.
        3. Checks the status of each template file and logs the findings.
        4. Prompts the user if there are issues with missing or multiple template files.
        
        :param requiredFilters: List of filters for which templates are required. If None, default filters from a YAML file are used.
        :type requiredFilters: list of str or None
        :return: List of file paths for the found template files
        :rtype: list of str
        '''
        
        import os
        import glob
        import shutil
        from functions import autophot_yaml, border_msg
        
        print(border_msg('Searching for template files'))
        
        fList = []  # Initialize list to store paths of found templates
        
        # Remove trailing slash from the directory path if present
        if self.input_yaml['fits_dir'].endswith('/'):
            self.input_yaml['fits_dir'] = self.input_yaml['fits_dir'][:-1]
        
        # Define the expected directory for template files
        templateDir = os.path.join(self.input_yaml['fits_dir'], 'templates')
        
        templateStatus = {}  # Dictionary to store the status and path of each template
        
        # If no required filters are provided, load default filters from a YAML file
        if not requiredFilters:
            baseFilepath = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/'))
            baseDatabase = os.path.join(baseFilepath, 'databases')
            
            filters_yml = 'filters.yml'
            requiredFilters = autophot_yaml(os.path.join(baseDatabase, filters_yml)).load()
            requiredFilters = requiredFilters['default_dmag'].keys()
        
        # Check if the template directory exists
        if not os.path.isdir(templateDir):
            print(f'No template folder found at expected location, skipping templates\nExpected position: {templateDir}')
            return []
        
        incorrectSetup = False  # Flag to track if any issues are detected
        
        # Iterate through each required filter
        for filter_x in requiredFilters:
            if filter_x in ['u', 'g', 'r', 'i', 'z']:
                filter_x += 'p'
            
            # Define the expected directory for the current filter's templates
            expectedTemplatedir = os.path.join(templateDir, f"{filter_x}_template")
            
            templateStatus[filter_x] = {}
            
            # Check if the directory for the current filter's templates exists
            if not os.path.isdir(expectedTemplatedir):
                templateStatus[filter_x]['status'] = 'directory does not exist'
                templateStatus[filter_x]['fpath'] = None
                continue
            
            # Search for template files and PSF model files in the directory
            templateFiles = glob.glob(os.path.join(expectedTemplatedir, '*.fits'))
            psfModelFiles = glob.glob(os.path.join(expectedTemplatedir, 'PSF_model_*'))
            
            # Exclude PSF model files from the list of template files
            templateFiles = list(set(templateFiles) - set(psfModelFiles))
            
            # Search for original files (possibly backups) and handle them
            originalFiles = glob.glob(os.path.join(expectedTemplatedir, '*.fits.original'))
            
            if len(originalFiles) == 1:
                # If exactly one original file is found, copy it to replace the .original extension
                templateStatus[filter_x]['status'] = 'found'
                shutil.copyfile(originalFiles[0], originalFiles[0].replace('.original', ''))
                templateStatus[filter_x]['fpath'] = originalFiles[0].replace('.original', '')
            
            elif len(templateFiles) == 0:
                # If no template files are found, log the status and continue
                templateStatus[filter_x]['status'] = 'not found'
                templateStatus[filter_x]['fpath'] = None
                incorrectSetup = True
            
            elif len(templateFiles) > 1:
                # If multiple template files are found, choose the largest one
                print(f'Multiple [{len(templateFiles)}] template files found for {filter_x}-band')
                incorrectSetup = True
                templateFileSize = {i: os.path.getsize(i) for i in templateFiles}
                bestGuess = max(templateFileSize, key=templateFileSize.get)
                templateStatus[filter_x]['status'] = 'multiple found'
                templateStatus[filter_x]['fpath'] = bestGuess
            
            else:
                # If exactly one template file is found, set its path
                templateStatus[filter_x]['status'] = 'found'
                templateStatus[filter_x]['fpath'] = templateFiles[0]
        
        # Print the status of each template found
        for key in templateStatus:
            print(f'[{key} template]')
            if templateStatus[key]['status'] == 'found':
                print(f'Filename: {templateStatus[key]["fpath"]}')
            elif templateStatus[key]['status'] in ['not found', 'directory does not exist']:
                print(f'Expected location: {os.path.join(expectedTemplatedir)}')
            elif templateStatus[key]['status'] == 'multiple found':
                print(f'Multiple files found, guessing this file\n->: {os.path.join(expectedTemplatedir)}')
        
        # Prompt the user if any issues were detected
        if incorrectSetup:
            quitAns = input('I have detected an incorrect setup with / missing the template files...\nDo you want to continue? [ y / [n] ] \n-> ' or 'n')
            if quitAns.lower() != 'y':
                raise Exception('Please check template directory and subdirectories and try again.')
        
        # Collect the paths of found template files
        fList = [templateStatus[i]['fpath'] for i in templateStatus]
        
        return fList

        
        
        
        