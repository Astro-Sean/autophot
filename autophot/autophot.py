#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:41:13 2022

@author: seanbrennan
"""



class automated_photometry():
    """
    A class to handle automated photometry tasks.
    """
    
    def __init__(self):
        """
        Initialize the AutomatedPhotometry instance.
        """
        pass

    def load():
        
        import os
        from functions import autophot_yaml, border_msg
        """
        Load default input settings from a YAML file.

        Returns:
            dict: The loaded configuration from the YAML file.
        """
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the name of the default input YAML file
        default_input_filename = 'default_input.yml'

        # Construct the full path to the default input YAML file
        default_input_filepath = os.path.join(script_dir, 'databases', default_input_filename)

        # Load the YAML file using the autophot_yaml function
        default_input_yaml = autophot_yaml(default_input_filepath, 'default_input').load()

        # Print a message indicating where the default input was loaded from
        border_msg(f'Default input loaded from: {default_input_filepath}', body='-', corner='+')

        return default_input_yaml

    # =============================================================================
    #     
    # =============================================================================
    

    def run_photometry(default_input, do_photometry=True):
        
        import os
        import copy
        import gc
        import numpy as np
        import yaml
        import subprocess
        import matplotlib.pyplot as plt
        from prepare import prepare
        from pathlib import Path
        from functions import concatenate_csv_files, print_progress_bar, border_msg
        """
        Perform photometry processing based on the provided input.
    
        Args:
            default_input (dict): Dictionary containing various configuration parameters.
            do_photometry (bool): Flag indicating whether to perform photometry or not.
    
        Returns:
            str: Path to the output lightcurve CSV file.
        """
        # Clean up any unused memory
        gc.collect()
        
        # Ensure 'fits_dir' does not end with a trailing slash
        if default_input['fits_dir'].endswith('/'):
            default_input['fits_dir'] = default_input['fits_dir'][:-1]
        
        # Close all matplotlib plots
        plt.close('all')
        
        
        if default_input['catalog']['build_catalog']:
            default_input['catalog']['use_catalog'] = 'custom'
    
        # Initialize the prepare object
        prepare_database = prepare(default_input=default_input)
    
        # Check catalog information
        available_filters = prepare_database.check_catalog()
    
        # Check TNS for coordinates and update input dictionary
        try:
            TNS_coords = prepare_database.check_TNS()
            default_input.update({
                'target_ra': TNS_coords['radeg'],
                'target_dec': TNS_coords['decdeg'],
                'name_prefix': TNS_coords['name_prefix'],
                'objname': TNS_coords['objname']
            })
        except: pass
    
        # Define paths and executable
        current_directory = os.path.dirname(os.path.abspath(__file__))
        autophot_exe = os.path.join(current_directory, 'main.py')
    
        if do_photometry:
            # Clean and check files
            flist = prepare_database.clean()
            if not default_input.get('skip_file_check', False):
                flist = prepare_database.check_files(flist=flist)
                flist, required_filters = prepare_database.check_filters(flist=flist, available_filters=available_filters)
            else:
                required_filters = available_filters
    
            required_filters = list(set(required_filters))
            
            # Download Pan-STARRS templates if needed
            if default_input['template_subtraction']['get_panstarrs_templates'] and default_input['template_subtraction']['do_subtraction']:
                from templates import download_panstarrs_template
    
                template_folder = os.path.join(default_input['fits_dir'], 'templates')
                for f in required_filters:
                    download_panstarrs_template(
                        ra=default_input['target_ra'],
                        dec=default_input['target_dec'],
                        size=default_input['template_subtraction']['panstarrs_templates_size'],
                        template_folder=template_folder,
                        f=f
                    )
            
            # Prepare template files if needed
            template_flist = []
            if default_input['template_subtraction']['do_subtraction']:
                template_flist = prepare_database.find_templates(requiredFilters=required_filters)
                template_flist = prepare_database.check_files(flist=template_flist)
                template_flist, _ = prepare_database.check_filters(flist=template_flist, available_filters=available_filters)
            
            # Backup input configuration to YAML
            backup_yaml = copy.deepcopy(default_input)
            wdir = backup_yaml['fits_dir']
            newDir = '_' + backup_yaml['outdir_name']
            baseDir = os.path.basename(wdir)
            workLoc = baseDir + newDir
            
            
            new_output_dir = os.path.join(os.path.dirname(wdir), workLoc)
            
            Path(new_output_dir).mkdir(parents=True, exist_ok=True)
            
            input_file = os.path.join(new_output_dir, 'input.yaml')
            with open(input_file, 'w') as file:
                yaml.dump(backup_yaml, file, default_flow_style=False)
            
            # Process template files
            if template_flist:
                print(border_msg('Reducing and calibrating template files'))
                for template in print_progress_bar(template_flist, title='Template files calibrated'):
                    args = f'python {autophot_exe} -f {template} -c {input_file} -temp'
                    subprocess.run(args, shell=True, check=True, text=True)
                            
                    gc.collect()
    
            # Process science files
            print(border_msg('Reducing and calibrating science files'))
            if flist:
                flist = np.sort(flist)[::-1]  # Sort in descending order
                for file in print_progress_bar(flist, title='Science files calibrated'):
                    args = f'python {autophot_exe} -f {file} -c {input_file}'
                    subprocess.run(args, shell=True, check=True, text=True)
                            
                    gc.collect()
            
        # Concatenate CSV files
        reduced_loc = f"{default_input['fits_dir']}_{default_input['outdir_name']}"
        output_loc = os.path.join(reduced_loc, 'lightcurve_output.csv')
        concatenate_csv_files(
            folder_path=reduced_loc,
            output_filename=output_loc,
            loc_file='output.csv'
        )
        return output_loc

    
    def create_lightcurve():
        
        
        return
    
    def create_latex():
        
        
        return