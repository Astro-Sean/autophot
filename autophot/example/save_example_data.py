#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from pathlib import Path

from shutil import copyfile
def save_fits_to_desktop(new_path = os.path.join(str(Path.home()),'Desktop'),
                         template_subtraction_example = False,
                         return_custom_catalog = False):

    # parent directory
    filepath = os.path.dirname(os.path.abspath(__file__))
    
    tutorial_dir = os.path.join(filepath,'tutorial_data')
    
    if template_subtraction_example:
        dir_name = 'autophot_host_subtraction_example'
    else:
        dir_name = 'autophot_example'
    
    # create folder on dekstop called autophot example
    example_directory_new_path = os.path.join(new_path,dir_name)
    
    # create directory on dekstop if not already created
    os.makedirs(example_directory_new_path , exist_ok=True)

    # default name - don't fix
    if not template_subtraction_example:
        
        example_fits_name = 'example.fits'
    
        # Location of example.fits
        example_fits_path = os.path.join(tutorial_dir,example_fits_name)
        
        # copy example.fits to desktop
        copyfile(example_fits_path,
                 os.path.join(example_directory_new_path,example_fits_name))
        
    else:
        example_fits_name = 'transient_with_host.fits'
        template_fits_name = 'template_with_host.fits'
        
           
        # Location of example.fits
        example_fits_path = os.path.join(tutorial_dir,example_fits_name)
        template_fits_path = os.path.join(tutorial_dir,template_fits_name)
                                                  
        template_dir = os.path.join(example_directory_new_path,'templates')
        # create directory on dekstop if not already created
        os.makedirs(template_dir , exist_ok=True)
        
        # copy example.fits to desktop
        copyfile(example_fits_path,os.path.join(example_directory_new_path,example_fits_name))
        
        # List of filters we want to make folders for 
        filter_list = ['up','gp','rp','ip','zp','U','B','V','R','I','J','H','K']
        
        for f in filter_list:
            
            
            filter_template_name = f + '_template'
            
            filter_template_location = os.path.join(template_dir,filter_template_name)
            
            # print('Creating folder for %s-band at: %s' % (f,filter_template_location))
            
            if not os.path.exists(filter_template_location): # if the directory does not exis
                
                os.mkdir(filter_template_location)
              
        g_template_folder = os.path.join(template_dir, 'gp_template')
        
        copyfile(template_fits_path,os.path.join(g_template_folder,template_fits_name))
        
    # Check that file is written to correct place
    if os.path.isfile(os.path.join(example_directory_new_path,example_fits_name)):
        print('Successful copy of %s written to: %s' % (example_fits_name,os.path.join(example_directory_new_path,example_fits_name)))
        
    if template_subtraction_example:
        if os.path.isfile(os.path.join(g_template_folder,template_fits_name)):
            print('\nSuccessful copy of %s written to: %s' % (template_fits_name,g_template_folder))
    
    if return_custom_catalog:
       
        
        
        copyfile(os.path.join(tutorial_dir,'my_first_catalog.csv'),os.path.join(example_directory_new_path,'my_first_catalog.csv'))
        
        print('\nReturning custom catalog: %s' % os.path.join(example_directory_new_path,'my_first_catalog.csv'))
        return os.path.join(example_directory_new_path,example_fits_name),os.path.join(example_directory_new_path,'my_first_catalog.csv')
        
    else:
        return os.path.join(example_directory_new_path,example_fits_name)
