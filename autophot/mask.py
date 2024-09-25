#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:01:06 2024

@author: seanbrennan
"""


# =============================================================================
#     
# =============================================================================

def append_id(filename):
    return "{0}{2}.{1}".format(*filename.rsplit('.', 1) + ['.mask'])

def is_program_installed(program_name):
    import subprocess
    try:
        # Run 'which' command to check if the program is installed
        result = subprocess.run(['which', program_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # If the command is found, it returns a path, otherwise the output is empty
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
# =============================================================================
#     
# =============================================================================
def run_maximask_with_file(filename):
    
    import os,subprocess
    program_name = 'maximask'
    if is_program_installed(program_name):
        try:
            # Run maximask with the given filename and suppress output
            subprocess.run([program_name, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            
            expected_fname =  append_id(filename)
            
            if os.path.exists(expected_fname):
                print(f"\nImage mask created using {program_name} ")
                
                return expected_fname
                
            else:
                raise Exception('maximask output not created')
                return None

        except Exception as e:
            print(f"\nAn error occurred while running {program_name}: {e}")
            
    else:
        print(f"\n{program_name} is not installed. Cannot run the command.")
        return None
    
# =============================================================================
#     
# =============================================================================
    
def create_image_mask(filename, layers = ['CR']):
    import numpy as np
    import os
    from astropy.io import fits
    from functions import border_msg
    
    if (filename is None):
        return None
    
    fname = os.path.basename(filename)
    
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(border_msg(f'Creating mask for {fname}'))
    
    maximask_masks = {'CR': [0,'Cosmic Rays'],
                     'HCL': [1,'Hot Columns/Lines'],
                     'DCL': [2,'Dead Columns/Lines/Clusters'],
                     'BG': [3,'Background'],
                     'DP': [4,'Dead Pixels'],
                     'P': [5,'Persistence'],
                     'TRL': [6,'TRaiLs'],
                     'FR': [7,'FRinge patterns'],
                    'NEB': [8,'NEBulosities'],
                     'SAT': [9,'SATurated pixels'],
                     'SP': [10,'diffraction SPikes'],
                     'OV': [11,'OVerscanned pixels'],
                    'BBG': [12, 'Bright BackGround pixel']}
    
    
    with fits.open(filename) as hdul:
        # Iterate through each HDU (Header Data Unit)
        imgs = hdul[0].data
        
    selected_layers = [maximask_masks[i] for i in layers]
    
    mask = np.zeros(imgs[0].shape)
    
    for layer in selected_layers:
        print(f'\t> Adding {layer[1]} layer to mask')
        mask+=imgs[layer[0]]
        
    mask[mask>1] = 1
    return mask.astype(int)
    
    
    
    
    
# =============================================================================
#         
# =============================================================================
    




