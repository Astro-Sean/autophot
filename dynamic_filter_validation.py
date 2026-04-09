#!/usr/bin/env python3
"""
Dynamic filter validation utilities for AutoPHOT.

This module provides functions to validate transmission curves and support
arbitrary filter names with custom catalogs or Gaia transmission curves.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_transmission_curve(curve_path: str, filter_name: str) -> Tuple[bool, str]:
    """
    Validate a transmission curve file for a given filter.
    
    Parameters
    ----------
    curve_path : str
        Path to the transmission curve file
    filter_name : str
        Name of the filter (for logging)
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not os.path.exists(curve_path):
        return False, f"Transmission curve file not found: {curve_path}"
    
    try:
        # Try to read the transmission curve
        data = pd.read_csv(curve_path, comment='#', delim_whitespace=True, header=None)
        
        if data.shape[1] < 2:
            return False, f"Transmission curve must have at least 2 columns (wavelength, throughput): {curve_path}"
        
        wavelength = data.iloc[:, 0].values
        throughput = data.iloc[:, 1].values
        
        # Validate wavelength range (should be in nm, reasonable range 100-5000 nm)
        if np.min(wavelength) < 100 or np.max(wavelength) > 5000:
            return False, f"Wavelength range {np.min(wavelength)}-{np.max(wavelength)} nm seems unreasonable for filter {filter_name}"
        
        # Validate throughput values (should be between 0 and 1)
        if np.min(throughput) < 0 or np.max(throughput) > 1:
            return False, f"Throughput values should be between 0 and 1 for filter {filter_name}"
        
        # Check if there's any non-zero throughput
        if np.max(throughput) <= 0:
            return False, f"No non-zero throughput found for filter {filter_name}"
        
        # Calculate effective wavelength for validation
        effective_wavelength = np.trapz(wavelength * throughput, wavelength) / np.trapz(throughput, wavelength)
        logger.info(f"Filter {filter_name}: Effective wavelength = {effective_wavelength:.0f} Å")
        
        return True, ""
        
    except Exception as e:
        return False, f"Error reading transmission curve for {filter_name}: {str(e)}"

def discover_filters_from_transmission_curves(curve_map: Dict[str, str]) -> Dict[str, str]:
    """
    Discover filters from transmission curve map and validate them.
    
    Parameters
    ----------
    curve_map : Dict[str, str]
        Mapping of filter names to transmission curve file paths
        
    Returns
    -------
    Dict[str, str]
        Mapping of valid filter names to their curve paths
    """
    valid_filters = {}
    
    for filter_name, curve_path in curve_map.items():
        is_valid, error_msg = validate_transmission_curve(curve_path, filter_name)
        
        if is_valid:
            valid_filters[filter_name] = curve_path
            logger.info(f"Validated transmission curve for filter {filter_name}")
        else:
            logger.warning(f"Invalid transmission curve for filter {filter_name}: {error_msg}")
    
    return valid_filters

def create_dynamic_filter_config(
    custom_catalog_path: str,
    transmission_curve_map: Optional[Dict[str, str]] = None,
    output_config_path: str = "dynamic_filters_config.yml"
) -> Dict:
    """
    Create a dynamic filter configuration based on custom catalog and transmission curves.
    
    Parameters
    ----------
    custom_catalog_path : str
        Path to custom catalog CSV file
    transmission_curve_map : Optional[Dict[str, str]]
        Mapping of filter names to transmission curve files
    output_config_path : str
        Path to write the configuration file
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    # Read custom catalog
    try:
        catalog_df = pd.read_csv(custom_catalog_path)
    except Exception as e:
        logger.error(f"Could not read custom catalog {custom_catalog_path}: {e}")
        return {}
    
    # Discover filters from catalog
    discovered_filters = []
    for col in catalog_df.columns:
        col_str = str(col).strip()
        
        # Skip non-photometric columns
        if col_str.lower() in ['ra', 'dec', 'name', 'objname', 'id', 'x', 'y']:
            continue
        # Skip error columns
        if col_str.endswith('_err') or col_str.endswith('err'):
            continue
        
        # Check if column has numeric data
        try:
            numeric_data = pd.to_numeric(catalog_df[col_str], errors='coerce')
            if numeric_data.notna().sum() > len(catalog_df) * 0.1:  # At least 10% valid data
                discovered_filters.append(col_str)
        except Exception:
            continue
    
    # Validate transmission curves if provided
    valid_curves = {}
    if transmission_curve_map:
        valid_curves = discover_filters_from_transmission_curves(transmission_curve_map)
    
    # Create configuration
    config = {
        'catalog': {
            'use_catalog': 'custom',
            'catalog_custom_fpath': custom_catalog_path,
            'discovered_filters': discovered_filters
        }
    }
    
    # Add Gaia custom configuration if transmission curves are valid
    if valid_curves:
        config['catalog']['use_catalog'] = 'gaia_custom'
        config['catalog']['transmission_curve_map'] = valid_curves
        config['catalog']['gaia_curve_map_max_sources'] = 100
        config['catalog']['gaia_curve_map_order_by'] = 'distance'
    
    # Write configuration file
    import yaml
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Dynamic filter configuration written to {output_config_path}")
    logger.info(f"Discovered {len(discovered_filters)} filters: {', '.join(discovered_filters)}")
    
    if valid_curves:
        logger.info(f"Validated {len(valid_curves)} transmission curves: {', '.join(valid_curves.keys())}")
    
    return config

def generate_example_transmission_curves(output_dir: str = "example_transmission_curves"):
    """
    Generate example transmission curve files for common non-standard filters.
    
    Parameters
    ----------
    output_dir : str
        Directory to write the example files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Example transmission curves for common filters
    example_curves = {
        'Y_band.dat': {
            'wavelength': np.linspace(900, 1100, 100),  # nm
            'throughput': np.exp(-((np.linspace(900, 1100, 100) - 1000) / 50) ** 2)
        },
        'w_band.dat': {
            'wavelength': np.linspace(400, 800, 100),  # nm  
            'throughput': np.exp(-((np.linspace(400, 800, 100) - 600) / 100) ** 2)
        },
        'custom_NIR.dat': {
            'wavelength': np.linspace(1000, 1300, 100),  # nm
            'throughput': np.exp(-((np.linspace(1000, 1300, 100) - 1150) / 75) ** 2)
        }
    }
    
    for filename, data in example_curves.items():
        output_path = os.path.join(output_dir, filename)
        df = pd.DataFrame({
            'wavelength_nm': data['wavelength'],
            'throughput': data['throughput']
        })
        df.to_csv(output_path, index=False, sep=' ', header=False)
        logger.info(f"Generated example transmission curve: {output_path}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate example transmission curves
    generate_example_transmission_curves()
    
    # Example: Create dynamic configuration from a custom catalog
    # config = create_dynamic_filter_config(
    #     "my_custom_catalog.csv",
    #     {"Y": "transmission_curves/Y_band.dat", "w": "transmission_curves/w_band.dat"}
    # )
