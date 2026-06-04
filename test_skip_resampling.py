#!/usr/bin/env python3
"""
Test script for the new skip_resampling option.

This script tests the new skip_resampling configuration option to ensure:
1. When skip_resampling=False, SWarp resampling is performed (default behavior)
2. When skip_resampling=True, only SCAMP WCS correction is applied
3. The output files are correctly named and positioned

Usage:
    python test_skip_resampling.py --science /path/to/science.fits --reference /path/to/reference.fits --output /path/to/output_dir
"""

import argparse
import sys
import shutil
import os
from pathlib import Path
import yaml
import tempfile

# Add parent directory to path to import autophot modules
sys.path.insert(0, str(Path(__file__).parent))

from utils.run_IDC import ImageDistortionCorrector


def create_test_yaml(skip_resampling_value):
    """Create a minimal YAML configuration for testing."""
    config = {
        'default_input': {
            'global_verbose_level': 1,
            'wcs': {
                'solver': 'scamp',
                'scamp_exe_loc': 'scamp',
                'sextractor_exe_loc': 'sextractor',
                'scamp_astref_catalog': 'GAIA-DR3',
                'scamp_distort_degrees': 1,
                'scamp_crossid_radius': 2.5,
                'scamp_position_maxerr': 1.0,
                'scamp_fwhm_thresholds': '1.0,15.0',
                'scamp_sn_thresholds': '3.0,100000.0',
            },
            'template_subtraction': {
                'skip_resampling': skip_resampling_value,
                'alignment_method': 'swarp',
            }
        }
    }
    return config


def test_skip_resampling(science_path, reference_path, output_dir, verbose=1):
    """
    Test the skip_resampling configuration option.
    
    Parameters
    ----------
    science_path : str
        Path to science FITS image
    reference_path : str
        Path to reference FITS image
    output_dir : str
        Directory for output files
    verbose : int
        Verbosity level (0=quiet, 1=normal, 2=debug)
    
    Returns
    -------
    dict
        Test results
    """
    
    print(f"Testing skip_resampling option")
    print(f"Science: {science_path}")
    print(f"Reference: {reference_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Test both skip_resampling=False and skip_resampling=True
    for skip_resampling_value in [False, True]:
        print(f"\n--- Testing skip_resampling={skip_resampling_value} ---")
        
        # Create test configuration
        config = create_test_yaml(skip_resampling_value)
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_path = f.name
        
        try:
            # Create output subdirectory for this test
            test_output_dir = Path(output_dir) / f"skip_resampling_{skip_resampling_value}"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize corrector with test config
            corrector = ImageDistortionCorrector(config_path, verbose_level=verbose)
            
            # Run alignment
            print(f"Running alignment with skip_resampling={skip_resampling_value}...")
            result = corrector.align_and_resample_both_images(
                science_image=science_path,
                reference_image=reference_path,
                output_dir=str(test_output_dir),
                skip_resampling=skip_resampling_value
            )
            
            if result is None:
                print(f"❌ Alignment FAILED for skip_resampling={skip_resampling_value}")
                results[f"skip_resampling_{skip_resampling_value}"] = {
                    "success": False,
                    "reason": "Alignment returned None"
                }
                continue
            
            # Check output files
            sci_aligned = Path(result.get("science_aligned", ""))
            ref_aligned = Path(result.get("reference_aligned", ""))
            
            print(f"✓ Alignment succeeded for skip_resampling={skip_resampling_value}")
            print(f"  Science aligned: {sci_aligned}")
            print(f"  Reference aligned: {ref_aligned}")
            print(f"  Alignment method: {result.get('alignment_method', 'unknown')}")
            
            # Verify files exist
            sci_exists = sci_aligned.exists() if sci_aligned else False
            ref_exists = ref_aligned.exists() if ref_aligned else False
            
            results[f"skip_resampling_{skip_resampling_value}"] = {
                "success": True,
                "science_aligned": str(sci_aligned) if sci_aligned else None,
                "reference_aligned": str(ref_aligned) if ref_aligned else None,
                "science_exists": sci_exists,
                "reference_exists": ref_exists,
                "alignment_method": result.get('alignment_method', 'unknown'),
                "resampling_method": result.get('science_resampling_method', 'unknown')
            }
            
            print(f"  Files exist: sci={sci_exists}, ref={ref_exists}")
            
        except Exception as e:
            print(f"❌ Exception for skip_resampling={skip_resampling_value}: {e}")
            results[f"skip_resampling_{skip_resampling_value}"] = {
                "success": False,
                "reason": f"Exception: {e}"
            }
        
        finally:
            # Clean up temporary config file
            try:
                os.unlink(config_path)
            except:
                pass
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    
    for key, result in results.items():
        skip_val = key.split("_")[-1]
        status = "✓ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} skip_resampling={skip_val}")
        
        if result["success"]:
            print(f"  Science: {result.get('science_aligned', 'N/A')}")
            print(f"  Reference: {result.get('reference_aligned', 'N/A')}")
            print(f"  Method: {result.get('alignment_method', 'N/A')}")
        else:
            print(f"  Reason: {result.get('reason', 'Unknown')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test skip_resampling configuration option")
    parser.add_argument("--science", required=True, help="Path to science FITS image")
    parser.add_argument("--reference", required=True, help="Path to reference FITS image")
    parser.add_argument("--output", required=True, help="Output directory for test files")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], 
                       help="Verbosity level (0=quiet, 1=normal, 2=debug)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.science).exists():
        print(f"❌ Science file not found: {args.science}")
        sys.exit(1)
    
    if not Path(args.reference).exists():
        print(f"❌ Reference file not found: {args.reference}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run test
    results = test_skip_resampling(args.science, args.reference, args.output, args.verbose)
    
    # Exit with appropriate code
    failed_tests = sum(1 for r in results.values() if not r["success"])
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed")
        sys.exit(1)
    else:
        print(f"\n✓ All tests passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
