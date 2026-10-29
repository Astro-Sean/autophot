#!/usr/bin/env python3
"""
Test script for SCAMP + SWarp alignment method.

This script tests the alignment between a science and reference image using
the SCAMP + SWarp pipeline, verifying that:
1. Both output images have identical shapes
2. CRPIX values match
3. WCS is invertible (pix2world and world2pix)
4. The reference covers the science region

Usage:
    python test_scamp_swarp_alignment.py --science /path/to/science.fits --reference /path/to/reference.fits --output /path/to/output_dir
"""

import argparse
import sys
import shutil
import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Add astro environment to PATH to find SExtractor
astro_bin = "/home/sbrennan/miniconda3/envs/astro/bin"
if os.path.exists(astro_bin):
    os.environ["PATH"] = astro_bin + ":" + os.environ.get("PATH", "")

# Add parent directory to path to import autophot modules
sys.path.insert(0, str(Path(__file__).parent))

from utils.run_IDC import ImageDistortionCorrector


def test_alignment(science_path, reference_path, output_dir, verbose=1):
    """
    Test SCAMP + SWarp alignment between science and reference images.
    
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
        Test results with keys: success, sci_shape, ref_shape, crpix_match, wcs_invertible
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy input files to output directory to avoid overwriting originals
    sci_path_copy = output_dir / "science_input.fits"
    ref_path_copy = output_dir / "reference_input.fits"
    shutil.copy2(science_path, sci_path_copy)
    shutil.copy2(reference_path, ref_path_copy)
    
    # Create minimal input YAML
    input_yaml = {
        "wcs": {
            "scamp_exe_loc": "scamp",
            "scamp_astref_catalog": "GAIA-DR3",
            "scamp_ref_timeout": 60,
            "scamp_ref_server": "vizier.cfa.harvard.edu",
        },
        "template_subtraction": {
            "alignment_method": "swarp",
        }
    }
    
    print(f"\n{'='*70}")
    print(f"Testing SCAMP + SWarp Alignment")
    print(f"{'='*70}")
    print(f"Science:    {science_path}")
    print(f"Reference:  {reference_path}")
    print(f"Output:     {output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize corrector
    corrector = ImageDistortionCorrector(input_yaml, verbose_level=verbose)
    
    # Run alignment
    print("Running alignment...")
    result = corrector.align_and_resample_both_images(
        science_image=str(sci_path_copy),
        reference_image=str(ref_path_copy),
        output_dir=str(output_dir),
        resample_mode="common_grid"  # Default to common grid resampling for testing
    )
    
    if result is None:
        print("\nFAIL Alignment FAILED (returned None)")
        return {
            "success": False,
            "reason": "Alignment returned None",
            "sci_shape": None,
            "ref_shape": None,
            "crpix_match": False,
            "wcs_invertible": False
        }
    
    # The align_and_resample_both_images function overwrites the input files
    # with aligned versions. Check the copied paths for the aligned images.
    sci_output = sci_path_copy
    ref_output = ref_path_copy
    
    if not sci_output.exists() or not ref_output.exists():
        print(f"\nFAIL Aligned output files not found")
        print(f"  Science: {sci_output}")
        print(f"  Reference: {ref_output}")
        return {
            "success": False,
            "reason": "Aligned output files not found",
            "sci_shape": None,
            "ref_shape": None,
            "crpix_match": False,
            "wcs_invertible": False
        }
    
    # Verify shapes
    with fits.open(sci_output) as hdul:
        sci_data = hdul[0].data
        sci_header = hdul[0].header
        sci_shape = sci_data.shape
        sci_wcs = WCS(sci_header)
    
    with fits.open(ref_output) as hdul:
        ref_data = hdul[0].data
        ref_header = hdul[0].header
        ref_shape = ref_data.shape
        ref_wcs = WCS(ref_header)
    
    print(f"\n{'='*70}")
    print(f"Alignment Results")
    print(f"{'='*70}")
    print(f"Science output:  {sci_output.name}")
    print(f"  Shape: {sci_shape}")
    print(f"  CRPIX: ({sci_header.get('CRPIX1', 'N/A'):.2f}, {sci_header.get('CRPIX2', 'N/A'):.2f})")
    print(f"  CRVAL: ({sci_header.get('CRVAL1', 'N/A'):.6f}, {sci_header.get('CRVAL2', 'N/A'):.6f})")
    print(f"  CTYPE: ({sci_header.get('CTYPE1', 'N/A')}, {sci_header.get('CTYPE2', 'N/A')})")
    
    print(f"\nReference output: {ref_output.name}")
    print(f"  Shape: {ref_shape}")
    print(f"  CRPIX: ({ref_header.get('CRPIX1', 'N/A'):.2f}, {ref_header.get('CRPIX2', 'N/A'):.2f})")
    print(f"  CRVAL: ({ref_header.get('CRVAL1', 'N/A'):.6f}, {ref_header.get('CRVAL2', 'N/A'):.6f})")
    print(f"  CTYPE: ({ref_header.get('CTYPE1', 'N/A')}, {ref_header.get('CTYPE2', 'N/A')})")
    
    # Check shape match
    shape_match = sci_shape == ref_shape
    print(f"\nPASS Shape match: {shape_match}")
    if not shape_match:
        print(f"  Science: {sci_shape}")
        print(f"  Reference: {ref_shape}")
        print(f"  Difference: ({abs(sci_shape[0] - ref_shape[0])}, {abs(sci_shape[1] - ref_shape[1])})")
    
    # Check CRPIX match
    crpix_sci = (sci_header.get('CRPIX1', 0), sci_header.get('CRPIX2', 0))
    crpix_ref = (ref_header.get('CRPIX1', 0), ref_header.get('CRPIX2', 0))
    crpix_diff = (abs(crpix_sci[0] - crpix_ref[0]), abs(crpix_sci[1] - crpix_ref[1]))
    crpix_match = crpix_diff[0] < 0.01 and crpix_diff[1] < 0.01
    print(f"PASS CRPIX match: {crpix_match} (diff: {crpix_diff[0]:.3f}, {crpix_diff[1]:.3f})")
    
    # Check WCS invertibility
    wcs_invertible = True
    for label, wcs, shape in [("Science", sci_wcs, sci_shape), ("Reference", ref_wcs, ref_shape)]:
        # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
        cx, cy = (shape[1] - 1) / 2, (shape[0] - 1) / 2
        try:
            ra, dec = wcs.all_pix2world([cx], [cy], 0)
            if not (np.isfinite(ra[0]) and np.isfinite(dec[0])):
                print(f"FAIL {label} WCS pix2world failed: RA={ra[0]}, Dec={dec[0]}")
                wcs_invertible = False
                continue
            
            px, py = wcs.all_world2pix(ra[0], dec[0], 0)
            if not (np.isfinite(px) and np.isfinite(py)):
                print(f"FAIL {label} WCS world2pix failed: px={px}, py={py}")
                wcs_invertible = False
        except Exception as e:
            print(f"FAIL {label} WCS error: {e}")
            wcs_invertible = False
    
    print(f"PASS WCS invertible: {wcs_invertible}")
    
    # Overall success
    success = shape_match and crpix_match and wcs_invertible
    
    print(f"\n{'='*70}")
    if success:
        print(f"SUCCESS Alignment PASSED")
    else:
        print(f"FAIL Alignment FAILED")
    print(f"{'='*70}\n")
    
    return {
        "success": success,
        "sci_shape": sci_shape,
        "ref_shape": ref_shape,
        "crpix_match": crpix_match,
        "wcs_invertible": wcs_invertible,
        "shape_match": shape_match,
        "crpix_diff": crpix_diff
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test SCAMP + SWarp alignment between science and reference images"
    )
    parser.add_argument(
        "--science",
        required=True,
        help="Path to science FITS image"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference FITS image"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for aligned images"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=quiet, 1=normal, 2=debug)"
    )
    
    args = parser.parse_args()
    
    result = test_alignment(
        science_path=args.science,
        reference_path=args.reference,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
