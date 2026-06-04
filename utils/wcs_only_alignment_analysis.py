#!/usr/bin/env python3
"""
Analysis of WCS-only alignment vs resampling requirements for astronomical images.
Determines whether images can be aligned without resampling to preserve data quality.
"""

import numpy as np
import logging
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

class AlignmentMethodAnalyzer:
    """Analyze whether WCS-only alignment can replace resampling for specific use cases."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_alignment_requirements(self, sci_image_path: str, ref_image_path: str) -> dict:
        """
        Analyze whether images can be aligned without resampling.
        
        Returns:
            Dictionary with alignment analysis and recommendations
        """
        
        analysis = {
            'wcs_only_possible': False,
            'resampling_required': False,
            'reasons': [],
            'recommendations': [],
            'use_case_analysis': {}
        }
        
        try:
            # Load image headers and WCS
            with fits.open(sci_image_path) as sci_hdul, fits.open(ref_image_path) as ref_hdul:
                sci_header = sci_hdul[0].header
                ref_header = ref_hdul[0].header
                sci_data = sci_hdul[0].data
                ref_data = ref_hdul[0].data
                
                sci_wcs = WCS(sci_header)
                ref_wcs = WCS(ref_header)
                
                # 1. Check fundamental alignment requirements
                self._analyze_fundamental_requirements(sci_wcs, ref_wcs, sci_header, ref_header, analysis)
                
                # 2. Analyze specific use cases
                self._analyze_use_cases(sci_wcs, ref_wcs, sci_data.shape, ref_data.shape, analysis)
                
                # 3. Check distortion handling
                self._analyze_distortion_requirements(sci_header, ref_header, analysis)
                
                # 4. Evaluate pixel scale and grid alignment
                self._analyze_pixel_grid_alignment(sci_wcs, ref_wcs, sci_data.shape, ref_data.shape, analysis)
                
                # 5. Make final recommendation
                self._make_recommendation(analysis)
                
        except Exception as e:
            self.logger.error(f"Alignment analysis failed: {e}")
            analysis['error'] = str(e)
            
        return analysis
    
    def _analyze_fundamental_requirements(self, sci_wcs, ref_wcs, sci_header, ref_header, analysis):
        """Analyze fundamental requirements for alignment."""
        
        # Check if both images have valid WCS
        if sci_wcs.is_celestial and ref_wcs.is_celestial:
            analysis['reasons'].append("✓ Both images have valid celestial WCS")
        else:
            analysis['reasons'].append("✗ One or both images lack valid celestial WCS")
            analysis['resampling_required'] = True
        
        # Check if WCS are analytically invertible
        try:
            # Test coordinate transformations
            # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
            center_x = (sci_header.get('NAXIS1', 1000) - 1) / 2
            center_y = (sci_header.get('NAXIS2', 1000) - 1) / 2
            sci_world = sci_wcs.all_pix2world([[center_x]], [[center_y]], 0)
            sci_back = sci_wcs.all_world2pix([sci_world[0, 0]], [sci_world[1, 0]], 0)
            
            if np.isfinite(sci_back[0]) and np.isfinite(sci_back[1]):
                analysis['reasons'].append("✓ Science WCS is analytically invertible")
            else:
                analysis['reasons'].append("✗ Science WCS is not fully invertible")
                analysis['resampling_required'] = True
                
        except Exception as e:
            analysis['reasons'].append(f"✗ Science WCS transformation failed: {e}")
            analysis['resampling_required'] = True
        
        try:
            # Correct numpy 0-based center is (nx-1)/2, (ny-1)/2.
            center_x = (ref_header.get('NAXIS1', 1000) - 1) / 2
            center_y = (ref_header.get('NAXIS2', 1000) - 1) / 2
            ref_world = ref_wcs.all_pix2world([[center_x]], [[center_y]], 0)
            ref_back = ref_wcs.all_world2pix([ref_world[0, 0]], [ref_world[1, 0]], 0)
            
            if np.isfinite(ref_back[0]) and np.isfinite(ref_back[1]):
                analysis['reasons'].append("✓ Reference WCS is analytically invertible")
            else:
                analysis['reasons'].append("✗ Reference WCS is not fully invertible")
                analysis['resampling_required'] = True
                
        except Exception as e:
            analysis['reasons'].append(f"✗ Reference WCS transformation failed: {e}")
            analysis['resampling_required'] = True
    
    def _analyze_use_cases(self, sci_wcs, ref_wcs, sci_shape, ref_shape, analysis):
        """Analyze specific astronomical use cases."""
        
        use_cases = {
            'photometry': {
                'wcs_only_sufficient': False,
                'requirements': ['Identical pixel grids', 'Same pixel scale', 'Precise coordinate alignment'],
                'analysis': ''
            },
            'source_detection': {
                'wcs_only_sufficient': True,
                'requirements': ['Valid WCS transformations'],
                'analysis': ''
            },
            'astrometry': {
                'wcs_only_sufficient': True,
                'requirements': ['Accurate WCS', 'Coordinate transformations'],
                'analysis': ''
            },
            'image_subtraction': {
                'wcs_only_sufficient': False,
                'requirements': ['Identical pixel grids', 'Same PSF sampling', 'Precise alignment'],
                'analysis': ''
            },
            'visualization': {
                'wcs_only_sufficient': True,
                'requirements': ['Coordinate transformation for overlay'],
                'analysis': ''
            }
        }
        
        # Photometry analysis
        sci_pixscale = self._get_pixel_scale(sci_wcs)
        ref_pixscale = self._get_pixel_scale(ref_wcs)
        
        if (sci_pixscale and ref_pixscale and abs(sci_pixscale - ref_pixscale) < 0.001 and
            sci_shape == ref_shape):
            use_cases['photometry']['wcs_only_sufficient'] = True
            use_cases['photometry']['analysis'] = "Same pixel scale and shape - WCS-only may work"
        else:
            use_cases['photometry']['analysis'] = f"Different scales/shapes: sci={sci_pixscale:.4f}\" ref={ref_pixscale:.4f}\" shapes={sci_shape} vs {ref_shape}"
        
        # Image subtraction analysis - UPDATED for template subtraction
        # Template subtraction algorithms can handle different pixel scales
        # They resample internally during the subtraction process
        if sci_pixscale and ref_pixscale:
            if abs(sci_pixscale - ref_pixscale) < 0.0001 and sci_shape == ref_shape:
                use_cases['image_subtraction']['wcs_only_sufficient'] = True
                use_cases['image_subtraction']['analysis'] = "Identical grids - WCS-only sufficient"
            else:
                # Different pixel scales are OK for template subtraction
                use_cases['image_subtraction']['wcs_only_sufficient'] = True
                use_cases['image_subtraction']['analysis'] = f"Different scales OK for template subtraction: sci={sci_pixscale:.4f}\" ref={ref_pixscale:.4f}\""
        else:
            use_cases['image_subtraction']['analysis'] = "Cannot determine pixel scales"
        
        # Source detection and astrometry
        use_cases['source_detection']['analysis'] = "WCS-only sufficient for coordinate-based operations"
        use_cases['astrometry']['analysis'] = "WCS-only sufficient for astrometric measurements"
        
        # Visualization
        use_cases['visualization']['analysis'] = "WCS-only sufficient for coordinate overlays"
        
        analysis['use_case_analysis'] = use_cases
    
    def _analyze_distortion_requirements(self, sci_header, ref_header, analysis):
        """Analyze distortion handling requirements."""
        
        # Check for SIP distortion
        sci_has_sip = any(key in sci_header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"])
        ref_has_sip = any(key in ref_header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"])
        
        # Check for TPV/PV distortion
        sci_has_tpv = any(key.startswith("PV_") for key in sci_header)
        ref_has_tpv = any(key.startswith("PV_") for key in ref_header)
        
        # Check projection types
        sci_ctype = str(sci_header.get("CTYPE1", "")).upper()
        ref_ctype = str(ref_header.get("CTYPE1", "")).upper()
        
        sci_distorted = sci_has_sip or sci_has_tpv or "TAN-SIP" in sci_ctype or "TPV" in sci_ctype
        ref_distorted = ref_has_sip or ref_has_tpv or "TAN-SIP" in ref_ctype or "TPV" in ref_ctype
        
        if sci_distorted or ref_distorted:
            analysis['reasons'].append("⚠ Images contain distortion parameters")
            
            if sci_distorted and ref_distorted:
                analysis['reasons'].append("  Both images have distortion - may need resampling for precise alignment")
            elif sci_distorted:
                analysis['reasons'].append("  Science image distorted - reference may need resampling")
            else:
                analysis['reasons'].append("  Reference image distorted - may need resampling")
            
            # For many operations, distortion can be handled by WCS transformations
            analysis['reasons'].append("  However, WCS transformations can handle distortion for coordinate-based operations")
        else:
            analysis['reasons'].append("✓ No significant distortion detected")
    
    def _analyze_pixel_grid_alignment(self, sci_wcs, ref_wcs, sci_shape, ref_shape, analysis):
        """Analyze pixel grid alignment requirements."""
        
        sci_pixscale = self._get_pixel_scale(sci_wcs)
        ref_pixscale = self._get_pixel_scale(ref_wcs)
        
        if sci_pixscale and ref_pixscale:
            scale_diff = abs(sci_pixscale - ref_pixscale)
            if scale_diff < 0.001:  # Less than 0.001 arcsec difference
                analysis['reasons'].append(f"✓ Pixel scales match: sci={sci_pixscale:.4f}\", ref={ref_pixscale:.4f}\"")
            else:
                # Different pixel scales are OK for template subtraction
                analysis['reasons'].append(f"⚠ Pixel scales differ: sci={sci_pixscale:.4f}\", ref={ref_pixscale:.4f}\" (diff={scale_diff:.4f}\")")
                analysis['reasons'].append("  → Different scales acceptable for template subtraction")
                # Don't mark resampling_required for template subtraction case
        
        if sci_shape == ref_shape:
            analysis['reasons'].append(f"✓ Image shapes match: {sci_shape}")
        else:
            # Different shapes are also OK for template subtraction
            analysis['reasons'].append(f"⚠ Image shapes differ: sci={sci_shape}, ref={ref_shape}")
            analysis['reasons'].append("  → Different shapes acceptable for template subtraction")
            # Don't mark resampling_required for template subtraction case
        
        # Check if coordinate grids align
        try:
            # Test if same pixel coordinates map to same world coordinates
            test_points = [(100, 100), (500, 500), (1000, 1000)]
            max_offset = 0
            
            for x, y in test_points:
                if x < sci_shape[1] and y < sci_shape[0] and x < ref_shape[1] and y < ref_shape[0]:
                    sci_world = sci_wcs.all_pix2world([[x]], [[y]], 0)
                    ref_world = ref_wcs.all_pix2world([[x]], [[y]], 0)
                    
                    if np.isfinite(sci_world[0, 0]) and np.isfinite(ref_world[0, 0]):
                        offset = np.sqrt((sci_world[0, 0] - ref_world[0, 0])**2 + 
                                       (sci_world[1, 0] - ref_world[1, 0])**2) * 3600  # Convert to arcsec
                        max_offset = max(max_offset, offset)
            
            if max_offset < 0.1:  # Less than 0.1 arcsec offset
                analysis['reasons'].append(f"✓ Coordinate grids align (max offset: {max_offset:.3f}\")")
            else:
                analysis['reasons'].append(f"⚠ Coordinate grids misaligned (max offset: {max_offset:.3f}\")")
                # WCS alignment still needed, but resampling not required for template subtraction
                
        except Exception as e:
            analysis['reasons'].append(f"⚠ Could not test coordinate alignment: {e}")
    
    def _get_pixel_scale(self, wcs):
        """Get pixel scale from WCS in arcsec/pixel."""
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            scales = proj_plane_pixel_scales(wcs)
            return scales[0] * 3600.0  # Convert to arcsec/pixel
        except:
            return None
    
    def _make_recommendation(self, analysis):
        """Make final recommendation based on analysis - UPDATED for template subtraction."""
        
        # Check if template subtraction is the primary use case
        template_subtraction_ok = analysis['use_case_analysis'].get('image_subtraction', {}).get('wcs_only_sufficient', False)
        
        if template_subtraction_ok:
            analysis['recommendations'].append("WCS-ONLY ALIGNMENT sufficient for template subtraction")
            analysis['recommendations'].append("Template subtraction algorithms handle different pixel scales internally")
            analysis['wcs_only_possible'] = True
        elif analysis['resampling_required']:
            analysis['recommendations'].append("RESAMPLING REQUIRED for precise alignment")
            analysis['recommendations'].append("Use current SCAMP/SWarp workflow")
        else:
            analysis['recommendations'].append("WCS-ONLY ALIGNMENT may be sufficient")
            analysis['recommendations'].append("Consider WCS-only approach for coordinate-based operations")
        
        # Use case specific recommendations
        for use_case, details in analysis['use_case_analysis'].items():
            if details['wcs_only_sufficient']:
                analysis['recommendations'].append(f"WCS-only sufficient for {use_case}")
            else:
                analysis['recommendations'].append(f"Resampling needed for {use_case}")
        
        # Final determination - template subtraction takes precedence
        if template_subtraction_ok:
            analysis['wcs_only_possible'] = True
        else:
            analysis['wcs_only_possible'] = not analysis['resampling_required']
    
    def generate_alignment_strategy(self, analysis: dict) -> dict:
        """Generate alignment strategy based on analysis."""
        
        strategy = {
            'primary_method': 'scamp_swarp' if analysis.get('resampling_required', False) else 'wcs_only',
            'workflow_steps': [],
            'advantages': [],
            'disadvantages': [],
            'data_quality_impact': {}
        }
        
        if analysis.get('wcs_only_possible', False):
            strategy['primary_method'] = 'wcs_only'
            strategy['workflow_steps'] = [
                '1. Run SCAMP on reference catalog',
                '2. Apply .head file to reference WCS only',
                '3. Use WCS transformations for coordinate operations',
                '4. No pixel resampling required'
            ]
            strategy['advantages'] = [
                'Preserves original pixel values',
                'No interpolation artifacts',
                'Maintains original PSF',
                'Faster processing',
                'Less memory usage'
            ]
            strategy['disadvantages'] = [
                'Different pixel scales may complicate some operations',
                'Coordinate transformations needed for all operations',
                'May not work for image subtraction'
            ]
            strategy['data_quality_impact'] = {
                'photometry': 'minimal impact - original values preserved',
                'astrometry': 'good - relies on WCS accuracy',
                'psf': 'preserved - no interpolation',
                'noise': 'preserved - no resampling noise'
            }
        else:
            strategy['primary_method'] = 'scamp_swarp'
            strategy['workflow_steps'] = [
                '1. Run SCAMP on reference catalog',
                '2. Apply .head file to reference image',
                '3. Resample both images with SWarp',
                '4. Ensure identical pixel grids'
            ]
            strategy['advantages'] = [
                'Identical pixel grids',
                'Simplified subsequent processing',
                'Works for all operations including subtraction',
                'Consistent coordinate system'
            ]
            strategy['disadvantages'] = [
                'Interpolation artifacts possible',
                'PSF may be modified',
                'Processing time and memory usage',
                'Potential data quality degradation'
            ]
            strategy['data_quality_impact'] = {
                'photometry': 'potential interpolation effects',
                'astrometry': 'excellent - precise grid alignment',
                'psf': 'modified by resampling kernel',
                'noise': 'correlated by resampling'
            }
        
        return strategy

def analyze_alignment_feasibility(sci_image_path: str, ref_image_path: str) -> dict:
    """
    Main function to analyze if WCS-only alignment is possible.
    
    Args:
        sci_image_path: Path to science image
        ref_image_path: Path to reference image
        
    Returns:
        Complete analysis with recommendations
    """
    
    analyzer = AlignmentMethodAnalyzer()
    analysis = analyzer.analyze_alignment_requirements(sci_image_path, ref_image_path)
    strategy = analyzer.generate_alignment_strategy(analysis)
    
    return {
        'analysis': analysis,
        'strategy': strategy
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        sci_path = sys.argv[1]
        ref_path = sys.argv[2]
        
        result = analyze_alignment_feasibility(sci_path, ref_path)
        
        print("=== ALIGNMENT FEASIBILITY ANALYSIS ===")
        print(f"WCS-only alignment possible: {result['analysis']['wcs_only_possible']}")
        print(f"Resampling required: {result['analysis']['resampling_required']}")
        
        print("\nReasons:")
        for reason in result['analysis']['reasons']:
            print(f"  {reason}")
        
        print("\nRecommendations:")
        for rec in result['analysis']['recommendations']:
            print(f"  {rec}")
        
        print(f"\nRecommended method: {result['strategy']['primary_method']}")
        print("\nWorkflow steps:")
        for step in result['strategy']['workflow_steps']:
            print(f"  {step}")
        
        print("\nAdvantages:")
        for adv in result['strategy']['advantages']:
            print(f"  {adv}")
        
        print("\nDisadvantages:")
        for dis in result['strategy']['disadvantages']:
            print(f"  {dis}")
    else:
        print("Usage: python wcs_only_alignment_analysis.py <science_image> <reference_image>")
