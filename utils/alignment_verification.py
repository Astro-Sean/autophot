#!/usr/bin/env python3
"""
Alignment verification utilities for SCAMP/SWarp precise alignment validation.
Provides comprehensive diagnostics to ensure science and reference images are precisely aligned.
"""

import logging
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class AlignmentVerifier:
    """Comprehensive alignment verification for SCAMP/SWarp processed images.

    Verifies WCS consistency, pixel-grid alignment, coordinate transformation
    accuracy, and resampling quality between science and reference images
    after astrometric alignment.
    """

    def __init__(self, verbose_level: int = 1):
        """Initialise the verifier.

        Parameters
        ----------
        verbose_level : int
            Logging verbosity (0 = quiet, 1 = info, 2 = debug).
        """
        self.verbose_level = verbose_level
        self.logger = logging.getLogger(__name__)

    def verify_precise_alignment(
        self,
        sci_image_path: str,
        ref_image_path: str,
        tolerance_pixels: float = 0.5,
        output_dir: str = None,
    ) -> dict:
        """Perform comprehensive alignment verification between two images.

        Parameters
        ----------
        sci_image_path : str
            Path to the aligned science image.
        ref_image_path : str
            Path to the aligned reference image.
        tolerance_pixels : float
            Acceptable alignment tolerance in pixels (default 0.5).
        output_dir : str, optional
            Directory to save diagnostic plots.  If ``None``, no plots
            are generated.

        Returns
        -------
        dict
            Alignment verification results with keys for WCS consistency,
            pixel alignment, coordinate accuracy, resampling quality, and
            an overall alignment score.
        """
        
        results = {
            'precise_alignment': False,
            'alignment_quality': 'unknown',
            'wcs_consistency': {},
            'pixel_alignment': {},
            'source_matching': {},
            'resampling_quality': {},
            'diagnostics': {}
        }
        
        try:
            # Load images and WCS
            with fits.open(sci_image_path) as sci_hdul, fits.open(ref_image_path) as ref_hdul:
                sci_data = sci_hdul[0].data
                ref_data = ref_hdul[0].data
                sci_header = sci_hdul[0].header
                ref_header = ref_hdul[0].header
                
                sci_wcs = WCS(sci_header)
                ref_wcs = WCS(ref_header)
                
                # 1. Verify WCS consistency
                wcs_results = self._verify_wcs_consistency(sci_wcs, ref_wcs, sci_header, ref_header)
                results['wcs_consistency'] = wcs_results
                
                # 2. Verify pixel grid alignment
                pixel_results = self._verify_pixel_grid_alignment(sci_data, ref_data, sci_header, ref_header)
                results['pixel_alignment'] = pixel_results
                
                # 3. Verify coordinate transformation accuracy
                coord_results = self._verify_coordinate_accuracy(sci_wcs, ref_wcs, sci_data.shape)
                results['coordinate_accuracy'] = coord_results
                
                # 4. Verify resampling quality
                resamp_results = self._verify_resampling_quality(sci_data, ref_data)
                results['resampling_quality'] = resamp_results
                
                # 5. Overall alignment assessment
                alignment_score = self._calculate_alignment_score(results)
                results['alignment_score'] = alignment_score
                results['precise_alignment'] = alignment_score > 0.8
                results['alignment_quality'] = self._get_quality_label(alignment_score)
                
                # 6. Generate diagnostic plots
                if output_dir:
                    self._generate_diagnostics(sci_data, ref_data, sci_wcs, ref_wcs, 
                                             results, output_dir)
                    results['diagnostics']['plots_generated'] = True
                
                self._log_verification_results(results)
                
        except Exception as e:
            self.logger.error(f"Alignment verification failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def _verify_wcs_consistency(self, sci_wcs: WCS, ref_wcs: WCS, sci_header, ref_header) -> dict:
        """Check that key WCS parameters match between science and reference."""
        
        results = {
            'consistent': True,
            'differences': {},
            'issues': []
        }
        
        # Check key WCS parameters
        wcs_params = ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 
                     'CDELT1', 'CDELT2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        
        for param in wcs_params:
            sci_val = sci_header.get(param, None)
            ref_val = ref_header.get(param, None)
            
            if sci_val is not None and ref_val is not None:
                if param.startswith('CRVAL'):  # World coordinates - small tolerance
                    tolerance = 1e-6  # degrees
                    diff = abs(float(sci_val) - float(ref_val))
                    if diff > tolerance:
                        results['consistent'] = False
                        results['issues'].append(f"{param} differs by {diff:.8f} deg")
                    results['differences'][param] = diff
                    
                elif param.startswith('CRPIX'):  # Pixel coordinates - very small tolerance
                    tolerance = 0.01  # pixels
                    diff = abs(float(sci_val) - float(ref_val))
                    if diff > tolerance:
                        results['consistent'] = False
                        results['issues'].append(f"{param} differs by {diff:.3f} px")
                    results['differences'][param] = diff
                    
                elif param.startswith('CDELT'):  # Pixel scale - small tolerance
                    tolerance = 1e-6  # degrees/pixel
                    diff = abs(float(sci_val) - float(ref_val))
                    if diff > tolerance:
                        results['consistent'] = False
                        results['issues'].append(f"{param} differs by {diff:.8f} deg/px")
                    results['differences'][param] = diff
        
        # Check projection types
        sci_ctype1 = str(sci_header.get('CTYPE1', '')).upper()
        ref_ctype1 = str(ref_header.get('CTYPE1', '')).upper()
        if sci_ctype1 != ref_ctype1:
            results['consistent'] = False
            results['issues'].append(f"CTYPE1 mismatch: {sci_ctype1} vs {ref_ctype1}")
        
        return results
    
    def _verify_pixel_grid_alignment(self, sci_data, ref_data, sci_header, ref_header) -> dict:
        """Check that image shapes and NAXIS keywords match."""
        
        results = {
            'consistent': True,
            'shape_match': sci_data.shape == ref_data.shape,
            'issues': []
        }
        
        if not results['shape_match']:
            results['consistent'] = False
            results['issues'].append(f"Shape mismatch: sci {sci_data.shape} vs ref {ref_data.shape}")
        
        # Check NAXIS keywords
        sci_naxis1 = sci_header.get('NAXIS1', sci_data.shape[1])
        sci_naxis2 = sci_header.get('NAXIS2', sci_data.shape[0])
        ref_naxis1 = ref_header.get('NAXIS1', ref_data.shape[1])
        ref_naxis2 = ref_header.get('NAXIS2', ref_data.shape[0])
        
        if sci_naxis1 != ref_naxis1 or sci_naxis2 != ref_naxis2:
            results['consistent'] = False
            results['issues'].append(f"NAXIS mismatch: sci ({sci_naxis1},{sci_naxis2}) vs ref ({ref_naxis1},{ref_naxis2})")
        
        return results
    
    def _verify_coordinate_accuracy(self, sci_wcs: WCS, ref_wcs: WCS, image_shape) -> dict:
        """Measure round-trip pixel→world→pixel offsets across a 10×10 grid."""
        
        results = {
            'max_offset_pixels': 0,
            'mean_offset_pixels': 0,
            'consistent': True,
            'test_points': []
        }
        
        # Test coordinate transformation at various points across the image
        ny, nx = image_shape
        
        # Create a grid of test points
        test_points = []
        for y in np.linspace(0, ny-1, 10):
            for x in np.linspace(0, nx-1, 10):
                test_points.append((x, y))
        
        offsets = []
        
        for x, y in test_points:
            # Transform pixel coordinates to world coordinates using both WCS
            try:
                sci_world = sci_wcs.all_pix2world([[x]], [[y]], 0)
                ref_world = ref_wcs.all_pix2world([[x]], [[y]], 0)
                
                # Transform back to pixels using opposite WCS
                sci_back_to_ref = ref_wcs.all_world2pix(sci_world[0, 0], sci_world[1, 0], 0)
                ref_back_to_sci = sci_wcs.all_world2pix(ref_world[0, 0], ref_world[1, 0], 0)
                
                # Calculate offsets
                offset_x = abs(sci_back_to_ref[0] - x)
                offset_y = abs(sci_back_to_ref[1] - y)
                offset_total = np.sqrt(offset_x**2 + offset_y**2)
                
                offsets.append(offset_total)
                results['test_points'].append({
                    'position': (x, y),
                    'offset_pixels': offset_total
                })
                
            except Exception as e:
                self.logger.warning(f"Coordinate test failed at ({x},{y}): {e}")
                results['consistent'] = False
        
        if offsets:
            results['max_offset_pixels'] = max(offsets)
            results['mean_offset_pixels'] = np.mean(offsets)
            results['std_offset_pixels'] = np.std(offsets)
            
            # Consider alignment precise if max offset < 0.5 pixels
            if results['max_offset_pixels'] > 0.5:
                results['consistent'] = False
        
        return results
    
    def _verify_resampling_quality(self, sci_data, ref_data) -> dict:
        """Assess resampling quality via NaN fraction and outlier statistics."""
        
        results = {
            'quality_score': 0,
            'issues': [],
            'statistics': {}
        }
        
        # Basic statistics
        sci_mean, sci_median, sci_std = sigma_clipped_stats(sci_data)
        ref_mean, ref_median, ref_std = sigma_clipped_stats(ref_data)
        
        results['statistics'] = {
            'science': {'mean': sci_mean, 'median': sci_median, 'std': sci_std},
            'reference': {'mean': ref_mean, 'median': ref_median, 'std': ref_std}
        }
        
        # Check for resampling artifacts
        # 1. NaN values
        sci_nan_frac = np.isnan(sci_data).sum() / sci_data.size
        ref_nan_frac = np.isnan(ref_data).sum() / ref_data.size
        
        if sci_nan_frac > 0.01 or ref_nan_frac > 0.01:
            results['issues'].append(f"High NaN fraction: sci {sci_nan_frac:.3f}, ref {ref_nan_frac:.3f}")
        
        # 2. Extreme values (possible resampling artifacts)
        sci_extreme_frac = (np.abs(sci_data - sci_median) > 5 * sci_std).sum() / sci_data.size
        ref_extreme_frac = (np.abs(ref_data - ref_median) > 5 * ref_std).sum() / ref_data.size
        
        if sci_extreme_frac > 0.001 or ref_extreme_frac > 0.001:
            results['issues'].append(f"High extreme value fraction: sci {sci_extreme_frac:.4f}, ref {ref_extreme_frac:.4f}")
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= min(sci_nan_frac * 10, 0.3)  # Penalize NaN values
        quality_score -= min(ref_nan_frac * 10, 0.3)
        quality_score -= min(sci_extreme_frac * 50, 0.2)  # Penalize extreme values
        quality_score -= min(ref_extreme_frac * 50, 0.2)
        
        results['quality_score'] = max(0, quality_score)
        
        return results
    
    def _calculate_alignment_score(self, results: dict) -> float:
        """Compute a weighted alignment score (0–1) from all sub-checks."""
        
        score = 1.0
        
        # WCS consistency (30% weight)
        if results['wcs_consistency']['consistent']:
            score -= 0.0
        else:
            score -= 0.3 * len(results['wcs_consistency']['issues']) / 5.0
        
        # Pixel alignment (25% weight)
        if not results['pixel_alignment']['consistent']:
            score -= 0.25
        
        # Coordinate accuracy (30% weight)
        if 'coordinate_accuracy' in results:
            coord_max_offset = results['coordinate_accuracy'].get('max_offset_pixels', 0)
            score -= min(coord_max_offset * 0.6, 0.3)  # 0.5 px offset = 0.3 penalty
        
        # Resampling quality (15% weight)
        if 'resampling_quality' in results:
            resamp_score = results['resampling_quality'].get('quality_score', 1.0)
            score -= (1.0 - resamp_score) * 0.15
        
        return max(0, score)
    
    def _get_quality_label(self, score: float) -> str:
        """Map a numeric score to a quality label string."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.6:
            return "poor"
        else:
            return "failed"
    
    def _generate_diagnostics(self, sci_data, ref_data, sci_wcs, ref_wcs, results, output_dir):
        """Save diagnostic plots (image trio + coordinate offset map) to *output_dir*."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Difference image
        diff_data = sci_data - ref_data
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(sci_data, cmap='gray', origin='lower')
        plt.title('Science Image')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(ref_data, cmap='gray', origin='lower')
        plt.title('Reference Image')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(diff_data, cmap='RdBu_r', origin='lower', vmin=-np.percentile(np.abs(diff_data), 99),
                   vmax=np.percentile(np.abs(diff_data), 99))
        plt.title('Difference (Science - Reference)')
        plt.colorbar()
        
        # 2. Coordinate offset map
        if 'coordinate_accuracy' in results and results['coordinate_accuracy']['test_points']:
            plt.subplot(2, 2, 4)
            points = results['coordinate_accuracy']['test_points']
            if points:
                xs = [p['position'][0] for p in points]
                ys = [p['position'][1] for p in points]
                offsets = [p['offset_pixels'] for p in points]
                
                scatter = plt.scatter(xs, ys, c=offsets, cmap='viridis', s=20)
                plt.colorbar(scatter, label='Offset (pixels)')
                plt.title('Coordinate Offset Map')
                plt.xlabel('X (pixels)')
                plt.ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'alignment_verification.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Alignment verification plots saved to {output_dir}")
    
    def _log_verification_results(self, results: dict):
        """Log a human-readable summary of all verification sub-checks."""
        
        self.logger.info(f"Alignment Verification Results:")
        self.logger.info(f"  Overall Quality: {results['alignment_quality']} (score: {results.get('alignment_score', 0):.3f})")
        self.logger.info(f"  Precise Alignment: {results['precise_alignment']}")
        
        if 'wcs_consistency' in results:
            wcs = results['wcs_consistency']
            self.logger.info(f"  WCS Consistency: {'PASS' if wcs['consistent'] else 'FAIL'}")
            if wcs['issues']:
                for issue in wcs['issues']:
                    self.logger.warning(f"    {issue}")
        
        if 'pixel_alignment' in results:
            pix = results['pixel_alignment']
            self.logger.info(f"  Pixel Grid: {'PASS' if pix['consistent'] else 'FAIL'}")
            if not pix['consistent']:
                for issue in pix['issues']:
                    self.logger.warning(f"    {issue}")
        
        if 'coordinate_accuracy' in results:
            coord = results['coordinate_accuracy']
            self.logger.info(f"  Coordinate Accuracy: max offset = {coord.get('max_offset_pixels', 0):.3f} px")
        
        if 'resampling_quality' in results:
            resamp = results['resampling_quality']
            self.logger.info(f"  Resampling Quality: {resamp.get('quality_score', 0):.3f}")
            if resamp['issues']:
                for issue in resamp['issues']:
                    self.logger.warning(f"    {issue}")
