"""Shared pytest fixtures for AutoPhOT tests.

Provides synthetic data fixtures and common configuration so that all
test modules can rely on known-truth images with reproducible random seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the autophot package directory is on sys.path so that top-level
# modules (functions, psf, aperture, etc.) can be imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.synthetic.images import (
    make_single_source_image,
    make_synthetic_image,
    make_faint_source_image,
    make_saturated_source_image,
    make_edge_source_image,
    make_nan_region_image,
)
from tests.synthetic.catalogs import make_isolated_catalog, make_blended_catalog
from tests.synthetic.psf_models import gaussian_psf_2d, moffat_psf_2d


# ---------------------------------------------------------------------------
# Basic synthetic images
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_source():
    """Single bright isolated source at image centre (high S/N)."""
    return make_single_source_image(
        flux=50000.0, fwhm=4.0, image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def faint_source():
    """Single faint source at ~5 sigma."""
    return make_faint_source_image(
        snr_target=5.0, fwhm=4.0, image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def very_faint_source():
    """Single very faint source at ~3 sigma."""
    return make_faint_source_image(
        snr_target=3.0, fwhm=4.0, image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def saturated_source():
    """Saturated source with clipped peak."""
    return make_saturated_source_image(
        flux=1000000.0, saturate=50000.0, fwhm=4.0,
        image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def edge_source():
    """Source near image edge."""
    return make_edge_source_image(
        flux=10000.0, fwhm=4.0, edge_offset=5.0,
        image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def nan_region_image():
    """Image with NaN chip-gap band."""
    return make_nan_region_image(
        flux=10000.0, fwhm=4.0, nan_fraction=0.1,
        image_shape=(256, 256), seed=42,
    )


@pytest.fixture
def isolated_field():
    """Field of 50 well-separated sources."""
    cat = make_isolated_catalog(
        n_sources=50, image_shape=(512, 512),
        min_separation=30.0, seed=42,
    )
    return make_synthetic_image(
        catalog=cat, image_shape=(512, 512),
        fwhm=4.0, background=100.0, background_rms=10.0,
        seed=42,
    )


@pytest.fixture
def crowded_field():
    """Field of 100 closely spaced sources."""
    cat = make_isolated_catalog(
        n_sources=100, image_shape=(512, 512),
        min_separation=8.0, seed=42,
    )
    return make_synthetic_image(
        catalog=cat, image_shape=(512, 512),
        fwhm=4.0, background=100.0, background_rms=10.0,
        seed=42,
    )


@pytest.fixture
def blended_field():
    """Field with blended source pairs."""
    cat = make_blended_catalog(
        n_pairs=10, image_shape=(512, 512),
        separation_range=(2.0, 5.0), seed=42,
    )
    return make_synthetic_image(
        catalog=cat, image_shape=(512, 512),
        fwhm=4.0, background=100.0, background_rms=10.0,
        seed=42,
    )


@pytest.fixture
def gradient_field():
    """Field with background gradient."""
    cat = make_isolated_catalog(
        n_sources=30, image_shape=(512, 512),
        min_separation=30.0, seed=42,
    )
    return make_synthetic_image(
        catalog=cat, image_shape=(512, 512),
        fwhm=4.0, background=100.0, background_rms=10.0,
        add_background_gradient=20.0, seed=42,
    )


# ---------------------------------------------------------------------------
# PSF model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_psf():
    """Return the gaussian_psf_2d function."""
    return gaussian_psf_2d


@pytest.fixture
def moffat_psf():
    """Return the moffat_psf_2d function."""
    return moffat_psf_2d


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_input_yaml():
    """Minimal input_yaml dict for testing photometry functions."""
    return {
        "target_name": "TestTarget",
        "target_ra": 150.0,
        "target_dec": 30.0,
        "fwhm": 4.0,
        "fwhm_err": 0.2,
        "gain": 1.0,
        "read_noise": 5.0,
        "exposure_time": 30.0,
        "saturate": 50000.0,
        "pixel_scale": 1.0,
        "scale": 0,
        "photometry": {
            "detection_limit": 3.0,
            "aperture_radius": 6.0,
            "undersampled_fwhm_threshold": 2.5,
            "critical_fwhm_threshold": 3.0,
            "oversampled_fwhm_threshold": 5.0,
            "systematic_error_floor_mag": 0.0,
        },
        "limiting_magnitude": {
            "recovery_method": "AP",
            "completeness_target": 0.5,
            "inject_source_location": 6.0,
            "inject_final_n_sites": 20,
            "inject_min_radius_fwhm": 2.0,
            "inject_max_radius_fwhm": 6.0,
        },
        "rng_seed": 42,
    }
