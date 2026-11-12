"""Tests for the empirical empty-sky noise factor in aperture.py.

The factor compares the measured scatter of sky-subtracted aperture sums
at random empty positions against the propagated pixel-variance
prediction.  Resampling correlates pixel noise and unresolved faint
sources add scatter, so the ratio can only be >= 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from aperture import _empirical_noise_factor


NY = NX = 400
RADIUS, ANN_IN, ANN_OUT = 5.0, 10.0, 15.0


def test_white_noise_factor_near_one():
    """Independent pixel noise with an honest variance map -> ~1."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    error = np.full((NY, NX), 5.0)
    factor, n_used = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert n_used >= 50
    assert factor == pytest.approx(1.0, abs=0.25)


def test_correlated_noise_inflates_factor():
    """Smoothed (resampled) noise correlates neighbouring pixels; the
    aperture-sum scatter then exceeds the independent-pixel prediction
    and the factor must report it."""
    rng = np.random.default_rng(42)
    image = gaussian_filter(rng.normal(100.0, 5.0, (NY, NX)), 1.5)
    # Naive pipeline: per-pixel variance of the smoothed image.
    error = np.full((NY, NX), image.std())
    factor, _ = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert factor > 1.5


def test_confusion_noise_inflates_factor():
    """Undetected faint sources inside test apertures scatter the sums
    beyond the noise-model prediction."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    for _ in range(400):
        x0, y0 = rng.uniform(15, 385, 2)
        image += rng.uniform(50, 200) * np.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2) / 4.0
        )
    error = np.full((NY, NX), 5.0)
    factor, _ = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert factor > 1.2


def test_detected_sources_are_excluded():
    """A bright source sitting inside a test aperture is rejected by
    source_xy exclusion plus the residual sigma clip, so a field with
    one bright star still measures the noise floor."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    yy, xx = np.mgrid[0:NY, 0:NX]
    image += 5e4 * np.exp(-((xx - 200.0) ** 2 + (yy - 200.0) ** 2) / 9.0)
    error = np.full((NY, NX), 5.0)
    factor, _ = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT, source_xy=[[200.0, 200.0]]
    )
    assert factor == pytest.approx(1.0, abs=0.3)


def test_factor_floored_at_one():
    """Errors larger than reality must not shrink the reported error."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    error = np.full((NY, NX), 50.0)  # variance map 10x too large
    factor, _ = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert factor == 1.0


def test_missing_error_map_returns_one():
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    assert _empirical_noise_factor(image, None, RADIUS, ANN_IN, ANN_OUT) == (
        1.0,
        0,
    )


def test_tiny_image_returns_one():
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (24, 24))
    error = np.full((24, 24), 5.0)
    factor, n_used = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert factor == 1.0
    assert n_used == 0


def test_heavily_masked_image_returns_one():
    """With most pixels masked there are too few usable empty apertures;
    the factor degrades to 1.0 rather than a junk value."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 5.0, (NY, NX))
    error = np.full((NY, NX), 5.0)
    defects = np.zeros((NY, NX), dtype=bool)
    defects[:, ::2] = True  # every other column bad
    factor, n_used = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT, defects_mask=defects
    )
    assert factor == 1.0
