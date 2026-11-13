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

from autophot.photometry.aperture import _empirical_noise_factor


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


def test_spatially_varying_noise_stays_near_one():
    """Two depth zones with an honest variance map must not inflate the
    factor: standardising each residual by its own prediction keeps the
    estimator unbiased when the noise varies across the frame."""
    rng = np.random.default_rng(42)
    image = rng.normal(100.0, 3.0, (NY, NX))
    image[:, NX // 2 :] = rng.normal(100.0, 12.0, (NY, NX // 2))
    error = np.full((NY, NX), 3.0)
    error[:, NX // 2 :] = 12.0
    factor, n_used = _empirical_noise_factor(
        image, error, RADIUS, ANN_IN, ANN_OUT
    )
    assert n_used >= 50
    assert factor == pytest.approx(1.0, abs=0.2)


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


def _correlated_image(seed=0):
    """Smoothed noise renormalized to per-pixel sigma 30, matching the
    rms map: correlation alone then breaks the aperture-sum prediction
    (factor > 1.5), mimicking a resampled frame."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.normal(0.0, 1.0, (NY, NX)), 1.5)
    return 100.0 + 30.0 * n / n.std()


def _minimal_input_yaml(tmp_path, wdir=True):
    return {
        "fwhm": 4.0,
        "imageFilter": "r",
        "gain": 1.0,
        "exposure_time": 60.0,
        "read_noise": 0.0,
        "global_verbose_level": 0,
        "wdir": str(tmp_path) if wdir else None,
        "fpath": str(tmp_path / "TEST_0001_APT.fits"),
        "photometry": {
            "aperture_radius": RADIUS,
            "empirical_noise_factor": True,
        },
    }


def test_measure_writes_ledger_once_per_file(tmp_path):
    """Repeated measure() passes on one file append a single ledger line."""
    import pandas as pd

    from autophot.driver import _noise_factor_ledger_path
    from autophot.photometry.aperture import Aperture

    image = _correlated_image()
    rms = np.full((NY, NX), 30.0)
    input_yaml = _minimal_input_yaml(tmp_path)
    sources = pd.DataFrame({"x_pix": [200.0], "y_pix": [200.0]})

    # Two Aperture instances sharing input_yaml mimic the repeated
    # measure() passes a file undergoes inside one subprocess.
    for _ in range(2):
        Aperture(input_yaml=input_yaml, image=image).measure(
            sources=sources.copy(), background_rms=rms
        )

    ledger = _noise_factor_ledger_path(tmp_path)
    with open(ledger) as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 1
    tag, _, val = lines[0].rpartition(",")
    assert tag == "TEST_0001_APT.fits"
    assert float(val) > 1.5


def test_measure_no_ledger_when_quiet(tmp_path):
    """A field whose factor stays <= 1.5 writes no ledger entry."""
    import pandas as pd

    from autophot.driver import _noise_factor_ledger_path
    from autophot.photometry.aperture import Aperture

    rng = np.random.default_rng(7)
    image = rng.normal(100.0, 5.0, (NY, NX))
    rms = np.full((NY, NX), 5.0)
    input_yaml = _minimal_input_yaml(tmp_path)
    sources = pd.DataFrame({"x_pix": [200.0], "y_pix": [200.0]})
    Aperture(input_yaml=input_yaml, image=image).measure(
        sources=sources, background_rms=rms
    )
    import os

    ledger = _noise_factor_ledger_path(tmp_path)
    assert not os.path.exists(ledger) or all(
        not ln.strip() for ln in open(ledger)
    )


def test_summary_emits_one_aggregated_warning(tmp_path):
    """The driver summary reports a single line for many images."""
    from autophot.driver import (
        _noise_factor_ledger_path,
        _report_noise_factor_summary,
        _reset_noise_factor_ledger,
    )

    _reset_noise_factor_ledger(tmp_path)
    ledger = _noise_factor_ledger_path(tmp_path)
    with open(ledger, "a") as fh:
        for i, f in enumerate((2.1, 2.4, 3.0, 1.6)):
            fh.write(f"FILE_{i:03d}_APT.fits,{f:.3f}\n")
        fh.write("FILE_001_APT.fits,2.9\n")  # duplicate tag dedupes

    lines = []
    _report_noise_factor_summary(tmp_path, lines.append)
    assert len(lines) == 1
    msg = lines[0]
    assert "4 image(s)" in msg
    assert "worst 3.0x" in msg


def test_summary_silent_without_ledger(tmp_path):
    from autophot.driver import (
        _report_noise_factor_summary,
        _reset_noise_factor_ledger,
    )

    _reset_noise_factor_ledger(tmp_path)
    lines = []
    _report_noise_factor_summary(tmp_path, lines.append)
    assert lines == []
