"""End-to-end MaxiMask test on a synthetic contaminated frame.

Builds the CCD-like test image from ``synthetic_image.py`` (cosmic rays,
hot/dead columns and pixels, a satellite trail, a saturated star with
diffraction spikes, nebulosity, fringing, overscan), runs the real
TensorFlow inference from Paillassa, Bertin & Bouy (2020,
arXiv:1907.08298) and checks that the strong-signal contaminants are
recovered. Paper-style plots are written to ``output/`` next to this
file.

Recall bounds are set well below the measured values (in parentheses)
to tolerate batch-level nondeterminism in the network:

    CR 0.83, HCL 0.76, DCL 0.67, HP 0.32, DP 0.82,
    TRL 0.88, SAT 0.91, BG-on-sky 0.97, defect false-positive 0.4%.

FR, NEB, SP and OV are exercised but not asserted: the injected
synthetic versions are weak proxies for the real signatures the network
was trained on (smooth nebulosity is absorbed by its internal
background normalisation or flagged as BBG, and single-pixel hot
clusters/spikes only partially match real diffraction spikes).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maximask_and_maxitrack", reason="MaxiMask not installed")
pytest.importorskip("tensorflow", reason="tensorflow not installed")

from astropy.io import fits

from tests.MaxiMask import run_maximask as mm

pytestmark = pytest.mark.slow

CLASS_IDX = {a: i for i, a in enumerate(mm.CLASS_ABBR)}
DEFECT_CLASSES = ["CR", "HCL", "DCL", "HP", "TRL", "SAT", "SP"]


@pytest.fixture(scope="module")
def result():
    """Run the whole pipeline once; reuses cached FITS/prob maps."""
    return mm.run_pipeline(figures=True)


def _recall(result, abbr):
    det = result["masks"][CLASS_IDX[abbr]]
    tru = result["data"]["truth"][abbr]
    assert tru.any()
    return float((det & tru).sum()) / float(tru.sum())


def test_probmap_cube_shape_and_range(result):
    probs = result["probs"]
    assert probs.shape == (len(mm.CLASS_ABBR),) + mm.syn.IMAGE_SHAPE
    assert np.isfinite(probs).all()
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_mask_fits_header_tags():
    with fits.open(mm.MASK_FITS) as hdul:
        hdr = hdul[0].header
        assert hdr["PRIORS"] == "Yes"
        for abbr in mm.CLASS_ABBR:
            assert f"{abbr}_DIM" in hdr


def test_satellite_trail_recall(result):
    assert _recall(result, "TRL") >= 0.6


def test_cosmic_ray_recall(result):
    assert _recall(result, "CR") >= 0.5


def test_saturated_core_recall(result):
    assert _recall(result, "SAT") >= 0.6


def test_column_recall(result):
    assert _recall(result, "HCL") >= 0.5
    assert _recall(result, "DCL") >= 0.4


def test_pixel_recall(result):
    # Isolated synthetic hot pixels are a hard case for the network.
    assert _recall(result, "HP") >= 0.10
    assert _recall(result, "DP") >= 0.5


def test_background_class_covers_sky(result):
    det = result["masks"][CLASS_IDX["BG"]]
    clean = result["data"]["clean"]
    assert float((det & clean).sum()) / float(clean.sum()) >= 0.8


def test_low_false_positive_rate(result):
    clean = result["data"]["clean"]
    defect = np.zeros(clean.shape, dtype=bool)
    for abbr in DEFECT_CLASSES:
        defect |= result["masks"][CLASS_IDX[abbr]]
    assert float((defect & clean).sum()) / float(clean.sum()) < 0.02


def test_plots_and_summary_written(result):
    for key in ("input", "overlay", "probmaps", "zooms"):
        path = result["paths"][key]
        assert path.exists() and path.stat().st_size > 30_000
    summary = mm.OUTPUT_DIR / "summary.txt"
    assert summary.exists() and "TRL" in summary.read_text()


def test_integrates_into_hardware_defects(result):
    """background.use_maximask unions detections into the defect masks.

    Exercises the real wiring used downstream: the returned
    hardware_defects_mask feeds PSF candidate exclusion and
    defects_mask, so detections must land there - not just in a
    standalone array.
    """
    from autophot.pipeline.background import BackgroundSubtractor

    sub = BackgroundSubtractor(
        {
            "saturate": 60000.0,
            "background": {
                "use_maximask": True,
                # Defaults exclude TRL; request it explicitly so the test
                # is independent of the shipped default class set.
                "maximask_classes": ["TRL"],
            },
        }
    )
    out = sub.remove(result["data"]["image"], plot=False)
    hard = out["hardware_defects_mask"]
    tru = result["data"]["truth"]["TRL"]
    assert float((hard & tru).sum()) / float(tru.sum()) >= 0.5
