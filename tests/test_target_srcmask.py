"""Tests for the target-protecting detected-source mask merge.

On a difference image the detected-source mask keeps residual light from
host and neighbour footprints out of the aperture annulus and the PSF
local background.  For a host-embedded transient that same mask used to
cover the whole measurement region: the annulus worker then failed with
``annulus_too_many_nans`` and the PSF fit collapsed to ~zero flux.
``_merge_target_source_mask`` reopens only the component touching the
target, first inside the fit window and - when the annulus or fit region
would still be unusable - out to the measurement edge.  Unrelated
footprints and hardware defects must stay masked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label

from autophot.pipeline.main import _merge_target_source_mask
from autophot.photometry.aperture import Aperture


NY = NX = 200
TX, TY = 100.0, 100.0
FWHM = 7.0
AP_RAD = 16.0

# Geometry as computed in run_photometry for these values.
R_KEEP = min(max(AP_RAD, FWHM), AP_RAD + 0.5 * FWHM)
ANN_IN = float(np.ceil(AP_RAD + 0.75 * FWHM))
ANN_OUT = float(np.ceil(ANN_IN + 2.0 * FWHM))
R_FIT = max(ANN_IN, 3.0 * FWHM)
R_MEAS = max(ANN_OUT, R_FIT)
R_SELF = min(R_KEEP, max(1.5, 0.5 * FWHM))

POSITIONS = [(TX, TY)]
YY, XX = np.mgrid[0:NY, 0:NX]


def _gau(x0, y0, fwhm, amp):
    s = fwhm / 2.3548
    return amp * np.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * s * s))


def _disk(x0, y0, r):
    return (XX - x0) ** 2 + (YY - y0) ** 2 <= r**2


def _annulus_ring(x0=TX, y0=TY):
    d2 = (XX - x0) ** 2 + (YY - y0) ** 2
    return (d2 >= ANN_IN**2) & (d2 < ANN_OUT**2)


def _merge(src, hw=None, image=None):
    if hw is None:
        hw = np.zeros((NY, NX), dtype=bool)
    return _merge_target_source_mask(
        hw,
        src,
        POSITIONS,
        r_keep=R_KEEP,
        r_self=R_SELF,
        r_fit=R_FIT,
        ann_in=ANN_IN,
        ann_out=ANN_OUT,
        r_meas=R_MEAS,
        image=image,
    )


def _legacy_merge(src, hw=None):
    """Pre-fix behaviour: union plus keep disk, no component reopen."""
    if hw is None:
        hw = np.zeros((NY, NX), dtype=bool)
    merged = hw | src
    merged[_disk(TX, TY, R_KEEP)] = False
    return merged


def _measure_aperture(image, mask):
    yam = {
        "fwhm": FWHM,
        "read_noise": 0.0,
        "gain": 1.0,
        "exposure_time": 1.0,
        "imageFilter": "r",
        "target_name": "SN",
        "global_verbose_level": 0,
        "photometry": {
            "aperture_radius": AP_RAD,
            "enforce_nonnegative_local_background": False,
        },
    }
    sources = pd.DataFrame({"x_pix": [TX], "y_pix": [TY]})
    return Aperture(input_yaml=yam, image=image, verbose=0).measure(
        sources=sources,
        ap_size=AP_RAD,
        exposure_time=1.0,
        gain=1.0,
        mask=mask,
    )


def test_host_embedded_target_recovers_annulus():
    """A host footprint covering the annulus used to fail the aperture
    worker outright; the component reopen must leave a usable annulus
    and a positive flux while the separate neighbour stays masked."""
    rng = np.random.default_rng(0)
    image = rng.normal(0.0, 1.0, (NY, NX))
    image += _gau(TX, TY, 5.0, 500.0)     # SN residual
    image += _gau(80.0, 90.0, 25.0, 30.0)  # host residual, offset nucleus
    image += _gau(140.0, 140.0, 4.0, 200.0)  # neighbour residual
    src = _disk(80.0, 90.0, 40.0) | _disk(140.0, 140.0, 10.0)

    legacy = _legacy_merge(src)
    ann_ring = _annulus_ring()
    legacy_ann_bad = np.count_nonzero(legacy & ann_ring) / ann_ring.sum()
    assert legacy_ann_bad > 0.5
    out = _measure_aperture(image, legacy)
    assert out["fail_reason"].iloc[0] == "annulus_too_many_nans"

    merged, flags = _merge(src, image=image)
    assert "target_srcmask_relaxed" in flags
    new_ann_bad = np.count_nonzero(merged & ann_ring) / ann_ring.sum()
    assert new_ann_bad <= 0.5
    out = _measure_aperture(image, merged)
    assert out["fail_reason"].iloc[0] in ("", None) or pd.isna(
        out["fail_reason"].iloc[0]
    )
    assert np.isfinite(out["flux_AP"].iloc[0])
    assert out["flux_AP"].iloc[0] > 0

    # The unrelated neighbour footprint is untouched.
    assert merged[140, 140]


def test_target_footprint_wings_reopened():
    """The target's own detected component must not carve out its wings
    inside the fit window even when the footprint extends beyond the
    keep disk."""
    image = np.zeros((NY, NX))
    src = _disk(TX, TY, 18.0)  # larger than R_KEEP=16, inside R_FIT=22
    merged, flags = _merge(src, image=image)
    assert "target_srcmask_relaxed" in flags
    # A component pixel between the keep disk and the fit edge is open.
    assert not merged[int(TY), int(TX + 17)]


def test_annulus_relaxation_for_extended_footprint():
    """A target component reaching into the annulus is reopened to the
    measurement edge when the annulus would otherwise be unusable."""
    image = np.zeros((NY, NX))
    src = _disk(TX, TY, 30.0)  # covers ~half the annulus ring
    merged, flags = _merge(src, image=image)
    assert "target_srcmask_relaxed" in flags
    assert "target_annulus_relaxed" in flags
    ann_ring = _annulus_ring()
    assert np.count_nonzero(merged & ann_ring) == 0


def test_unrelated_neighbour_stays_masked():
    """A footprint not touching the target is never reopened."""
    image = np.zeros((NY, NX))
    src = _disk(150.0, 150.0, 12.0)
    merged, flags = _merge(src, image=image)
    assert flags == []
    assert merged[150, 150]
    # ...while the keep disk around the target is clear.
    assert not merged[int(TY), int(TX)]


def test_neighbour_overlapping_annulus_dropped_last_resort():
    """A giant neighbour component that leaves the annulus unusable
    without touching the target is dropped inside the measurement
    region only as a last resort - and the drop must be flagged."""
    image = np.zeros((NY, NX))
    ann_ring = _annulus_ring()
    # ~85% of the annulus ring, leaving a wedge clear on the right.
    # The ring starts at ANN_IN > R_SELF, so it cannot touch the self
    # disk: this component is not target-associated.
    src = ann_ring & ~((XX > TX + 5) & (np.abs(YY - TY) < 10))
    lab, _ = ndi_label(src, structure=np.ones((3, 3), int))
    self_labels = np.unique(lab[_disk(TX, TY, R_SELF)])
    assert np.all(self_labels == 0)  # truly not touching the target
    covered = np.count_nonzero(src & ann_ring) / ann_ring.sum()
    assert covered > 0.5

    merged, flags = _merge(src, image=image)
    assert "target_srcmask_relaxed" not in flags
    assert "target_neighbour_relaxed" in flags
    assert np.count_nonzero(merged & ann_ring) == 0


def test_hardware_defects_never_reopened():
    """Hardware-defect pixels inside the measurement region stay masked
    even when they lie inside the target-associated component."""
    image = np.zeros((NY, NX))
    src = _disk(TX, TY, 20.0)
    hw = np.zeros((NY, NX), dtype=bool)
    hw[int(TY), int(TX + 18)] = True  # inside component, outside keep disk
    merged, flags = _merge(src, hw=hw, image=image)
    assert "target_srcmask_relaxed" in flags
    assert merged[int(TY), int(TX + 18)]


def test_no_source_mask_passes_through():
    """Without detected sources the mask is just the hardware mask with
    the keep disk cleared."""
    image = np.zeros((NY, NX))
    src = np.zeros((NY, NX), dtype=bool)
    hw = np.zeros((NY, NX), dtype=bool)
    hw[50, 50] = True
    merged, flags = _merge(src, hw=hw, image=image)
    assert flags == []
    assert merged[50, 50]
    assert not merged[int(TY), int(TX)]


def test_nonfinite_image_counts_toward_unusable(caplog):
    """NaN pixels count as unusable for the annulus trigger, matching
    the aperture worker's isfinite veto.  The detected footprint is
    inside the keep disk so only the NaNs can trip the fallback - the
    warning firing proves the non-finite pixels were counted."""
    rng = np.random.default_rng(1)
    image = np.zeros((NY, NX))
    ann = _annulus_ring()
    image[ann] = np.where(rng.uniform(size=ann.sum()) < 0.8, np.nan, 1.0)
    src = _disk(TX, TY, 8.0)
    with caplog.at_level("WARNING"):
        merged, flags = _merge(src, image=image)
    assert any(
        "were still masked" in r.getMessage() for r in caplog.records
    )
    # Nothing could be unmasked (footprint was already inside the keep
    # disk), so no relaxation token is recorded; the unusable NaN region
    # is reported separately via target_annulus_mask_frac in the CSV.
    assert flags == []
    assert not merged[int(TY), int(TX)]
