"""Regression tests for input-WCS preservation on degraded/failed solves.

Sparse fields (fewer matched stars than the tweak order requires) must
not let a TAN-only solve-field result strip a usable input distortion
model, and failed merges must restore the input WCS on the *same* header
object the caller holds.
"""

from __future__ import annotations

import pytest
from astropy.io import fits

from wcs import _has_nonzero_distortion, get_wcs

try:
    from functions import copy_wcs_from_header, remove_wcs_from_header
except ImportError:  # photutils / optional deps not installed
    copy_wcs_from_header = None
    remove_wcs_from_header = None

_needs_functions = pytest.mark.skipif(
    remove_wcs_from_header is None,
    reason="functions module not importable in this env",
)


def _base_wcs_header(ctype="RA---TAN"):
    hdr = fits.Header()
    hdr["CRPIX1"] = 525.0
    hdr["CRPIX2"] = 527.0
    hdr["CRVAL1"] = 142.0
    hdr["CRVAL2"] = 14.1
    hdr["CTYPE1"] = ctype
    hdr["CTYPE2"] = "DEC--TPV" if ctype.endswith("TPV") else "DEC--TAN"
    hdr["CD1_1"] = -6.7e-5
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 6.7e-5
    hdr["NAXIS1"] = 1049
    hdr["NAXIS2"] = 1054
    return hdr


def test_tan_header_has_no_distortion():
    assert not _has_nonzero_distortion(_base_wcs_header())


def test_zero_sip_coeffs_are_not_distortion():
    hdr = _base_wcs_header("RA---SIP")
    hdr["CTYPE2"] = "DEC--SIP"
    hdr["A_ORDER"] = 3
    hdr["B_ORDER"] = 3
    hdr["A_0_2"] = 0.0
    hdr["B_2_0"] = 0.0
    assert not _has_nonzero_distortion(hdr)


def test_nonzero_sip_coeff_is_distortion():
    hdr = _base_wcs_header("RA---SIP")
    hdr["CTYPE2"] = "DEC--SIP"
    hdr["A_ORDER"] = 3
    hdr["A_0_2"] = 1.0e-8
    assert _has_nonzero_distortion(hdr)


def test_tpv_pv_coeff_is_distortion():
    hdr = _base_wcs_header("RA---TPV")
    hdr["PV1_2"] = 1.0e-8
    assert _has_nonzero_distortion(hdr)


def test_order_keywords_alone_are_not_distortion():
    hdr = fits.Header()
    hdr["A_ORDER"] = 3
    hdr["B_ORDER"] = 3
    hdr["AP_ORDER"] = 3
    hdr["BP_ORDER"] = 3
    hdr["PV_ORDER"] = 7
    assert not _has_nonzero_distortion(hdr)


def test_none_header_has_no_distortion():
    assert not _has_nonzero_distortion(None)


@_needs_functions
def test_in_place_wcs_restore_preserves_header_identity():
    """The merge-failure restore must mutate self.header in place:
    callers keep a reference to that object, so rebinding would leave
    them holding a WCS-stripped header.
    """
    hdr = _base_wcs_header("RA---TPV")
    hdr["PV1_2"] = 1.0e-8
    backup = hdr.copy()
    caller_ref = hdr

    # Simulate a merge that strips WCS before failing.
    remove_wcs_from_header(hdr)
    assert get_wcs(hdr) is None

    # Restore exactly as plate_solve's _restore_input_wcs_inplace does.
    copy_wcs_from_header(backup, hdr)

    assert caller_ref is hdr
    assert get_wcs(caller_ref) is not None
    assert caller_ref["PV1_2"] == 1.0e-8
    assert caller_ref["CTYPE1"] == "RA---TPV"
    assert _has_nonzero_distortion(caller_ref)
