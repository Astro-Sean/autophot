"""Target-ban helper for subtraction priors.

A matched pair at the configured target position is transient-plus-host
flux; it must never seed the SFFT/HOTPANTS/ZOGY kernel fit, flux scale,
or stamp lists.  ``_sources_far_from_positions`` vetoes only positions
measurably close to a banned (x, y) - non-finite rows are not dropped
as a side effect.
"""

import numpy as np

from autophot.pipeline.main import _sources_far_from_positions


def test_drops_row_within_radius():
    xy = np.array([[100.0, 100.0], [400.0, 400.0], [103.0, 100.0]])
    keep = _sources_far_from_positions(xy, [[100.0, 100.0]], 5.0)
    np.testing.assert_array_equal(keep, [False, True, False])


def test_boundary_radius_is_exclusive_at_zero():
    # dist == radius still counts as inside the ban.
    keep = _sources_far_from_positions([[110.0, 0.0]], [[100.0, 0.0]], 10.0)
    assert not keep[0]
    keep = _sources_far_from_positions(
        [[110.1, 0.0]], [[100.0, 0.0]], 10.0
    )
    assert keep[0]


def test_empty_ban_list_keeps_all():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        _sources_far_from_positions(xy, [], 5.0), [True, True]
    )
    np.testing.assert_array_equal(
        _sources_far_from_positions(xy, [[np.nan, np.nan]], 5.0),
        [True, True],
    )


def test_nonfinite_rows_kept():
    # A NaN position cannot be verified near the target, so it is not
    # dropped by this helper (other filters own finite-coordinate cuts).
    xy = np.array([[np.nan, 50.0], [100.0, 100.0]])
    keep = _sources_far_from_positions(xy, [[100.0, 100.0]], 5.0)
    np.testing.assert_array_equal(keep, [True, False])


def test_multiple_ban_positions():
    xy = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
    keep = _sources_far_from_positions(
        xy, [[100.0, 100.0], [300.0, 300.0]], 5.0
    )
    np.testing.assert_array_equal(keep, [False, True, False])


def test_empty_input():
    keep = _sources_far_from_positions(
        np.empty((0, 2)), [[1.0, 1.0]], 5.0
    )
    assert keep.dtype == bool and len(keep) == 0


def test_flat_input_reshaped():
    keep = _sources_far_from_positions(
        [100.0, 100.0, 500.0, 500.0], [[100.0, 100.0]], 5.0
    )
    np.testing.assert_array_equal(keep, [False, True])
