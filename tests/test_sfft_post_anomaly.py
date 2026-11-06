"""Regression tests for SFFT post-anomaly bookkeeping and matching rescue.

Sparse fields produced meaningless anomaly statistics: the flag-column
heuristic accepted continuous PAC columns (RATIO/DFSUM_PostAnomaly) when
they had few unique values, so nearly every subsource was exported as an
anomaly and the retry feedback fired on clean subtractions.  The export
must select only the genuine 0/1 ``MASK_PostAnomaly`` column and skip
prior-banned subsources.

``_ranked_keep_mask`` backs the starved-prior rescue in the matching
funnel: when a profile-concentration or elongation cut would leave too
few sources for the SFFT kernel, the best-ranked candidates are
retained instead of applying the hard threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions import _sfft_match_bad_mask, _sfft_post_anomaly_retry_viable
from utils.run_sfft import (
    _adopt_prior_free_retry,
    _coords_from_prep_data,
    _diff_source_residual_scale,
    _flag_mask_from_columns,
    _write_post_anomaly_sources_csv,
    _xy_from_sfft_catalog,
)

try:
    from main import (
        _additional_target_sfft_ban_xy,
        _header_numeric_value,
        _ranked_keep_mask,
    )
except ImportError:  # optional pipeline deps not installed
    _ranked_keep_mask = None
    _header_numeric_value = None
    _additional_target_sfft_ban_xy = None

_needs_main = pytest.mark.skipif(
    _ranked_keep_mask is None, reason="main.py deps not installed"
)


def _subsource_df(mask=(False, True, False, False), ban=None):
    n = len(mask)
    df = pd.DataFrame(
        {
            "X_IMAGE_REF_SCI_MEAN": np.arange(10.0, 10.0 + n),
            "Y_IMAGE_REF_SCI_MEAN": np.arange(20.0, 20.0 + n),
            "RATIO_PostAnomaly": np.linspace(0.2, 0.9, n),
            "DFSUM_PostAnomaly": np.linspace(100.0, 200.0, n),
            "MASK_PostAnomaly": np.asarray(mask, dtype=bool),
        }
    )
    if ban is not None:
        df["MASK_PriorBan"] = np.asarray(ban, dtype=bool)
    return df


class TestFlagMaskFromColumns:
    def test_selects_boolean_post_anomaly_mask(self):
        df = _subsource_df()
        mask = _flag_mask_from_columns(
            df, include_tokens=("post", "anom"), exclude_tokens=("prior",)
        )
        assert mask.tolist() == [False, True, False, False]

    def test_continuous_columns_not_flag_like(self):
        # A 4-row table gives RATIO/DFSUM <= 6 unique values - exactly the
        # sparse-field case that used to be misread as flags.
        df = _subsource_df().drop(columns=["MASK_PostAnomaly"])
        mask = _flag_mask_from_columns(
            df, include_tokens=("post", "anom"), exclude_tokens=("prior",)
        )
        assert not mask.any()

    def test_prior_columns_excluded_by_token(self):
        df = _subsource_df(ban=[False, False, True, False])
        mask = _flag_mask_from_columns(
            df, include_tokens=("post", "anom"), exclude_tokens=("prior",)
        )
        # MASK_PriorBan must not satisfy include/exclude token selection.
        assert mask.tolist() == [False, True, False, False]

    def test_empty_df_returns_empty(self):
        mask = _flag_mask_from_columns(
            pd.DataFrame(), include_tokens=("post", "anom")
        )
        assert mask.size == 0


class TestCoordsFromPrepData:
    def test_reads_post_anomaly_key_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        xy = _coords_from_prep_data({"PostAnomaly-XY": arr})
        assert xy.shape == (2, 2)

    def test_subsource_table_is_not_a_coord_key(self):
        df = _subsource_df()
        xy = _coords_from_prep_data({"SExCatalog-SubSource": df})
        assert xy.size == 0


class TestRetrySourceMetrics:
    def test_xy_catalog_coordinates_are_zero_based(self):
        df = pd.DataFrame(
            {
                "X_IMAGE_REF_SCI_MEAN": [10.0, np.nan],
                "Y_IMAGE_REF_SCI_MEAN": [20.0, 30.0],
            }
        )
        xy = _xy_from_sfft_catalog(df)
        assert xy.tolist() == [[9.0, 19.0]]

    def test_source_residual_prefers_clean_diff(self):
        clean = np.zeros((20, 20))
        dirty = clean.copy()
        dirty[8:12, 8:12] = 5.0
        xy = np.array([[10.0, 10.0], [4.0, 4.0]])
        clean_score = _diff_source_residual_scale(clean, xy, radius=2)
        dirty_score = _diff_source_residual_scale(dirty, xy, radius=2)
        assert clean_score == 0.0
        assert dirty_score > clean_score

    def test_retry_adopted_when_source_count_and_residuals_support_it(self):
        assert _adopt_prior_free_retry(127, 4, 10, 100.0, 90.0)
        assert not _adopt_prior_free_retry(127, 4, 10, 100.0, 120.0)

    def test_sparse_retry_needs_clear_residual_gain(self):
        assert _adopt_prior_free_retry(7, 4, 10, 100.0, 75.0)
        assert not _adopt_prior_free_retry(7, 4, 10, 100.0, 90.0)
        assert not _adopt_prior_free_retry(7, 4, 10, 100.0, None)

    def test_self_match_threshold_covers_sparse_prior_target(self):
        cfg = yaml.safe_load((ROOT / "databases/default_input.yml").read_text())
        ts_cfg = cfg["default_input"]["template_subtraction"]
        assert (
            ts_cfg["sfft_self_match_min_sources"]
            >= ts_cfg["sfft_sparse_prior_target"]
        )

    @_needs_main
    def test_additional_targets_enter_sfft_prior_ban(self):
        class _Wcs:
            def all_world2pix(self, ra, dec, origin):
                return ra + 1.0, dec + 1.0

        cfg = {
            "_additional_targets_resolved": [
                {"ra": 10.0, "dec": 20.0},
                {"ra": 200.0, "dec": 20.0},
            ]
        }
        xy = _additional_target_sfft_ban_xy(
            cfg, _Wcs(), (100, 100), 2.0, 0
        )
        assert xy == [[12.0, 22.0]]


class TestSfftMatchBadMask:
    def test_catalog_source_mask_does_not_starve_priors(self):
        sci = np.ones((8, 8))
        ref = np.ones((8, 8))
        defects = np.zeros((8, 8), dtype=bool)
        defects[3:6, 3:6] = True  # catalog source footprint
        hardware = np.zeros((8, 8), dtype=bool)
        hardware[0, 0] = True
        mask = _sfft_match_bad_mask(
            sci, ref, hardware_defects_mask=hardware
        )
        assert mask.sum() == 1
        assert mask[0, 0]
        assert not mask[4, 4]

    def test_only_invalid_pixels_count_without_hardware_mask(self):
        sci = np.ones((8, 8))
        ref = np.ones((8, 8))
        sci[2, 2] = np.nan
        mask = _sfft_match_bad_mask(sci, ref)
        assert mask.sum() == 1
        assert mask[2, 2]


class TestPostAnomalyRetryViable:
    """Veto for retries whose anomaly bans remove the kernel constraints.

    SN2024pba 2024-12-26 g: 3 vetted SubSources, 2 flagged PostAnomaly.
    Banning the anomaly positions left 1 SubSource, so the retry died at
    run_sfft's matched-source gate after a doomed subprocess call.
    """

    def test_vetoes_when_bans_eat_constraints_and_priors_thin(self):
        # surviving=1 < min=3 and only min priors -> veto.
        assert not _sfft_post_anomaly_retry_viable(1, True, 3, 3)

    def test_vetoes_when_no_vetted_survive(self):
        assert not _sfft_post_anomaly_retry_viable(0, True, 2, 3)

    def test_proceeds_when_enough_vetted_survive(self):
        assert _sfft_post_anomaly_retry_viable(4, True, 3, 3)

    def test_proceeds_with_large_prior_pool(self):
        # Thin vetted survivors but many priors - SFFT can still match
        # new SubSources.
        assert _sfft_post_anomaly_retry_viable(1, True, 10, 3)

    def test_proceeds_without_vetted_list(self):
        # No SFFT_Matching_Sources CSV -> cannot predict; keep old gate.
        assert _sfft_post_anomaly_retry_viable(0, False, 3, 3)


class TestWritePostAnomalyCsv:
    def test_exports_only_flagged_nonbanned(self, tmp_path):
        df = _subsource_df(
            mask=[False, True, True, True],
            ban=[False, False, True, False],
        )
        n = _write_post_anomaly_sources_csv(
            {"SExCatalog-SubSource": df}, None, str(tmp_path), "base"
        )
        assert n == 2
        out = pd.read_csv(tmp_path / "SFFT_PostAnomaly_Sources_base.csv")
        assert out["X_IMAGE_REF_SCI_MEAN"].tolist() == [11.0, 13.0]
        assert out["Y_IMAGE_REF_SCI_MEAN"].tolist() == [21.0, 23.0]

    def test_no_anomalies_returns_zero_and_clears_stale(self, tmp_path):
        stale = tmp_path / "SFFT_PostAnomaly_Sources_base.csv"
        stale.write_text("x,y\n1,2\n")
        df = _subsource_df(mask=[False] * 4)
        n = _write_post_anomaly_sources_csv(
            {"SExCatalog-SubSource": df}, None, str(tmp_path), "base"
        )
        assert n == 0
        assert not stale.exists()

    def test_missing_mask_column_returns_zero(self, tmp_path):
        df = _subsource_df().drop(columns=["MASK_PostAnomaly"])
        n = _write_post_anomaly_sources_csv(
            {"SExCatalog-SubSource": df}, None, str(tmp_path), "base"
        )
        assert n == 0
        assert not (tmp_path / "SFFT_PostAnomaly_Sources_base.csv").exists()

    def test_falls_back_to_matched_df(self, tmp_path):
        df = _subsource_df(mask=[True, False, False, False])
        n = _write_post_anomaly_sources_csv(
            {}, df, str(tmp_path), "base"
        )
        assert n == 1


@_needs_main
class TestRankedKeepMask:
    def _df(self, vals):
        return pd.DataFrame({"v": vals})

    def test_keeps_largest(self):
        df = self._df([0.9, 0.1, 0.8, 0.5, 0.95])
        valid = df["v"].notna()
        keep = _ranked_keep_mask(df, df["v"], valid, 2, keep_largest=True)
        assert keep.tolist() == [True, False, False, False, True]

    def test_keeps_smallest(self):
        df = self._df([1.1, 2.0, 1.2, 1.9, 1.05])
        valid = df["v"].notna()
        keep = _ranked_keep_mask(df, df["v"], valid, 3, keep_largest=False)
        assert keep.tolist() == [True, False, True, False, True]

    def test_nonfinite_rows_always_kept(self):
        df = self._df([1.1, np.nan, 2.0])
        valid = df["v"].notna()
        keep = _ranked_keep_mask(df, df["v"], valid, 1, keep_largest=False)
        assert keep.tolist() == [True, True, False]

    def test_keep_n_above_finite_count_keeps_all(self):
        df = self._df([0.1, 0.2, 0.3])
        valid = df["v"].notna()
        keep = _ranked_keep_mask(df, df["v"], valid, 10, keep_largest=True)
        assert keep.all()


@_needs_main
class TestHeaderNumericValue:
    """MaxIm-style junk strings in numeric metadata keywords must not
    propagate - a string readnoise reached `readnoise > 0` and crashed."""

    def test_keyword_numeric_value(self):
        header = {"RDNOISE": 8.2}
        assert _header_numeric_value("RDNOISE", header) == 8.2

    def test_keyword_numeric_string(self):
        header = {"RDNOISE": "8.2"}
        assert _header_numeric_value("RDNOISE", header) == 8.2

    def test_keyword_junk_string_returns_none(self):
        header = {"RDNOISE": "MaxIm DL Version 5.24"}
        assert _header_numeric_value("RDNOISE", header) is None

    def test_missing_keyword_returns_none(self):
        assert _header_numeric_value("RDNOISE", {}) is None

    def test_numeric_literal_entry_used_directly(self):
        # telescope.yml can store a value instead of a keyword name
        assert _header_numeric_value(8.2, {}) == 8.2
        assert _header_numeric_value("8.2", {}) == 8.2

    def test_numeric_literal_beats_same_named_keyword(self):
        # '8.2' cannot be a FITS keyword, but a float entry must not be
        # looked up as one
        header = {"GAIN": 1.5}
        assert _header_numeric_value(2.0, header) == 2.0

    def test_not_given_by_user_returns_none(self):
        header = {"RDNOISE": 8.2}
        assert _header_numeric_value("not_given_by_user", header) is None

    def test_none_and_empty_return_none(self):
        assert _header_numeric_value(None, {"RDNOISE": 1.0}) is None
        assert _header_numeric_value("", {"RDNOISE": 1.0}) is None
