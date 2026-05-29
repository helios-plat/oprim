"""Tests for oprim.stats.within_group_percentile (A6)."""

import warnings
import numpy as np
import pytest
from oprim import within_group_percentile


class TestWithinGroupPercentile:
    def test_single_group_median(self) -> None:
        values = [1, 2, 3, 4, 5]
        pct = within_group_percentile(values=values, target_idx=2)
        assert 0.4 <= pct <= 0.6  # median-ish

    def test_target_at_extreme(self) -> None:
        values = [1, 2, 3, 4, 100]
        pct = within_group_percentile(values=values, target_idx=4)
        assert pct == 1.0

    def test_target_idx_not_found_raises(self) -> None:
        with pytest.raises(IndexError):
            within_group_percentile(values=[1, 2, 3], target_idx=5)

    def test_group_size_1_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pct = within_group_percentile(values=[42], target_idx=0)
            assert pct == 0.5
            assert len(w) == 1

    def test_rank_vs_interpolate(self) -> None:
        values = [10, 20, 30, 40, 50]
        rank_pct = within_group_percentile(values=values, target_idx=2, method="rank")
        interp_pct = within_group_percentile(values=values, target_idx=2, method="interpolate")
        assert 0 <= rank_pct <= 1
        assert 0 <= interp_pct <= 1

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            within_group_percentile(values=[1, 2, 3], target_idx=1, method="bad")
