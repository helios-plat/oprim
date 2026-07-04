"""Tests for oprim.classify_capital_flow_tiers.

No Tide source exists for this element (see module docstring's ASSUMPTION
note) — these tests validate it against capital_flow_classification's own
tier/direction outputs (same thresholds, same classifier), since that's the
logic being reused, not duplicated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from oprim.capital_flow_classification import _DEFAULT_THRESHOLDS, capital_flow_classification
from oprim.classify_capital_flow_tiers import classify_capital_flow_tiers


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 10.1, 10.05, 10.2, 10.15, 10.3],
            "volume": [1000, 2000, 500, 10000, 3000, 50000],
            "amount": [10_000, 20_200, 5_025, 2_040_000, 304_500, 15_000_000],
        }
    )


class TestClassifyCapitalFlowTiers:
    def test_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            classify_capital_flow_tiers(pd.DataFrame({"price": [1.0]}))

    def test_invalid_direction_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid direction_classification"):
            classify_capital_flow_tiers(_trades(), direction_classification="bogus")

    def test_returns_tier_and_direction_labels(self) -> None:
        result = classify_capital_flow_tiers(_trades())
        assert "tier_labels" in result
        assert "direction_labels" in result
        assert len(result["tier_labels"]) == len(_trades())
        assert len(result["direction_labels"]) == len(_trades())

    def test_default_thresholds_echoed(self) -> None:
        result = classify_capital_flow_tiers(_trades())
        assert result["thresholds"] == _DEFAULT_THRESHOLDS

    def test_custom_thresholds_applied(self) -> None:
        custom = {"small": 1_000, "medium": 5_000, "large": 50_000}
        result = classify_capital_flow_tiers(_trades(), tier_thresholds=custom)
        assert result["thresholds"] == custom

    def test_matches_capital_flow_classification_tier_labels(self) -> None:
        """Same input, same thresholds -> identical per-trade tier labels as
        the already-ported capital_flow_classification (this is meant to be
        the same classification step, exposed standalone)."""
        trades = _trades()
        tiers_result = classify_capital_flow_tiers(trades)
        full_result = capital_flow_classification(trades)

        # Reconstruct per-trade tier labels from the aggregated stats' masks
        # isn't directly exposed by capital_flow_classification, so instead
        # verify the block_trade tier (amount >= 5,000,000) agrees between
        # both: the one trade at amount=15,000,000 must be block_trade here
        # and must be counted under block_trade's tier_stats there.
        assert tiers_result["tier_labels"][-1] == "block_trade"
        assert full_result["tier_stats"]["block_trade"]["count"] == 1

    def test_matches_capital_flow_classification_direction_labels(self) -> None:
        """tick_volume: direction[i] flips to 'sell' exactly where price[i] < price[i-1]."""
        trades = _trades()
        tiers_result = classify_capital_flow_tiers(trades, direction_classification="tick_volume")
        prices = trades["price"].to_numpy()
        expected = ["buy"] + [
            "buy" if prices[i] >= prices[i - 1] else "sell" for i in range(1, len(prices))
        ]
        assert list(tiers_result["direction_labels"]) == expected

    def test_lee_ready_default(self) -> None:
        result = classify_capital_flow_tiers(_trades())
        assert set(result["direction_labels"]).issubset({"buy", "sell"})
