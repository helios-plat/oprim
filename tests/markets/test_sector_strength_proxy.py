"""Tests for oprim.markets.sector_strength_proxy (A5)."""

import warnings
import numpy as np
import pytest
from oprim import sector_strength_proxy


class TestSectorStrengthProxy:
    def test_return_mode(self) -> None:
        returns = [0.02, 0.03, -0.01, 0.01, 0.04]
        score = sector_strength_proxy(returns=returns, scoring="return")
        assert 0 <= score <= 100

    def test_volume_adj_mode(self) -> None:
        returns = [0.02, 0.03, -0.01]
        volumes = [1000, 5000, 2000]
        score = sector_strength_proxy(returns=returns, volumes=volumes, scoring="volume_adj_return")
        assert 0 <= score <= 100

    def test_breadth_mode(self) -> None:
        returns = [0.01, 0.02, -0.01, 0.03, -0.02]
        score = sector_strength_proxy(returns=returns, scoring="breadth")
        # 3/5 positive = 60%
        assert score == pytest.approx(60.0)

    def test_empty_constituents_zero(self) -> None:
        score = sector_strength_proxy(returns=[], scoring="return")
        assert score == 0.0

    def test_short_lookback_warn(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sector_strength_proxy(returns=[0.01, 0.02], lookback=3)
            assert len(w) == 1
            assert "short" in str(w[0].message).lower()

    def test_unknown_scoring_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scoring"):
            sector_strength_proxy(returns=[0.01], scoring="invalid")

    def test_volume_adj_zero_volume(self) -> None:
        score = sector_strength_proxy(returns=[0.01, 0.02], volumes=[0, 0], scoring="volume_adj_return")
        assert 0 <= score <= 100
