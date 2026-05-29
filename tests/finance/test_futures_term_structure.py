"""Tests for oprim.finance.futures_curve_shape."""

import numpy as np
import pytest

from oprim.finance import futures_curve_shape


class TestFuturesCurveShape:
    def test_contango_shape(self):
        """Prices increase with tenor -> shape='contango'."""
        prices = {1: 100.0, 3: 102.0, 6: 104.0, 12: 107.0}
        result = futures_curve_shape(prices, spot_price=99.0)
        assert result["shape"] == "contango"

    def test_backwardation_shape(self):
        """Prices decrease with tenor -> shape='backwardation'."""
        prices = {1: 98.0, 3: 96.0, 6: 94.0, 12: 90.0}
        result = futures_curve_shape(prices, spot_price=100.0)
        assert result["shape"] == "backwardation"

    def test_flat_shape(self):
        """Near-identical prices -> shape='flat'."""
        prices = {1: 100.0, 6: 100.0}
        result = futures_curve_shape(prices, spot_price=100.0)
        assert result["shape"] == "flat"

    def test_empty_raises(self):
        """Empty prices_by_tenor raises ValueError."""
        with pytest.raises(ValueError):
            futures_curve_shape({}, spot_price=100.0)

    def test_non_positive_spot_raises(self):
        """spot_price <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            futures_curve_shape({1: 100.0}, spot_price=0.0)
        with pytest.raises(ValueError, match="positive"):
            futures_curve_shape({1: 100.0}, spot_price=-5.0)

    def test_returns_all_keys(self):
        """Result must contain shape, slope, curvature, roll_yield_pct."""
        prices = {1: 101.0, 6: 103.0}
        result = futures_curve_shape(prices, spot_price=100.0)
        assert {"shape", "slope", "curvature", "roll_yield_pct"}.issubset(result.keys())

    def test_roll_yield_computation(self):
        """roll_yield_pct = (spot - nearest_future) / spot * (12 / nearest_tenor)."""
        spot = 100.0
        nearest_future = 99.0
        nearest_tenor = 1
        prices = {nearest_tenor: nearest_future, 6: 98.0}
        result = futures_curve_shape(prices, spot_price=spot)
        expected_roll = (spot - nearest_future) / spot * (12 / nearest_tenor)
        assert result["roll_yield_pct"] == pytest.approx(expected_roll, rel=1e-9)

    def test_curvature_with_three_tenors(self):
        """Curvature is non-zero when prices are not linear."""
        prices = {1: 100.0, 6: 102.0, 12: 101.0}  # concave -> negative curvature
        result = futures_curve_shape(prices, spot_price=100.0)
        # curvature = (prices[-1] + prices[0] - 2*prices[mid]) / spot_price
        expected = (101.0 + 100.0 - 2 * 102.0) / 100.0
        assert result["curvature"] == pytest.approx(expected, rel=1e-9)
