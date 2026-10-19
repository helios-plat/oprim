"""Tests for oprim.math_currency_round/math_calc_percentage/math_tax_multiplier."""

from __future__ import annotations

from oprim.math_calc_percentage import math_calc_percentage
from oprim.math_currency_round import math_currency_round
from oprim.math_tax_multiplier import math_tax_multiplier


class TestMathCurrencyRound:
    def test_rounds_up_at_half(self):
        assert math_currency_round(19.995) == 2000

    def test_rounds_down_below_half(self):
        assert math_currency_round(19.994) == 1999

    def test_default_decimals_is_two(self):
        assert math_currency_round(1.0) == 100

    def test_custom_decimals(self):
        assert math_currency_round(1.2345, decimals=3) == 1235

    def test_zero_amount(self):
        assert math_currency_round(0.0) == 0

    def test_avoids_float_binary_imprecision(self):
        # 0.1 + 0.2 == 0.30000000000000004 in raw float arithmetic.
        assert math_currency_round(0.1 + 0.2) == 30


class TestMathCalcPercentage:
    def test_basic_percentage(self):
        assert math_calc_percentage(1000, percent=8.5) == 85.0

    def test_zero_percent(self):
        assert math_calc_percentage(1000, percent=0) == 0.0

    def test_hundred_percent(self):
        assert math_calc_percentage(1000, percent=100) == 1000.0

    def test_zero_base(self):
        assert math_calc_percentage(0, percent=50) == 0.0


class TestMathTaxMultiplier:
    def test_basic_rate(self):
        assert math_tax_multiplier(8.5) == 1.085

    def test_zero_rate(self):
        assert math_tax_multiplier(0) == 1.0

    def test_hundred_percent_rate(self):
        assert math_tax_multiplier(100) == 2.0
