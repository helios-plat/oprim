"""Tests for oprim.performance.cagr."""
import math
import numpy as np
import pandas as pd
import pytest

from oprim.performance.annualization import cagr


class TestCAGRGeometric:
    def test_cagr_geometric_basic(self):
        """252 bars of 0.001/day ≈ 28.3% CAGR (geometric)."""
        returns = [0.001] * 252
        result = cagr(returns, periods_per_year=252, method="geometric")
        # (1.001)^252 - 1 ≈ 0.2838...
        expected = 1.001**252 - 1
        assert result == pytest.approx(expected, rel=1e-8)

    def test_cagr_arithmetic_basic(self):
        """Arithmetic mean * periods_per_year."""
        returns = [0.001, 0.002, 0.003]
        result = cagr(returns, periods_per_year=252, method="arithmetic")
        expected = np.mean(returns) * 252
        assert result == pytest.approx(expected, rel=1e-12)

    def test_cagr_one_year_data(self):
        """Exactly 252 bars → CAGR = total_return exactly."""
        returns = [0.001] * 252
        result = cagr(returns, periods_per_year=252, method="geometric")
        total_return = math.prod(1.0 + r for r in returns) - 1.0
        assert result == pytest.approx(total_return, rel=1e-8)

    def test_cagr_zero_returns_returns_zero(self):
        """All-zero returns → CAGR = 0."""
        returns = [0.0] * 50
        assert cagr(returns) == pytest.approx(0.0, abs=1e-12)

    def test_cagr_multi_year_compounding(self):
        """2 years (504 bars): verify compound correctly."""
        returns = [0.001] * 504  # ~2 years
        result = cagr(returns, periods_per_year=252, method="geometric")
        # prod = 1.001^504, annualized = prod^(252/504) - 1 = 1.001^252 - 1
        expected = 1.001**252 - 1
        assert result == pytest.approx(expected, rel=1e-8)

    def test_cagr_invalid_periods_per_year(self):
        """periods_per_year=0 → ValueError."""
        with pytest.raises(ValueError, match="periods_per_year"):
            cagr([0.01] * 10, periods_per_year=0)

    def test_cagr_empty_returns_raises(self):
        """Empty returns → ValueError."""
        with pytest.raises(ValueError):
            cagr([])

    @pytest.mark.academic_reference
    def test_cagr_matches_bodie_example(self):
        """Bodie, Kane, Marcus (2014): annualized geometric return formula.

        For 4 quarterly returns of 5% each:
        CAGR = (1.05^4)^(1/1) - 1 = 1.21550625 - 1 = 0.21550625
        with periods_per_year=4, len=4 → (1+r)^(4/4)-1 = same
        """
        returns = [0.05] * 4
        result = cagr(returns, periods_per_year=4, method="geometric")
        expected = 1.05**4 - 1.0  # ~21.55%
        assert result == pytest.approx(expected, rel=1e-8)

    def test_cagr_pandas_series_input(self):
        """pd.Series input works correctly."""
        s = pd.Series([0.01, 0.02, -0.01])
        result = cagr(s, periods_per_year=12)
        expected = cagr([0.01, 0.02, -0.01], periods_per_year=12)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_cagr_negative_periods_per_year_raises(self):
        """periods_per_year < 0 → ValueError."""
        with pytest.raises(ValueError):
            cagr([0.01], periods_per_year=-1)

    def test_cagr_arithmetic_negative_returns(self):
        """Arithmetic CAGR works with negative average."""
        returns = [-0.001] * 252
        result = cagr(returns, periods_per_year=252, method="arithmetic")
        assert result == pytest.approx(-0.001 * 252, rel=1e-12)

    def test_cagr_invalid_method_raises(self):
        """Unknown method → ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            cagr([0.01], method="compound")
