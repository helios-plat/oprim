"""Tests for oprim.finance.nelson_siegel_yield_curve."""

import numpy as np
import pytest

from oprim.finance import nelson_siegel_yield_curve


class TestNelsonSiegelYieldCurve:
    def _standard_tenors(self):
        return np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

    def test_returns_keys(self):
        """Result must contain expected keys."""
        tenors = self._standard_tenors()
        yields = np.array([0.02, 0.021, 0.022, 0.024, 0.025, 0.027, 0.028, 0.03])
        result = nelson_siegel_yield_curve(tenors, yields)
        expected_keys = {"beta_0", "beta_1", "beta_2", "lambda", "fitted_yields", "residuals", "r_squared"}
        assert expected_keys.issubset(result.keys())

    def test_flat_curve(self):
        """All yields equal -> beta_1 and beta_2 near zero."""
        tenors = self._standard_tenors()
        level = 0.03
        yields = np.full(len(tenors), level)
        result = nelson_siegel_yield_curve(tenors, yields)
        assert abs(result["beta_1"]) < 0.1
        assert abs(result["beta_2"]) < 0.1

    def test_upward_sloping(self):
        """Short rates < long rates -> beta_1 < 0 (short-term factor is negative for upward slope)."""
        tenors = self._standard_tenors()
        yields = np.array([0.01, 0.015, 0.02, 0.025, 0.027, 0.03, 0.032, 0.035])
        result = nelson_siegel_yield_curve(tenors, yields)
        # beta_0 is long-term level; fitted curve should be increasing
        assert result["fitted_yields"][-1] > result["fitted_yields"][0]

    def test_r_squared_range(self):
        """r_squared must be in [0, 1]."""
        tenors = self._standard_tenors()
        yields = np.array([0.02, 0.021, 0.022, 0.024, 0.025, 0.027, 0.028, 0.03])
        result = nelson_siegel_yield_curve(tenors, yields)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_residuals_length(self):
        """Residuals must have same length as tenors."""
        tenors = self._standard_tenors()
        yields = np.array([0.02, 0.021, 0.022, 0.024, 0.025, 0.027, 0.028, 0.03])
        result = nelson_siegel_yield_curve(tenors, yields)
        assert len(result["residuals"]) == len(tenors)

    def test_insufficient_data_raises(self):
        """len(tenors) < 4 raises ValueError."""
        with pytest.raises(ValueError):
            nelson_siegel_yield_curve(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.02, 0.025, 0.03]),
            )

    def test_length_mismatch_raises(self):
        """tenors and yields of different length raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            nelson_siegel_yield_curve(
                np.array([1.0, 2.0, 5.0, 10.0]),
                np.array([0.02, 0.025]),
            )

    @pytest.mark.academic_reference
    def test_nelson_siegel_1987(self):
        """Nelson & Siegel (1987): beta_0 is the long-run asymptotic level.

        As tenor -> inf, the NS model converges to beta_0.
        We verify fitted_yields[-1] is close to beta_0 for a long tenor.
        """
        # Use a wide range of tenors including very long maturity
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        yields = np.array([0.02, 0.022, 0.025, 0.028, 0.031, 0.033, 0.034, 0.035])
        result = nelson_siegel_yield_curve(tenors, yields)
        # The long end of the fitted curve should be close to beta_0
        beta_0 = result["beta_0"]
        fitted_last = result["fitted_yields"][-1]
        # At 30y tenor, the loading factors (1-exp(-t/lam))/(t/lam) approach 0 -> y -> beta_0
        assert abs(fitted_last - beta_0) < 0.02
