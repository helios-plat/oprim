"""Kelly sizing and empirical portfolio-risk tests, including input boundaries."""

import math

import pytest

from oprim._signal_analysis import SignalAnalysisError
from oprim.risk import (
    calculate_kelly_criterion,
    calculate_portfolio_var,
    continuous_kelly_fraction,
    correlation_matrix,
    fractional_kelly,
    historical_cvar,
    historical_var,
    maximum_drawdown,
    parametric_cvar,
    parametric_var,
)

RETURNS = [-0.10, -0.05, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]


def test_discrete_kelly_matches_binary_payoff_formula() -> None:
    assert calculate_kelly_criterion(0.55, 1.5) == pytest.approx(0.25)


def test_discrete_kelly_negative_edge_and_zero_win_probability_are_zero() -> None:
    assert calculate_kelly_criterion(0.2, 1.0) == 0.0
    assert calculate_kelly_criterion(0.0, 2.0) == 0.0


def test_discrete_kelly_caps_exposure_and_fractional_kelly_scales() -> None:
    assert calculate_kelly_criterion(1.0, 10.0, max_fraction=0.2) == 0.2
    assert fractional_kelly(0.8, 0.25, max_fraction=0.3) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("probability", "odds", "fraction", "cap"),
    [(-0.1, 1, 1, 1), (0.5, 0, 1, 1), (0.5, 2, 1.1, 1), (0.5, 2, 1, -0.1)],
)
def test_discrete_kelly_rejects_invalid_boundary_inputs(
    probability: float, odds: float, fraction: float, cap: float
) -> None:
    with pytest.raises(ValueError):
        calculate_kelly_criterion(probability, odds, fraction, max_fraction=cap)


def test_continuous_kelly_signed_fractional_and_symmetric_cap() -> None:
    assert continuous_kelly_fraction(0.02, 0.1) == pytest.approx(0.2)
    assert continuous_kelly_fraction(-0.02, 0.1, fraction=0.5) == pytest.approx(-0.1)
    assert continuous_kelly_fraction(0.2, 0.1, max_fraction=0.3) == 0.3


def test_historical_var_and_cvar_use_observed_left_tail() -> None:
    assert historical_var(RETURNS, confidence=0.8) == pytest.approx(0.026)
    assert historical_cvar(RETURNS, confidence=0.8) == pytest.approx(0.075)


def test_parametric_var_and_cvar_are_finite_and_ordered() -> None:
    var = parametric_var(RETURNS, confidence=0.95)
    cvar = parametric_cvar(RETURNS, confidence=0.95)
    assert math.isfinite(var)
    assert cvar >= var


def test_var_rejects_invalid_confidence_and_empty_data() -> None:
    with pytest.raises(ValueError):
        historical_var(RETURNS, confidence=1.0)
    with pytest.raises(ValueError):
        historical_var([])
    with pytest.raises(ValueError):
        parametric_var([0.01])


def test_maximum_drawdown_peak_to_trough_and_zero_curve_boundaries() -> None:
    assert maximum_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)
    assert maximum_drawdown([0, 0, 0]) == 0.0
    with pytest.raises(ValueError):
        maximum_drawdown([100, -1])


def test_portfolio_risk_requires_observations_and_uses_requested_method() -> None:
    with pytest.raises(ValueError, match="returns are required"):
        calculate_portfolio_var(10_000, [{"size": 1, "price": 2_000}])
    result = calculate_portfolio_var(
        10_000,
        [{"size": 1, "price": 2_000}],
        confidence=0.8,
        returns=RETURNS,
    )
    assert result["total_exposure"] == 2_000
    assert result["var_estimated"] == pytest.approx(260.0)
    assert result["method"] == "historical"


def test_portfolio_risk_parametric_path_and_invalid_portfolio_boundary() -> None:
    result = calculate_portfolio_var(500, [], returns=RETURNS, method="parametric")
    assert result["method"] == "parametric"
    with pytest.raises(ValueError):
        calculate_portfolio_var(0, [], returns=RETURNS)


def test_correlation_matrix_values_and_bad_inputs() -> None:
    corr = correlation_matrix(returns={"A": [1, 2, 3], "B": [2, 4, 6]})
    assert corr["A"]["A"] == pytest.approx(1.0)
    assert corr["A"]["B"] == pytest.approx(1.0)
    with pytest.raises(SignalAnalysisError):
        correlation_matrix(returns={"A": [1, 2]})
    with pytest.raises(SignalAnalysisError):
        correlation_matrix(returns={"A": [1, 2], "B": [1, 1]})
    with pytest.raises(SignalAnalysisError):
        correlation_matrix(returns={"A": [1, 2], "B": [1, 2]}, method="invalid")
