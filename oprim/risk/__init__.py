"""Risk primitives submodule."""

from oprim._signal_analysis import correlation_matrix
from oprim.risk.cvar import cvar
from oprim.risk.dispersion import mean_deviation
from oprim.risk.kelly_position import (
    calculate_kelly_criterion,
    calculate_portfolio_var,
    continuous_kelly_fraction,
    fractional_kelly,
    historical_cvar,
    historical_var,
    maximum_drawdown,
    parametric_cvar,
    parametric_var,
)

__all__ = [
    "cvar",
    "mean_deviation",
    "calculate_kelly_criterion",
    "calculate_portfolio_var",
    "continuous_kelly_fraction",
    "fractional_kelly",
    "historical_cvar",
    "historical_var",
    "maximum_drawdown",
    "parametric_cvar",
    "parametric_var",
    "correlation_matrix",
]
