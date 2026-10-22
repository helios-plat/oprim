"""Risk primitives submodule. Exports legacy statistics and read-only TV risk helpers."""

from oprim._signal_analysis import correlation_matrix
from oprim.risk.atr_position_cap import atr_position_cap
from oprim.risk.cvar import cvar
from oprim.risk.cvar_portfolio_optimize import cvar_portfolio_optimize
from oprim.risk.dispersion import mean_deviation
from oprim.risk.fee_edge_filter import fee_edge_filter
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
from oprim.risk.net_exposure_clip import net_exposure_clip

__all__ = [
    "atr_position_cap",
    "correlation_matrix",
    "cvar",
    "cvar_portfolio_optimize",
    "calculate_kelly_criterion",
    "calculate_portfolio_var",
    "continuous_kelly_fraction",
    "fee_edge_filter",
    "fractional_kelly",
    "historical_cvar",
    "historical_var",
    "maximum_drawdown",
    "mean_deviation",
    "net_exposure_clip",
    "parametric_cvar",
    "parametric_var",
]
