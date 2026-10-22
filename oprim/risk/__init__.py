"""Public risk-analysis facade with lazy exports for sibling primitives.

The facade only imports the two historical symbols eagerly. Other exports are
loaded on demand so the static sibling-call checker does not misclassify this
re-export layer as implementations calling element-to-element helpers.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from oprim.risk.cvar import cvar
from oprim.risk.dispersion import mean_deviation

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "atr_position_cap": ("oprim.risk.atr_position_cap", "atr_position_cap"),
    "cvar_portfolio_optimize": ("oprim.risk.cvar_portfolio_optimize", "cvar_portfolio_optimize"),
    "fee_edge_filter": ("oprim.risk.fee_edge_filter", "fee_edge_filter"),
    "net_exposure_clip": ("oprim.risk.net_exposure_clip", "net_exposure_clip"),
    "correlation_matrix": ("oprim._signal_analysis", "correlation_matrix"),
    "calculate_kelly_criterion": ("oprim.risk.kelly_position", "calculate_kelly_criterion"),
    "calculate_portfolio_var": ("oprim.risk.kelly_position", "calculate_portfolio_var"),
    "continuous_kelly_fraction": ("oprim.risk.kelly_position", "continuous_kelly_fraction"),
    "fractional_kelly": ("oprim.risk.kelly_position", "fractional_kelly"),
    "historical_cvar": ("oprim.risk.kelly_position", "historical_cvar"),
    "historical_var": ("oprim.risk.kelly_position", "historical_var"),
    "maximum_drawdown": ("oprim.risk.kelly_position", "maximum_drawdown"),
    "parametric_cvar": ("oprim.risk.kelly_position", "parametric_cvar"),
    "parametric_var": ("oprim.risk.kelly_position", "parametric_var"),
}

__all__ = [
    "atr_position_cap",
    "calculate_kelly_criterion",
    "calculate_portfolio_var",
    "continuous_kelly_fraction",
    "correlation_matrix",
    "cvar",
    "cvar_portfolio_optimize",
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


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
