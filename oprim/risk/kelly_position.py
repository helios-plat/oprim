"""Gloomberb Kelly position & portfolio risk calculation.
MIT License - adapted from Gloomberb.
"""
from __future__ import annotations
from typing import Any

def calculate_kelly_criterion(win_probability: float, win_loss_ratio: float, fraction: float = 1.0) -> float:
    """Calculate Kelly fraction: f* = (p * b - (1 - p)) / b."""
    if win_loss_ratio <= 0:
        return 0.0
    f_star = (win_probability * win_loss_ratio - (1.0 - win_probability)) / win_loss_ratio
    return max(0.0, f_star * fraction)

def calculate_portfolio_var(portfolio_value: float, positions: list[dict[str, Any]], confidence: float = 0.95) -> dict[str, float]:
    """Calculate simplified portfolio Value at Risk."""
    total_exposure = sum(p.get('size', 0) * p.get('price', 0) for p in positions)
    # Simplified parametric VaR proxy
    var_95 = total_exposure * 0.02 * (1.0 if confidence == 0.95 else 1.5)
    return {
        "total_exposure": total_exposure,
        "portfolio_value": portfolio_value,
        "var_estimated": var_95,
        "risk_ratio": var_95 / portfolio_value if portfolio_value > 0 else 0.0
    }
