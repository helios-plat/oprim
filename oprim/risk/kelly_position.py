"""Kelly sizing and observed-return portfolio risk calculations.

The historical implementation carried a hard-coded 2% VaR proxy.  This module
now requires explicit observations for loss estimates and never fabricates a
risk number.  Gloomberb (https://github.com/gloom-sh/gloomberb, MIT) was
reviewed as the requested reference; no Gloomberb source code is copied here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


def _validate_probability(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _validate_fraction(value: float, name: str = "fraction") -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def calculate_kelly_criterion(
    win_probability: float,
    win_loss_ratio: float,
    fraction: float = 1.0,
    *,
    max_fraction: float = 1.0,
) -> float:
    """Return a bounded fractional Kelly stake for a binary payoff.

    ``win_loss_ratio`` is net profit per unit lost.  Negative-edge bets return
    zero (no short-bet interpretation); the result is capped by ``max_fraction``.
    """
    probability = _validate_probability(win_probability, "win_probability")
    multiplier = float(win_loss_ratio)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("win_loss_ratio must be finite and positive")
    scale = _validate_fraction(fraction)
    cap = _validate_fraction(max_fraction, "max_fraction")
    full_kelly = (probability * multiplier - (1.0 - probability)) / multiplier
    return min(cap, max(0.0, full_kelly * scale))


def continuous_kelly_fraction(
    expected_return: float,
    variance: float,
    *,
    fraction: float = 1.0,
    max_fraction: float = 1.0,
) -> float:
    """Return continuous Kelly ``mu / variance`` with symmetric cap truncation."""
    mean = float(expected_return)
    variance_value = float(variance)
    if not math.isfinite(mean):
        raise ValueError("expected_return must be finite")
    if not math.isfinite(variance_value) or variance_value <= 0.0:
        raise ValueError("variance must be finite and positive")
    scale = _validate_fraction(fraction)
    cap = _validate_fraction(max_fraction, "max_fraction")
    raw = (mean / variance_value) * scale
    return min(cap, max(-cap, raw))


def fractional_kelly(
    full_kelly_fraction: float,
    fraction: float = 0.5,
    *,
    max_fraction: float = 1.0,
) -> float:
    """Scale an already-computed Kelly fraction and cap its absolute exposure."""
    full = float(full_kelly_fraction)
    if not math.isfinite(full):
        raise ValueError("full_kelly_fraction must be finite")
    scale = _validate_fraction(fraction)
    cap = _validate_fraction(max_fraction, "max_fraction")
    return min(cap, max(-cap, full * scale))


def historical_var(
    returns: Sequence[float], *, confidence: float = 0.95
) -> float:
    """Estimate positive loss VaR from an observed return sequence."""
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    observations = np.asarray(returns, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    return float(-np.quantile(observations, 1.0 - confidence_value))


def parametric_var(
    returns: Sequence[float], *, confidence: float = 0.95
) -> float:
    """Estimate positive loss VaR assuming normally distributed observed returns."""
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    observations = np.asarray(returns, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size < 2:
        raise ValueError("parametric VaR requires at least two finite returns")
    z_score = NormalDist().inv_cdf(confidence_value)
    return float(-observations.mean() + z_score * observations.std(ddof=1))


def historical_cvar(
    returns: Sequence[float], *, confidence: float = 0.95
) -> float:
    """Estimate expected loss in the empirical tail at ``confidence``."""
    observations = np.asarray(returns, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    threshold = historical_var(observations, confidence=confidence)
    losses = -observations
    tail = losses[losses >= threshold]
    return float(tail.mean())


def parametric_cvar(
    returns: Sequence[float], *, confidence: float = 0.95
) -> float:
    """Estimate normal-distribution expected shortfall from observed returns."""
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    observations = np.asarray(returns, dtype=float)
    observations = observations[np.isfinite(observations)]
    if observations.size < 2:
        raise ValueError("parametric CVaR requires at least two finite returns")
    z_score = NormalDist().inv_cdf(confidence_value)
    density = math.exp(-0.5 * z_score * z_score) / math.sqrt(2.0 * math.pi)
    sigma = float(observations.std(ddof=1))
    return float(-observations.mean() + sigma * density / (1.0 - confidence_value))


def maximum_drawdown(equity_curve: Sequence[float]) -> float:
    """Return maximum peak-to-trough drawdown as a fraction of peak equity."""
    values = np.asarray(equity_curve, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("equity_curve must be a non-empty finite one-dimensional series")
    if (values < 0.0).any():
        raise ValueError("equity values must be non-negative")
    peaks = np.maximum.accumulate(values)
    valid = peaks > 0.0
    if not valid.any():
        return 0.0
    drawdowns = np.zeros_like(values)
    drawdowns[valid] = (peaks[valid] - values[valid]) / peaks[valid]
    return float(drawdowns.max())


def calculate_portfolio_var(
    portfolio_value: float,
    positions: list[dict[str, Any]],
    confidence: float = 0.95,
    *,
    returns: Sequence[float] | None = None,
    method: str = "historical",
) -> dict[str, float | str]:
    """Summarize portfolio exposure and loss risk using supplied observations.

    The old implementation guessed VaR as a fixed percentage of exposure.  A
    real return series is now mandatory; absence of data fails closed.
    """
    portfolio = float(portfolio_value)
    if not math.isfinite(portfolio) or portfolio <= 0.0:
        raise ValueError("portfolio_value must be finite and positive")
    if returns is None:
        raise ValueError("returns are required; risk is not inferred from exposure")
    exposure = 0.0
    for position in positions:
        size, price = float(position.get("size", 0.0)), float(position.get("price", 0.0))
        if not math.isfinite(size) or not math.isfinite(price):
            raise ValueError("position size and price must be finite")
        exposure += abs(size * price)
    if method == "historical":
        var_value = historical_var(returns, confidence=confidence)
        cvar_value = historical_cvar(returns, confidence=confidence)
    elif method == "parametric":
        var_value = parametric_var(returns, confidence=confidence)
        cvar_value = parametric_cvar(returns, confidence=confidence)
    else:
        raise ValueError("method must be 'historical' or 'parametric'")
    var_amount = var_value * portfolio
    cvar_amount = cvar_value * portfolio
    return {
        "total_exposure": exposure,
        "portfolio_value": portfolio,
        "var_estimated": var_amount,
        "cvar_estimated": cvar_amount,
        "risk_ratio": var_value,
        "method": method,
    }
