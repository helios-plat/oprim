"""Finance atomic operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


def drawdown_curve(
    equity_or_returns: pd.Series,
    input_type: Literal["equity", "returns"] = "equity",
    compound: bool = True,
) -> dict[str, Any]:
    """Compute drawdown time series and max drawdown statistics.

    drawdown_t = (equity_t - peak_t) / peak_t

    Parameters
    ----------
    equity_or_returns : pd.Series
        Equity curve or returns series.
    input_type : {"equity", "returns"}
        Type of input.
    compound : bool
        Whether to compound returns (only for input_type="returns").

    Returns
    -------
    dict with drawdown_series, max_drawdown, max_drawdown_start/end/recovery, underwater_duration_days.
    """
    if input_type == "returns":
        if compound:
            equity = (1 + equity_or_returns).cumprod()
        else:
            equity = 1 + equity_or_returns.cumsum()
    else:
        equity = equity_or_returns

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    max_dd = float(drawdown.min())

    # Find max drawdown period
    if max_dd == 0:
        return {
            "drawdown_series": drawdown,
            "max_drawdown": 0.0,
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "max_drawdown_recovery": None,
            "underwater_duration_days": 0,
        }

    dd_end_idx = drawdown.idxmin()
    # Start = last peak before dd_end
    peak_before = equity.loc[:dd_end_idx]
    dd_start_idx = peak_before.idxmax()

    # Recovery = first time equity >= peak after dd_end
    peak_val = equity.loc[dd_start_idx]
    after_end = equity.loc[dd_end_idx:]
    recovery_mask = after_end >= peak_val
    recovery_idx = recovery_mask.idxmax() if recovery_mask.any() else None
    if recovery_idx == dd_end_idx and not recovery_mask.iloc[0]:
        recovery_idx = None

    # Underwater duration
    if recovery_idx is not None:
        if isinstance(dd_start_idx, pd.Timestamp):
            underwater = (recovery_idx - dd_start_idx).days
        else:
            underwater = int(recovery_idx - dd_start_idx)
    else:
        if isinstance(dd_start_idx, pd.Timestamp):
            underwater = (equity.index[-1] - dd_start_idx).days
        else:
            underwater = int(len(equity) - 1 - dd_start_idx)

    return {
        "drawdown_series": drawdown,
        "max_drawdown": max_dd,
        "max_drawdown_start": dd_start_idx,
        "max_drawdown_end": dd_end_idx,
        "max_drawdown_recovery": recovery_idx,
        "underwater_duration_days": underwater,
    }


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    annualization_factor: int = 252,
    ddof: int = 1,
) -> float:
    """Compute annualized Sharpe ratio.

    SR = mean(r - r_f) / std(r - r_f) × √annualization_factor

    Parameters
    ----------
    returns : pd.Series
        Return series.
    risk_free_rate : float | pd.Series
        Risk-free rate (scalar or time series).
    annualization_factor : int
        Annualization factor (252 equity, 365 crypto).
    ddof : int
        Degrees of freedom for std.

    Returns
    -------
    float
        Annualized Sharpe ratio. NaN if std == 0.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    excess = returns - risk_free_rate
    excess = excess.dropna()

    std = excess.std(ddof=ddof)
    if std < 1e-15 or np.isnan(std):
        return np.nan

    return float(excess.mean() / std * np.sqrt(annualization_factor))


def beta_alpha_ols(
    asset_returns: pd.Series,
    market_returns: pd.Series | pd.DataFrame,
    use_hac: bool = False,
    hac_lags: int | None = None,
    min_samples: int = 30,
) -> dict[str, Any]:
    """CAPM / multi-factor OLS regression.

    Parameters
    ----------
    asset_returns : pd.Series
        Asset return series.
    market_returns : pd.Series | pd.DataFrame
        Market/factor returns. Series = single factor, DataFrame = multi-factor.
    use_hac : bool
        Use Newey-West HAC standard errors.
    hac_lags : int | None
        Number of lags for HAC. None = auto.
    min_samples : int
        Minimum samples required.

    Returns
    -------
    dict with alpha, beta, alpha_se, beta_se, r_squared, adj_r_squared, n_samples, p_values.
    """
    import statsmodels.api as sm

    if not isinstance(asset_returns, pd.Series):
        asset_returns = pd.Series(asset_returns)

    if isinstance(market_returns, pd.Series):
        market_returns = pd.DataFrame({"market": market_returns})
        single_factor = True
    else:
        single_factor = False

    # Align
    combined = pd.concat([asset_returns.rename("y"), market_returns], axis=1).dropna()
    if len(combined) < min_samples:
        raise ValueError(f"Only {len(combined)} samples, need >= {min_samples}")

    y = combined["y"]
    X = sm.add_constant(combined.drop(columns=["y"]))

    if use_hac:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    else:
        model = sm.OLS(y, X).fit()

    alpha = float(model.params.iloc[0])
    alpha_se = float(model.bse.iloc[0])

    if single_factor:
        beta = float(model.params.iloc[1])
        beta_se = float(model.bse.iloc[1])
        p_values = {"alpha": float(model.pvalues.iloc[0]), "beta": float(model.pvalues.iloc[1])}
    else:
        beta = {col: float(model.params[col]) for col in market_returns.columns}
        beta_se = {col: float(model.bse[col]) for col in market_returns.columns}
        p_values = {"alpha": float(model.pvalues.iloc[0])}
        for col in market_returns.columns:
            p_values[col] = float(model.pvalues[col])

    return {
        "alpha": alpha,
        "beta": beta,
        "alpha_se": alpha_se,
        "beta_se": beta_se,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_samples": len(combined),
        "p_values": p_values,
    }


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: Literal["historical", "parametric", "cornish_fisher"] = "historical",
    include_es: bool = True,
) -> dict[str, float]:
    """Value at Risk (VaR) and Expected Shortfall (ES).

    Parameters
    ----------
    returns : pd.Series
        Return series.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR).
    method : {"historical", "parametric", "cornish_fisher"}
        VaR estimation method.
    include_es : bool
        Whether to compute Expected Shortfall.

    Returns
    -------
    dict with var, es, method, confidence_level.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    returns_clean = returns.dropna()
    alpha = 1 - confidence_level

    if method == "historical":
        var = float(-np.percentile(returns_clean, alpha * 100))
    elif method == "parametric":
        mu = returns_clean.mean()
        sigma = returns_clean.std(ddof=1)
        var = float(-(mu + sigma * stats.norm.ppf(alpha)))
    elif method == "cornish_fisher":
        mu = returns_clean.mean()
        sigma = returns_clean.std(ddof=1)
        s = float(stats.skew(returns_clean, bias=False))
        k = float(stats.kurtosis(returns_clean, fisher=True, bias=False))
        z = stats.norm.ppf(alpha)
        # Cornish-Fisher expansion
        cf_z = (z + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36)
        var = float(-(mu + sigma * cf_z))
    else:
        raise ValueError(f"Unknown method: {method}")

    es = None
    if include_es:
        # ES = expected loss given loss > VaR
        tail = returns_clean[returns_clean <= -var]
        es = float(-tail.mean()) if len(tail) > 0 else var

    return {
        "var": var,
        "es": es,
        "method": method,
        "confidence_level": confidence_level,
    }
