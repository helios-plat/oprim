"""CVaR-optimal portfolio weights via historical mean-risk optimization."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def cvar_portfolio_optimize(
    returns: pd.DataFrame,
    *,
    method_mu: Literal["hist"] = "hist",
    method_cov: Literal["hist"] = "hist",
    rm: Literal["CVaR"] = "CVaR",
    obj: Literal["Sharpe"] = "Sharpe",
    rf: float = 0.0,
    l: float = 0.0,
    hist: bool = True,
    alpha: float = 0.05,
) -> dict:
    """Historical CVaR-based mean-risk portfolio optimization (Riskfolio-Lib).

    Fits a `riskfolio.Portfolio` on historical mean/covariance estimates and
    solves the Classic mean-risk model with CVaR as the risk measure and a
    Sharpe-style risk-adjusted-return objective (long-only, fully-invested).

    This element does NOT catch optimization failures — callers that need a
    fallback (e.g. equal-weight on non-convergence or insufficient data)
    should do so one layer up (oskill), not here.

    Parameters
    ----------
    returns : pd.DataFrame
        Per-asset return series, columns = symbols, rows = time-aligned
        observations (no NaNs — caller must align/dropna first).
    method_mu, method_cov : {"hist"}
        Estimation method for expected returns / covariance. Only "hist"
        (historical) is supported by this wrapper.
    rm : {"CVaR"}
        Risk measure. Only CVaR is exposed by this element.
    obj : {"Sharpe"}
        Optimization objective. Only Sharpe (risk-adjusted return) is exposed.
    rf : float
        Risk-free rate, same frequency as `returns`.
    l : float
        Risk-aversion coefficient (irrelevant for obj="Sharpe", kept for
        Riskfolio-Lib API compatibility).
    hist : bool
        Use the historical scenario matrix directly (not a parametric model).
    alpha : float
        CVaR significance level, e.g. 0.05 for 95% CVaR.

    Returns
    -------
    dict
        ``{"weights": {symbol: float}, "converged": bool}`` — weights sum to 1
        (renormalized to absorb solver rounding).

    Raises
    ------
    ImportError
        If riskfolio-lib is not installed (optional dependency, extra
        ``oprim[portfolio]``).
    ValueError
        If `returns` has fewer than 2 columns, any column is constant/NaN,
        or the solver returns no weights.

    References
    ----------
    .. [1] Cajas, D. Riskfolio-Lib. https://github.com/dcajasn/Riskfolio-Lib
    .. [2] Extraction source: helixa project,
       services/portfolio-optimizer/src/main.py:optimize_cvar
    """
    if returns.shape[1] < 2:
        raise ValueError(f"need >= 2 assets, got {returns.shape[1]}")
    if returns.isna().any().any():
        raise ValueError("returns must not contain NaNs — align/dropna first")

    try:
        import riskfolio as rp
    except ImportError as e:
        raise ImportError(
            "riskfolio-lib is required for cvar_portfolio_optimize "
            "(install extra: oprim[portfolio])"
        ) from e

    port = rp.Portfolio(returns=returns)
    port.alpha = alpha
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    w = port.optimization(model="Classic", rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    if w is None or w.empty:
        raise ValueError("optimizer did not converge (empty weights)")

    weights = {sym: float(w.loc[sym, "weights"]) for sym in returns.columns if sym in w.index}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"optimizer returned non-positive total weight: {total}")
    weights = {k: v / total for k, v in weights.items()}

    return {"weights": weights, "converged": True}
