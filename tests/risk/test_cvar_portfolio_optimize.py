"""Tests for oprim.risk.cvar_portfolio_optimize."""

import numpy as np
import pandas as pd
import pytest

from oprim.risk.cvar_portfolio_optimize import cvar_portfolio_optimize


def _synthetic_returns(n_assets=3, n_obs=300, seed=42):
    rng = np.random.default_rng(seed)
    cols = [f"SYM{i}" for i in range(n_assets)]
    data = rng.normal(0.0003, 0.01, size=(n_obs, n_assets))
    return pd.DataFrame(data, columns=cols)


def test_weights_sum_to_one():
    returns = _synthetic_returns()
    result = cvar_portfolio_optimize(returns)
    assert result["converged"] is True
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_cover_all_symbols():
    returns = _synthetic_returns(n_assets=4)
    result = cvar_portfolio_optimize(returns)
    assert set(result["weights"].keys()) == set(returns.columns)


def test_fewer_than_two_assets_raises():
    returns = _synthetic_returns(n_assets=1)
    with pytest.raises(ValueError):
        cvar_portfolio_optimize(returns)


def test_nan_in_returns_raises():
    returns = _synthetic_returns()
    returns.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError):
        cvar_portfolio_optimize(returns)


def test_custom_alpha_accepted():
    returns = _synthetic_returns()
    result = cvar_portfolio_optimize(returns, alpha=0.10)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_are_non_negative_long_only():
    returns = _synthetic_returns(n_assets=5, seed=7)
    result = cvar_portfolio_optimize(returns)
    assert all(w >= -1e-9 for w in result["weights"].values())
