"""Tests for oprim.risk.cvar (Conditional Value at Risk)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from oprim.risk.cvar import cvar


def test_cvar_historical_basic():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, 252)
    result = cvar(returns, alpha=0.05, method="historical")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_cvar_gaussian_basic():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.02, 252)
    result = cvar(returns, alpha=0.05, method="gaussian")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_cvar_historical_matches_manual():
    rng = np.random.default_rng(10)
    returns = rng.normal(0, 0.01, 1000)
    alpha = 0.05
    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    expected = float(-tail.mean())
    result = cvar(returns, alpha=alpha, method="historical")
    assert result == pytest.approx(expected, rel=1e-10)


def test_cvar_gaussian_matches_closed_form():
    rng = np.random.default_rng(20)
    returns = rng.normal(0.001, 0.02, 10000)
    alpha = 0.05
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = stats.norm.ppf(alpha)
    phi = stats.norm.pdf(z)
    expected = -(mu - sigma * phi / alpha)
    result = cvar(returns, alpha=alpha, method="gaussian")
    assert result == pytest.approx(expected, rel=1e-10)


def test_cvar_alpha_zero_raises():
    with pytest.raises(ValueError):
        cvar(np.array([0.01, -0.02, 0.03]), alpha=0.0)


def test_cvar_alpha_one_raises():
    with pytest.raises(ValueError):
        cvar(np.array([0.01, -0.02, 0.03]), alpha=1.0)


def test_cvar_empty_returns_raises():
    with pytest.raises(ValueError):
        cvar(np.array([]))


def test_cvar_invalid_method_raises():
    with pytest.raises(ValueError):
        cvar(np.array([0.01, -0.02, 0.03]), method="unknown")


def test_cvar_accepts_series():
    rng = np.random.default_rng(55)
    s = pd.Series(rng.normal(0, 0.01, 100))
    result = cvar(s, alpha=0.05)
    assert np.isfinite(result)


def test_cvar_positive_cvar_for_negative_skew():
    returns = np.array([-0.05, -0.04, -0.03, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
    result = cvar(returns, alpha=0.3, method="historical")
    # Bottom 30% are the large losses → positive CVaR
    assert result > 0


@pytest.mark.academic_reference
def test_cvar_rockafellar_uryasev_example():
    """Verify CVaR against Rockafellar & Uryasev (2000) closed-form Gaussian CVaR.

    For N(mu, sigma), CVaR_alpha = -(mu - sigma * phi(z_alpha) / alpha).
    Gaussian method is exact (analytical), so we verify using the sample's own
    mu and sigma — which equals the gaussian closed-form exactly.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 1.0, 100_000)
    alpha = 0.05
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = stats.norm.ppf(alpha)
    expected = -(mu - sigma * stats.norm.pdf(z) / alpha)
    result = cvar(returns, alpha=alpha, method="gaussian")
    assert result == pytest.approx(expected, rel=1e-10)
    # Also verify it's close to the theoretical value (2.0628)
    theoretical = stats.norm.pdf(z) / alpha  # ≈ 2.0628 for N(0,1)
    assert abs(result - theoretical) < 0.02


def test_cvar_accepts_list_input():
    """Line 56: cvar with plain list (not ndarray or Series) should work."""
    returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.05, 0.01, 0.02, -0.03, 0.01]
    result = cvar(returns, alpha=0.2)
    assert np.isfinite(result)


def test_cvar_no_tail_samples_uses_threshold():
    """Line 74: quantile threshold above all values → fallback to threshold."""
    # alpha=0.01 on a very small dataset may yield zero tail
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    result = cvar(returns, alpha=0.001, method="historical")
    assert np.isfinite(result)
