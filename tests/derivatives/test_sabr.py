from __future__ import annotations

import numpy as np
import pytest

from oprim.derivatives.sabr import sabr_implied_volatility


def test_sabr_atm_basic():
    r = sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=0.5, rho=0.0, nu=0.3)
    assert 0.01 < r["implied_volatility"] < 2.0


def test_sabr_atm_flag():
    r = sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=0.5)
    assert r["is_atm"] is True


def test_sabr_otm_not_atm():
    r = sabr_implied_volatility(100.0, 120.0, 1.0, alpha=0.2, beta=0.5)
    assert r["is_atm"] is False


def test_sabr_smile_skew_with_rho():
    r_low = sabr_implied_volatility(100.0, 80.0, 1.0, alpha=0.2, beta=0.5, rho=-0.5, nu=0.5)
    r_high = sabr_implied_volatility(100.0, 120.0, 1.0, alpha=0.2, beta=0.5, rho=-0.5, nu=0.5)
    assert r_low["implied_volatility"] > r_high["implied_volatility"]


def test_sabr_nu_zero_no_smile():
    r_low = sabr_implied_volatility(100.0, 80.0, 1.0, alpha=0.2, beta=0.5, rho=0.0, nu=0.0)
    r_atm = sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=0.5, rho=0.0, nu=0.0)
    ratio = r_low["implied_volatility"] / r_atm["implied_volatility"]
    assert 0.5 < ratio < 2.0


def test_sabr_beta_one():
    r = sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=1.0, rho=0.0, nu=0.0)
    np.testing.assert_allclose(r["implied_volatility"], 0.2, rtol=0.1)


def test_sabr_positive_iv():
    r = sabr_implied_volatility(100.0, 90.0, 0.5, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
    assert r["implied_volatility"] > 0


def test_sabr_formula_hagan_2002():
    r = sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=0.5, formula="hagan_2002")
    assert r["formula"] == "hagan_2002"
    assert r["implied_volatility"] > 0


def test_sabr_invalid_params_raises():
    with pytest.raises(ValueError):
        sabr_implied_volatility(-1.0, 100.0, 1.0, alpha=0.2, beta=0.5)
    with pytest.raises(ValueError):
        sabr_implied_volatility(100.0, 100.0, 1.0, alpha=0.2, beta=1.5)


def test_sabr_time_scaling():
    r_short = sabr_implied_volatility(100.0, 100.0, 0.1, alpha=0.2, beta=0.5, nu=0.5)
    r_long = sabr_implied_volatility(100.0, 100.0, 2.0, alpha=0.2, beta=0.5, nu=0.5)
    assert r_long["implied_volatility"] != r_short["implied_volatility"]


@pytest.mark.academic_reference
def test_sabr_hagan_2002_atm_formula():
    F, K, T = 100.0, 100.0, 1.0
    alpha, beta = 0.2, 0.5
    expected_atm = alpha / (F ** (1 - beta))
    r = sabr_implied_volatility(F, K, T, alpha=alpha, beta=beta, rho=0.0, nu=0.0)
    np.testing.assert_allclose(r["implied_volatility"], expected_atm, rtol=0.01)
