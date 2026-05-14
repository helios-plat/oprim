from __future__ import annotations

import numpy as np
import pytest

from oprim.volatility.rough import rough_volatility_simulate


def test_rough_vol_basic():
    r = rough_volatility_simulate(100.0, n_paths=100, n_time_steps=50, seed=42)
    assert "paths" in r and r["paths"].shape == (100, 51)


def test_rough_vol_paths_shape():
    r = rough_volatility_simulate(100.0, n_paths=50, n_time_steps=30, seed=0)
    assert r["variance_paths"].shape == (50, 31)


def test_rough_vol_positive_prices():
    r = rough_volatility_simulate(100.0, n_paths=200, n_time_steps=50, seed=1)
    assert np.all(r["paths"] > 0)


def test_rough_vol_positive_variance():
    r = rough_volatility_simulate(100.0, n_paths=100, n_time_steps=50, seed=2)
    assert np.all(r["variance_paths"] > 0)


def test_rough_vol_time_grid():
    r = rough_volatility_simulate(100.0, n_paths=10, n_time_steps=20, seed=3, time_to_expiry=0.5)
    assert len(r["time_grid"]) == 21
    assert abs(r["time_grid"][-1] - 0.5) < 1e-10


def test_rough_vol_initial_price():
    r = rough_volatility_simulate(200.0, n_paths=50, n_time_steps=20, seed=4)
    np.testing.assert_allclose(r["paths"][:, 0], 200.0)


def test_rough_vol_invalid_hurst_raises():
    with pytest.raises(ValueError):
        rough_volatility_simulate(100.0, hurst=0.5)
    with pytest.raises(ValueError):
        rough_volatility_simulate(100.0, hurst=0.6)


def test_rough_vol_seed_reproducible():
    r1 = rough_volatility_simulate(100.0, n_paths=50, n_time_steps=30, seed=99)
    r2 = rough_volatility_simulate(100.0, n_paths=50, n_time_steps=30, seed=99)
    np.testing.assert_array_equal(r1["paths"], r2["paths"])


def test_rough_vol_realized_vol_distribution():
    r = rough_volatility_simulate(100.0, n_paths=200, n_time_steps=50, seed=5)
    rvol = r["realized_vol_distribution"]
    assert len(rvol) == 200
    assert np.all(rvol >= 0)


def test_rough_heston_model():
    r = rough_volatility_simulate(100.0, model="rough_heston", n_paths=50, n_time_steps=30, seed=6)
    assert r["model"] == "rough_heston"
    assert r["paths"].shape == (50, 31)
