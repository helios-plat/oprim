"""Tests for compute_shapley_values (true non-linear Shapley)."""
import math
import pytest
from oprim._quant_analysis import compute_shapley_values
from oprim._quant_analysis import QuantAnalysisError


def test_additivity_exact():
    # linear agg: shapley should recover contributions exactly
    agg = lambda d: d.get("a", 0) + d.get("b", 0) + d.get("c", 0)
    sv = compute_shapley_values(features={"a": 3.0, "b": 2.0, "c": 1.0},
                                aggregate_fn=agg, method="exact")
    total = agg({"a": 3.0, "b": 2.0, "c": 1.0})
    assert math.isclose(sv["a"] + sv["b"] + sv["c"] + sv["baseline"] + sv["residual"],
                        total, abs_tol=1e-6)
    # linear → each shapley == its own contribution
    assert math.isclose(sv["a"], 3.0, abs_tol=1e-6)


def test_nonlinear_geometric_mean():
    # geometric mean is non-linear: Σcontrib ≠ total
    agg = lambda d: (max(d.get("a", 0), 0) * max(d.get("b", 0), 0)) ** 0.5
    sv = compute_shapley_values(features={"a": 4.0, "b": 9.0},
                                aggregate_fn=agg, method="exact")
    total = agg({"a": 4.0, "b": 9.0})
    # additivity holds even for non-linear
    assert math.isclose(sv["a"] + sv["b"] + sv["baseline"] + sv["residual"],
                        total, abs_tol=1e-6)


def test_determinism():
    agg = lambda d: sum(v ** 2 for v in d.values())
    f = {"x": 1.0, "y": 2.0, "z": 3.0}
    sv1 = compute_shapley_values(features=f, aggregate_fn=agg, method="monte_carlo", n_samples=500)
    sv2 = compute_shapley_values(features=f, aggregate_fn=agg, method="monte_carlo", n_samples=500)
    assert sv1 == sv2  # same seed → identical


def test_monte_carlo_approximates_exact():
    agg = lambda d: (max(d.get("a",0),0)*max(d.get("b",0),0)*max(d.get("c",0),0)) ** (1/3)
    f = {"a": 2.0, "b": 4.0, "c": 8.0}
    exact = compute_shapley_values(features=f, aggregate_fn=agg, method="exact")
    mc = compute_shapley_values(features=f, aggregate_fn=agg, method="monte_carlo", n_samples=3000)
    for d in ["a", "b", "c"]:
        assert math.isclose(exact[d], mc[d], abs_tol=0.05)


def test_negative_contributions():
    agg = lambda d: d.get("up", 0) - d.get("down", 0)
    sv = compute_shapley_values(features={"up": 2.0, "down": 5.0},
                                aggregate_fn=agg, method="exact")
    assert sv["up"] > 0
    assert sv["down"] < 0  # negative dim gets negative shapley


def test_empty_features_raises():
    with pytest.raises(QuantAnalysisError):
        compute_shapley_values(features={}, aggregate_fn=lambda d: 0.0)


def test_non_callable_raises():
    with pytest.raises(QuantAnalysisError):
        compute_shapley_values(features={"a": 1.0}, aggregate_fn="not callable")


def test_exact_blowup_raises():
    f = {f"d{i}": 1.0 for i in range(13)}  # n=13 > 12
    with pytest.raises(QuantAnalysisError):
        compute_shapley_values(features=f, aggregate_fn=lambda d: sum(d.values()), method="exact")


def test_baseline_separated():
    agg = lambda d: 100.0 + d.get("a", 0)  # constant baseline 100
    sv = compute_shapley_values(features={"a": 5.0}, aggregate_fn=agg, method="exact")
    assert math.isclose(sv["baseline"], 100.0, abs_tol=1e-6)
    assert math.isclose(sv["a"], 5.0, abs_tol=1e-6)
