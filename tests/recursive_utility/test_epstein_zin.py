"""Tests for oprim.recursive_utility.epstein_zin_aggregator."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.recursive_utility import epstein_zin_aggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS = dict(discount=0.96, risk_aversion=5.0, ies=0.5)


def _ez(c, ce, **kw):
    """Thin wrapper with default kwargs."""
    params = {**_BASE_KWARGS, **kw}
    return epstein_zin_aggregator(np.asarray(c, dtype=float), np.asarray(ce, dtype=float), **params)


# ---------------------------------------------------------------------------
# 1. Basic: output is strictly positive
# ---------------------------------------------------------------------------


class TestOutputPositive:
    def test_scalar_output_positive(self):
        """Single scalar inputs yield a positive value."""
        v = _ez(1.0, 1.0)
        assert float(v) > 0.0

    def test_array_output_all_positive(self):
        """Array inputs yield all-positive values."""
        c = np.array([0.5, 1.0, 2.0, 5.0])
        ce = np.array([0.8, 1.0, 1.5, 3.0])
        v = _ez(c, ce)
        assert np.all(v > 0.0)


# ---------------------------------------------------------------------------
# 2. Monotone in consumption
# ---------------------------------------------------------------------------


class TestMonotoneInConsumption:
    def test_v_increases_with_c(self):
        """Holding CE fixed, V strictly increases as C increases."""
        ce = np.ones(20)
        c = np.linspace(0.1, 5.0, 20)
        v = _ez(c, ce)
        assert np.all(np.diff(v) > 0.0)

    def test_v_increases_with_c_ies1(self):
        """Monotonicity holds in the Cobb-Douglas (ies=1) special case."""
        ce = np.ones(10)
        c = np.linspace(0.2, 3.0, 10)
        v = _ez(c, ce, ies=1.0)
        assert np.all(np.diff(v) > 0.0)


# ---------------------------------------------------------------------------
# 3. Monotone in continuation value (CE)
# ---------------------------------------------------------------------------


class TestMonotoneInCE:
    def test_v_increases_with_ce(self):
        """Holding C fixed, V strictly increases as CE increases."""
        c = np.ones(20)
        ce = np.linspace(0.1, 5.0, 20)
        v = _ez(c, ce)
        assert np.all(np.diff(v) > 0.0)


# ---------------------------------------------------------------------------
# 4. Special case ies = 1 → Cobb-Douglas
# ---------------------------------------------------------------------------


class TestCobbDouglas:
    def test_ies1_matches_cobb_douglas_scalar(self):
        """ies=1: V = C^(1-beta) * CE^beta."""
        c, ce, beta = 2.0, 3.0, 0.96
        v = _ez(c, ce, ies=1.0, discount=beta)
        expected = c ** (1.0 - beta) * ce**beta
        assert float(v) == pytest.approx(expected, rel=1e-10)

    def test_ies1_matches_cobb_douglas_array(self):
        """ies=1 holds element-wise for arrays."""
        rng = np.random.default_rng(42)
        c = rng.uniform(0.5, 3.0, 50)
        ce = rng.uniform(0.5, 3.0, 50)
        beta = 0.95
        v = _ez(c, ce, ies=1.0, discount=beta)
        expected = c ** (1.0 - beta) * ce**beta
        np.testing.assert_allclose(v, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. CRRA special case: gamma = 1/ies → V reduces to power-utility form
# ---------------------------------------------------------------------------


class TestCRRASpecialCase:
    def test_crra_homogeneity(self):
        """When gamma == 1/ies (CRRA), V is homogeneous of degree 1 in (C, CE).

        Scaling both C and CE by a factor k should scale V by k.
        """
        c = np.array([1.5, 2.0, 2.5])
        ce = np.array([1.2, 1.8, 2.2])
        k = 2.5
        # use psi = 1/gamma so rho = 1 - gamma
        gamma, ies = 3.0, 1.0 / 3.0
        v1 = _ez(c, ce, risk_aversion=gamma, ies=ies)
        v2 = _ez(k * c, k * ce, risk_aversion=gamma, ies=ies)
        np.testing.assert_allclose(v2, k * v1, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. discount = 0 → V depends only on C
# ---------------------------------------------------------------------------


class TestDiscountEdgeCases:
    def test_discount_near_zero_dominated_by_c(self):
        """Very small beta: V ≈ C (CE contribution vanishes)."""
        c = np.array([1.0, 2.0, 3.0])
        ce_a = np.array([1.0, 1.0, 1.0])
        ce_b = np.array([100.0, 100.0, 100.0])
        beta = 1e-9
        va = _ez(c, ce_a, discount=beta)
        vb = _ez(c, ce_b, discount=beta)
        # Both should be very close to C^(1/rho * (1-beta))
        np.testing.assert_allclose(va, vb, rtol=1e-5)

    def test_discount_near_one_dominated_by_ce(self):
        """beta very close to 1: V ≈ CE (C contribution vanishes)."""
        c_a = np.array([1.0, 1.0, 1.0])
        c_b = np.array([100.0, 100.0, 100.0])
        ce = np.array([2.0, 3.0, 4.0])
        beta = 1.0 - 1e-9
        va = _ez(c_a, ce, discount=beta)
        vb = _ez(c_b, ce, discount=beta)
        np.testing.assert_allclose(va, vb, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Scalar and array inputs both work
# ---------------------------------------------------------------------------


class TestInputShapes:
    def test_scalar_float_input(self):
        """Pure Python floats as inputs return a scalar-like array."""
        v = epstein_zin_aggregator(1.5, 2.0, discount=0.96, risk_aversion=5.0, ies=0.5)
        assert np.ndim(v) == 0 or np.isscalar(v) or v.shape == ()

    def test_1d_array_input(self):
        """1-D arrays pass through with correct shape."""
        c = np.array([1.0, 2.0, 3.0])
        ce = np.array([1.5, 2.5, 3.5])
        v = _ez(c, ce)
        assert v.shape == (3,)

    def test_2d_array_input(self):
        """2-D arrays maintain shape."""
        c = np.ones((4, 5))
        ce = np.ones((4, 5)) * 1.2
        v = _ez(c, ce)
        assert v.shape == (4, 5)


# ---------------------------------------------------------------------------
# 8. Invalid inputs raise ValueError
# ---------------------------------------------------------------------------


class TestValidation:
    def test_discount_zero_raises(self):
        with pytest.raises(ValueError, match="discount"):
            _ez(1.0, 1.0, discount=0.0)

    def test_discount_one_raises(self):
        with pytest.raises(ValueError, match="discount"):
            _ez(1.0, 1.0, discount=1.0)

    def test_discount_negative_raises(self):
        with pytest.raises(ValueError, match="discount"):
            _ez(1.0, 1.0, discount=-0.1)

    def test_risk_aversion_zero_raises(self):
        with pytest.raises(ValueError, match="risk_aversion"):
            _ez(1.0, 1.0, risk_aversion=0.0)

    def test_ies_zero_raises(self):
        with pytest.raises(ValueError, match="ies"):
            _ez(1.0, 1.0, ies=0.0)

    def test_consumption_zero_raises(self):
        with pytest.raises(ValueError, match="consumption"):
            _ez(0.0, 1.0)

    def test_consumption_negative_raises(self):
        with pytest.raises(ValueError, match="consumption"):
            _ez(-1.0, 1.0)

    def test_continuation_zero_raises(self):
        with pytest.raises(ValueError, match="continuation_value"):
            _ez(1.0, 0.0)

    def test_continuation_negative_raises(self):
        with pytest.raises(ValueError, match="continuation_value"):
            _ez(1.0, -2.0)


# ---------------------------------------------------------------------------
# 9. Large |rho| / near-zero ies: numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_very_small_ies_does_not_overflow(self):
        """ies near 0 (large negative rho) must not produce inf or nan."""
        c = np.array([1.0, 2.0, 3.0])
        ce = np.array([1.5, 2.5, 3.5])
        v = _ez(c, ce, ies=1e-4)
        assert np.all(np.isfinite(v))
        assert np.all(v > 0.0)

    def test_large_ies_does_not_overflow(self):
        """ies >> 1 (rho near 1) remains stable."""
        c = np.array([0.5, 1.0, 2.0])
        ce = np.array([0.8, 1.2, 2.5])
        v = _ez(c, ce, ies=100.0)
        assert np.all(np.isfinite(v))
        assert np.all(v > 0.0)

    def test_ies1_exact_boundary(self):
        """ies exactly 1.0 uses Cobb-Douglas branch, no division by zero."""
        c, ce = np.array([2.0, 3.0]), np.array([1.5, 2.5])
        v = _ez(c, ce, ies=1.0)
        assert np.all(np.isfinite(v))
