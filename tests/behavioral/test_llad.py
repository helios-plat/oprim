"""Tests for oprim.behavioral.large_loss_aversion_degree."""

from __future__ import annotations

import pytest

from oprim.behavioral.llad import large_loss_aversion_degree


def test_llad_default_tk_params():
    """TK default params (alpha=beta=0.88, lambda=2.25) → LLAD ≈ 2.513."""
    result = large_loss_aversion_degree(alpha=0.88, beta=0.88, loss_aversion=2.25)
    # LLAD = (0.88/0.88) * 2.25^(1/0.88) ≈ 2.5131
    assert result["llad"] == pytest.approx(2.5131, rel=1e-3)


def test_llad_keys():
    """Return dict always contains llad, well_posed, llad_threshold keys."""
    result = large_loss_aversion_degree(alpha=0.88, beta=0.88, loss_aversion=2.25)
    assert set(result.keys()) == {"llad", "well_posed", "llad_threshold"}


def test_well_posed_none_without_distribution():
    """Without distribution params, well_posed is None."""
    result = large_loss_aversion_degree(alpha=0.88, beta=0.88, loss_aversion=2.25)
    assert result["well_posed"] is None
    assert result["llad_threshold"] is None


def test_well_posed_with_tail_index():
    """With tail_index, well_posed and threshold are computed."""
    result = large_loss_aversion_degree(
        alpha=0.88,
        beta=0.88,
        loss_aversion=2.25,
        return_distribution_params={"tail_index": 3.0},
    )
    # threshold = 3 / (3-1) = 1.5; LLAD ≈ 2.518 > 1.5 → well_posed=True
    assert result["llad_threshold"] == pytest.approx(1.5)
    assert result["well_posed"] is True


def test_not_well_posed():
    """Low LLAD vs high threshold gives well_posed=False."""
    # alpha=beta=1, lambda=1 → LLAD = 1/1 * 1^1 = 1.0
    # tail_index=1.5 → threshold = 1.5/(0.5) = 3.0 → 1.0 < 3.0 → False
    result = large_loss_aversion_degree(
        alpha=1.0,
        beta=1.0,
        loss_aversion=1.0,
        return_distribution_params={"tail_index": 1.5},
    )
    assert result["well_posed"] is False


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        large_loss_aversion_degree(alpha=0.0, beta=0.88, loss_aversion=2.25)


def test_invalid_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        large_loss_aversion_degree(alpha=0.88, beta=1.5, loss_aversion=2.25)


def test_invalid_loss_aversion_raises():
    with pytest.raises(ValueError, match="loss_aversion"):
        large_loss_aversion_degree(alpha=0.88, beta=0.88, loss_aversion=0.9)


def test_tail_index_le_one_raises():
    with pytest.raises(ValueError, match="tail_index"):
        large_loss_aversion_degree(
            alpha=0.88,
            beta=0.88,
            loss_aversion=2.25,
            return_distribution_params={"tail_index": 1.0},
        )


def test_llad_formula_manual():
    """Manual calculation: LLAD = (beta/alpha) * lambda^(1/beta)."""
    alpha, beta, lam = 0.7, 0.8, 2.0
    expected = (beta / alpha) * lam ** (1.0 / beta)
    result = large_loss_aversion_degree(alpha=alpha, beta=beta, loss_aversion=lam)
    assert result["llad"] == pytest.approx(expected)
