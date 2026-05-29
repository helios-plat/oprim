"""Tests for oprim.spectral.ledoit_wolf_shrinkage."""
from __future__ import annotations

import numpy as np
import pytest

from oprim.spectral.ledoit_wolf import ledoit_wolf_shrinkage


def _make_returns(t: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n)) * 0.01


class TestLedoitWolfShrinkage:
    def test_returns_all_keys(self):
        r = _make_returns(100, 5)
        result = ledoit_wolf_shrinkage(r)
        assert set(result.keys()) == {"cov_lw", "alpha", "sample_cov", "target_cov"}

    def test_alpha_in_unit_interval(self):
        r = _make_returns(100, 10)
        result = ledoit_wolf_shrinkage(r)
        assert 0.0 <= result["alpha"] <= 1.0

    def test_cov_lw_is_symmetric(self):
        r = _make_returns(80, 8)
        result = ledoit_wolf_shrinkage(r)
        assert np.allclose(result["cov_lw"], result["cov_lw"].T, atol=1e-12)

    def test_cov_lw_is_psd(self):
        r = _make_returns(60, 6)
        result = ledoit_wolf_shrinkage(r)
        eigs = np.linalg.eigvalsh(result["cov_lw"])
        assert np.all(eigs >= -1e-12)

    def test_very_small_t_alpha_at_max(self):
        # T barely > N with seed=1 → sklearn LW pushes alpha toward 1
        r = _make_returns(5, 4, seed=1)
        result = ledoit_wolf_shrinkage(r)
        assert result["alpha"] >= 0.5

    def test_identity_target_shape(self):
        r = _make_returns(50, 5)
        result = ledoit_wolf_shrinkage(r, target="identity")
        assert result["target_cov"].shape == (5, 5)

    def test_identity_target_is_scaled_identity(self):
        r = _make_returns(50, 4)
        result = ledoit_wolf_shrinkage(r, target="identity")
        f = result["target_cov"]
        diag_vals = np.diag(f)
        # All diagonal elements equal
        assert np.allclose(diag_vals, diag_vals[0], rtol=1e-10)
        # Off-diagonal zero
        assert np.allclose(f - np.diag(diag_vals), 0.0, atol=1e-12)

    def test_diagonal_target_is_diagonal(self):
        r = _make_returns(50, 5)
        result = ledoit_wolf_shrinkage(r, target="diagonal")
        f = result["target_cov"]
        off_diag = f - np.diag(np.diag(f))
        assert np.allclose(off_diag, 0.0, atol=1e-12)

    def test_constant_corr_target_symmetric(self):
        r = _make_returns(80, 6)
        result = ledoit_wolf_shrinkage(r, target="constant_corr")
        f = result["target_cov"]
        assert np.allclose(f, f.T, atol=1e-12)

    def test_invalid_1d_raises(self):
        with pytest.raises(ValueError):
            ledoit_wolf_shrinkage(np.ones(10))

    def test_invalid_too_few_rows_raises(self):
        with pytest.raises(ValueError, match="T"):
            ledoit_wolf_shrinkage(np.ones((1, 5)))

    def test_invalid_too_few_cols_raises(self):
        with pytest.raises(ValueError, match="N"):
            ledoit_wolf_shrinkage(np.ones((10, 1)))
