"""Tests for oprim.spectral.marchenko_pastur_threshold."""
from __future__ import annotations

import math

import pytest

from oprim.spectral.marchenko_pastur import marchenko_pastur_threshold


class TestMarchenkoPasturThreshold:
    def test_returns_all_keys(self):
        result = marchenko_pastur_threshold(100, 50)
        assert set(result.keys()) == {"q", "lambda_min", "lambda_max", "mass_zero"}

    def test_q_value(self):
        result = marchenko_pastur_threshold(100, 50)
        assert math.isclose(result["q"], 0.5, rel_tol=1e-9)

    def test_lambda_max_q_half(self):
        # q=0.5: lambda_max = (1 + sqrt(0.5))^2 ≈ 2.9142
        result = marchenko_pastur_threshold(100, 50)
        expected = (1.0 + math.sqrt(0.5)) ** 2
        assert math.isclose(result["lambda_max"], expected, rel_tol=1e-9)

    def test_lambda_min_q_half(self):
        # q=0.5: lambda_min = (1 - sqrt(0.5))^2 ≈ 0.0858
        result = marchenko_pastur_threshold(100, 50)
        expected = (1.0 - math.sqrt(0.5)) ** 2
        assert math.isclose(result["lambda_min"], expected, rel_tol=1e-9)

    def test_lambda_min_zero_when_q_equals_one(self):
        # q=1: lambda_min = 0
        result = marchenko_pastur_threshold(100, 100)
        assert result["lambda_min"] == 0.0

    def test_mass_zero_q_greater_than_one(self):
        # q=2 (n_features=200, n_samples=100): mass_zero = (2-1)/2 = 0.5
        result = marchenko_pastur_threshold(100, 200)
        assert math.isclose(result["mass_zero"], 0.5, rel_tol=1e-9)

    def test_mass_zero_zero_when_q_less_than_one(self):
        result = marchenko_pastur_threshold(200, 100)
        assert result["mass_zero"] == 0.0

    def test_sigma_sq_scaling(self):
        r1 = marchenko_pastur_threshold(100, 50, sigma_sq=1.0)
        r2 = marchenko_pastur_threshold(100, 50, sigma_sq=4.0)
        assert math.isclose(r2["lambda_max"], 4.0 * r1["lambda_max"], rel_tol=1e-9)
        assert math.isclose(r2["lambda_min"], 4.0 * r1["lambda_min"], rel_tol=1e-9)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            marchenko_pastur_threshold(0, 10)

    def test_invalid_n_features(self):
        with pytest.raises(ValueError, match="n_features"):
            marchenko_pastur_threshold(10, 0)

    def test_invalid_sigma_sq(self):
        with pytest.raises(ValueError, match="sigma_sq"):
            marchenko_pastur_threshold(100, 50, sigma_sq=-1.0)

    def test_lambda_max_greater_than_lambda_min(self):
        result = marchenko_pastur_threshold(200, 100)
        assert result["lambda_max"] > result["lambda_min"]

    def test_equal_dimensions_lambda_max(self):
        # q=1: lambda_max = (1+1)^2 = 4
        result = marchenko_pastur_threshold(100, 100)
        assert math.isclose(result["lambda_max"], 4.0, rel_tol=1e-9)
