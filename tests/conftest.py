"""Shared test fixtures for oprim."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng):
    """Sample daily returns array (252 trading days)."""
    return rng.normal(0.0005, 0.02, size=252)
