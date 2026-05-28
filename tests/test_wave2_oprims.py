"""Tests for Wave 2 elements."""
import numpy as np
import pytest

from oprim.correlation_options import compute_option_skew_curve_data, compute_rolling_correlation_heatmap, OprimError


def test_rolling_corr_basic():
    data = np.random.randn(100, 4)
    r = compute_rolling_correlation_heatmap(data_matrix=data, window_size=30)
    assert len(r["correlation_cube"]) == 71
    assert len(r["correlation_cube"][0]) == 4

def test_rolling_corr_spearman():
    data = np.random.randn(50, 3)
    r = compute_rolling_correlation_heatmap(data_matrix=data, window_size=20, method="spearman")
    assert len(r["correlation_cube"]) > 0

def test_rolling_corr_window_too_large():
    with pytest.raises(OprimError):
        compute_rolling_correlation_heatmap(data_matrix=np.zeros((10, 3)), window_size=20)

def test_rolling_corr_single_col():
    with pytest.raises(OprimError):
        compute_rolling_correlation_heatmap(data_matrix=np.zeros((50, 1)), window_size=10)

def test_option_skew_basic():
    chain = {"result": [
        {"expiration": "2024-06-28", "strike": 90000, "mark_iv": 0.55},
        {"expiration": "2024-06-28", "strike": 95000, "mark_iv": 0.52},
        {"expiration": "2024-06-28", "strike": 100000, "mark_iv": 0.50},
        {"expiration": "2024-06-28", "strike": 105000, "mark_iv": 0.48},
        {"expiration": "2024-06-28", "strike": 110000, "mark_iv": 0.47},
    ]}
    r = compute_option_skew_curve_data(option_chain=chain, spot_price=100000)
    assert len(r["slices"]) == 1
    assert r["slices"][0]["atm_iv"] == 0.50

def test_option_skew_empty():
    with pytest.raises(OprimError):
        compute_option_skew_curve_data(option_chain={"result": []}, spot_price=100000)
