"""Tests for derivatives analytics oprim."""
from oprim.derivatives.analytics import compute_max_pain, compute_put_call_ratio

def test_max_pain():
    chain = [
        {'strike': 60000, 'open_interest': 100, 'type': 'put'},
        {'strike': 70000, 'open_interest': 150, 'type': 'call'},
    ]
    mp = compute_max_pain(chain)
    assert mp in [60000, 70000]

def test_pcr():
    res = compute_put_call_ratio({'puts_volume': 500.0, 'calls_volume': 1000.0})
    assert res['put_call_ratio'] == 0.5
