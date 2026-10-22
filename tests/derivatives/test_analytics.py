"""Fixed-input tests for deterministic derivatives analytics."""

from datetime import UTC, datetime

import pytest

from oprim.derivatives.analytics import (
    analyze_option_chain,
    compute_max_pain,
    compute_put_call_ratio,
)


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


@pytest.fixture
def option_snapshot():
    expiry = datetime(2026, 10, 22, tzinfo=UTC).timestamp() * 1000
    contracts = [
        (90, "call", 10, 0.30, 0.75),
        (90, "put", 20, 0.40, -0.20),
        (100, "call", 30, 0.25, 0.50),
        (100, "put", 40, 0.27, -0.50),
        (110, "call", 20, 0.22, 0.24),
        (110, "put", 10, 0.35, -0.75),
    ]
    return {
        "exchange": "test-fixture",
        "underlying": "TEST",
        "spot": 100.0,
        "options": [
            {
                "expiry": "261022",
                "expiration_timestamp": expiry,
                "strike": strike,
                "option_type": option_type,
                "open_interest": open_interest,
                "mark_iv": iv,
                "delta": delta,
                "gamma": 0.0001,
                "contract_multiplier": 1.0,
            }
            for strike, option_type, open_interest, iv, delta in contracts
        ],
    }


def test_open_interest_is_grouped_by_expiry(option_snapshot):
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["open_interest_by_expiry"]["2026-10-22"] == {
        "calls": 60.0,
        "puts": 70.0,
        "total": 130.0,
    }


def test_max_pain_uses_call_and_put_intrinsic_payout(option_snapshot):
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["max_pain_by_expiry"]["2026-10-22"] == 100.0


def test_pcr_uses_open_interest_and_zero_call_side_is_null():
    result = compute_put_call_ratio({"options": [
        {"option_type": "put", "open_interest": 6},
        {"option_type": "call", "open_interest": 3},
    ]})
    assert result["puts_open_interest"] == 6.0
    assert result["put_call_ratio"] == 2.0
    assert compute_put_call_ratio({"options": [{"option_type": "put", "open_interest": 1}]})[
        "put_call_ratio"
    ] is None


def test_atm_iv_is_average_of_available_types_at_nearest_strike(option_snapshot):
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["atm_iv_by_expiry"]["2026-10-22"] == {"strike": 100.0, "iv": 0.26}


def test_25_delta_skew_is_put_iv_minus_call_iv(option_snapshot):
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["skew_25d_by_expiry"]["2026-10-22"] == {
        "call_iv": 0.22,
        "put_iv": 0.4,
        "put_minus_call": pytest.approx(0.18),
    }


def test_gamma_exposure_is_signed_and_reports_approximation(option_snapshot):
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["gex_1pct_by_expiry"]["2026-10-22"] == pytest.approx(-0.1)
    assert "does not reveal dealer-side positioning" in result["gex_assumption"]


def test_delta_and_gamma_fallback_to_black_scholes_approximation(option_snapshot):
    for row in option_snapshot["options"]:
        row.pop("delta")
        row.pop("gamma")
    result = analyze_option_chain(option_snapshot, as_of=datetime(2026, 10, 20, tzinfo=UTC))
    assert result["skew_25d_by_expiry"]["2026-10-22"]["put_minus_call"] is not None
    assert result["gex_1pct"] is not None


def test_missing_spot_preserves_known_oi_but_does_not_fabricate_iv_or_gex(option_snapshot):
    option_snapshot.pop("spot")
    for row in option_snapshot["options"]:
        row.pop("underlying_price", None)
    result = analyze_option_chain(option_snapshot)
    assert result["open_interest_by_expiry"]["2026-10-22"]["total"] == 130.0
    assert result["atm_iv_by_expiry"]["2026-10-22"] is None
    assert result["gex_1pct"] is None


def test_missing_volume_and_empty_chain_fail_closed(option_snapshot):
    with pytest.raises(ValueError, match="provide non-negative"):
        compute_put_call_ratio({})
    with pytest.raises(ValueError, match="non-empty"):
        analyze_option_chain({"options": []})
