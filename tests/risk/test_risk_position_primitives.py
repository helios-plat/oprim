"""Tests for the risk primitives upstreamed from the Helios consumer line.

Covers ``atr_position_cap``, ``fee_edge_filter``, ``net_exposure_clip`` and
``cvar_portfolio_optimize``. The first three are pure arithmetic; the last one
needs the optional ``riskfolio-lib`` dependency and is skipped when absent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from oprim.risk import (
    atr_position_cap,
    cvar_portfolio_optimize,
    fee_edge_filter,
    net_exposure_clip,
)

# ---------------------------------------------------------------------------
# atr_position_cap
# ---------------------------------------------------------------------------


def test_atr_position_cap_scales_inversely_with_volatility():
    low_vol = atr_position_cap(0.01, risk_budget=0.01, min_position=0.0, max_position=1.0)
    high_vol = atr_position_cap(0.05, risk_budget=0.01, min_position=0.0, max_position=1.0)
    assert low_vol == pytest.approx(1.0)
    assert high_vol == pytest.approx(0.2)
    assert high_vol < low_vol


def test_atr_position_cap_clips_to_floor_and_ceiling():
    # risk_budget / atr_pct = 10 -> clipped to max_position
    assert atr_position_cap(0.001, risk_budget=0.01, min_position=0.05, max_position=0.5) == 0.5
    # risk_budget / atr_pct = 0.01 -> clipped to min_position
    assert atr_position_cap(1.0, risk_budget=0.01, min_position=0.05, max_position=0.5) == 0.05


def test_atr_position_cap_is_keyword_only_beyond_core_arg():
    with pytest.raises(TypeError):
        atr_position_cap(0.01, 0.01, 0.0, 1.0)  # noqa: B026 - positional must fail


@pytest.mark.parametrize(
    ("atr_pct", "risk_budget", "min_position", "max_position", "match"),
    [
        (0.0, 0.01, 0.0, 1.0, "atr_pct must be > 0"),
        (-0.01, 0.01, 0.0, 1.0, "atr_pct must be > 0"),
        (0.01, 0.0, 0.0, 1.0, "risk_budget must be > 0"),
        (0.01, 0.01, 0.9, 0.5, "must be <= max_position"),
    ],
)
def test_atr_position_cap_rejects_bad_inputs(atr_pct, risk_budget, min_position, max_position, match):
    with pytest.raises(ValueError, match=match):
        atr_position_cap(
            atr_pct, risk_budget=risk_budget, min_position=min_position, max_position=max_position
        )


# ---------------------------------------------------------------------------
# fee_edge_filter
# ---------------------------------------------------------------------------


def test_fee_edge_filter_passes_when_edge_clears_fees():
    out = fee_edge_filter(10_000.0, atr_pct=0.02, taker_fee_rate=0.0005, edge_multiple=1.5)
    # fee = 10000 * 0.0005 * 2 = 10 ; gross = 200 ; min_gross = 15
    assert out["fee"] == pytest.approx(10.0)
    assert out["expected_gross"] == pytest.approx(200.0)
    assert out["min_gross"] == pytest.approx(15.0)
    assert out["passes"] is True


def test_fee_edge_filter_rejects_when_edge_too_small():
    out = fee_edge_filter(10_000.0, atr_pct=0.0001, taker_fee_rate=0.0005, edge_multiple=1.5)
    # gross = 1.0 < min_gross = 15.0
    assert out["passes"] is False


def test_fee_edge_filter_zero_fee_always_passes():
    out = fee_edge_filter(1_000.0, atr_pct=0.001, taker_fee_rate=0.0, edge_multiple=1.0)
    assert out["fee"] == 0.0
    assert out["passes"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"atr_pct": 0.0, "taker_fee_rate": 0.001, "edge_multiple": 1.0}, "atr_pct must be > 0"),
        ({"atr_pct": 0.01, "taker_fee_rate": -1.0, "edge_multiple": 1.0}, "taker_fee_rate"),
        ({"atr_pct": 0.01, "taker_fee_rate": 0.001, "edge_multiple": 0.0}, "edge_multiple"),
    ],
)
def test_fee_edge_filter_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fee_edge_filter(1_000.0, **kwargs)


# ---------------------------------------------------------------------------
# net_exposure_clip
# ---------------------------------------------------------------------------


def test_net_exposure_clip_leaves_small_position_alone():
    out = net_exposure_clip(
        0.1, correlated_positions=[(0.2, 0.5)], max_net_exposure=1.0
    )
    assert out["allowed_size"] == pytest.approx(0.1)
    assert out["net_exposure"] == pytest.approx(0.2)
    assert out["was_clipped"] is False
    assert out["was_rejected"] is False


def test_net_exposure_clip_clips_to_bound():
    out = net_exposure_clip(
        0.9, correlated_positions=[(0.6, 0.5)], max_net_exposure=1.0
    )
    # already_used = 0.3 ; net = 1.2 > 1.0 -> allowed = 1.0 - 0.3 = 0.7
    assert out["allowed_size"] == pytest.approx(0.7)
    assert out["net_exposure"] == pytest.approx(1.0)
    assert out["was_clipped"] is True
    assert out["was_rejected"] is False


def test_net_exposure_clip_rejects_when_budget_already_spent():
    out = net_exposure_clip(
        0.5, correlated_positions=[(2.0, 0.9)], max_net_exposure=1.0
    )
    assert out["allowed_size"] == 0.0
    assert out["was_clipped"] is True
    assert out["was_rejected"] is True


def test_net_exposure_clip_preserves_sign_for_short_positions():
    out = net_exposure_clip(
        -0.9, correlated_positions=[(-0.6, 0.5)], max_net_exposure=1.0
    )
    assert out["allowed_size"] == pytest.approx(-0.7)
    assert out["net_exposure"] == pytest.approx(-1.0)


def test_net_exposure_clip_requires_positive_bound():
    with pytest.raises(ValueError, match="max_net_exposure must be > 0"):
        net_exposure_clip(0.1, correlated_positions=[], max_net_exposure=0.0)


# ---------------------------------------------------------------------------
# cvar_portfolio_optimize
# ---------------------------------------------------------------------------


def _two_asset_returns(n: int = 260) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(n, 2)),
        columns=["AAA", "BBB"],
    )


def test_cvar_portfolio_optimize_rejects_single_asset():
    with pytest.raises(ValueError, match="need >= 2 assets"):
        cvar_portfolio_optimize(_two_asset_returns()[["AAA"]])


def test_cvar_portfolio_optimize_rejects_nan():
    df = _two_asset_returns()
    df.iloc[3, 1] = float("nan")
    with pytest.raises(ValueError, match="must not contain NaNs"):
        cvar_portfolio_optimize(df)


def test_cvar_portfolio_optimize_weights_sum_to_one():
    pytest.importorskip("riskfolio")
    out = cvar_portfolio_optimize(_two_asset_returns())
    assert out["converged"] is True
    assert set(out["weights"]) == {"AAA", "BBB"}
    assert sum(out["weights"].values()) == pytest.approx(1.0)
    assert all(w >= 0.0 for w in out["weights"].values())
