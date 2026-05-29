import pandas as pd
import pytest

from oprim._exceptions import OprimError
from oprim.apply_screen_filter import ScreenRule, apply_screen_filter
from oprim.volume_ratio import volume_ratio


def _df():
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "pe": [10.0, 25.0, 50.0, 5.0],
            "turnover": [0.02, 0.15, 0.08, 0.01],
            "is_st": [False, False, True, False],
        }
    )


# ── volume_ratio ──────────────────────────────────────────────────────────────


def test_volume_ratio_normal():
    # avg of first 5 = 100, latest = 200 → ratio = 2.0
    result = volume_ratio(volumes=[100, 110, 90, 95, 105, 200], window=5)
    assert result == pytest.approx(2.0)


def test_volume_ratio_insufficient_returns_one():
    assert volume_ratio(volumes=[100, 200], window=5) == 1.0


def test_volume_ratio_zero_avg_returns_one():
    assert volume_ratio(volumes=[0, 0, 0, 0, 0, 100], window=5) == 1.0


def test_volume_ratio_amplified_above_one():
    result = volume_ratio(volumes=[100, 100, 100, 100, 100, 300], window=5)
    assert result == pytest.approx(3.0)


def test_volume_ratio_contracted_below_one():
    result = volume_ratio(volumes=[100, 100, 100, 100, 100, 50], window=5)
    assert result == pytest.approx(0.5)


# ── apply_screen_filter ───────────────────────────────────────────────────────


def test_screen_lte_keeps_low_pe():
    rules = [ScreenRule(field="pe", op="lte", threshold=20.0, reason="PE too high")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert set(result.passed) == {"A", "D"}
    assert result.stats["total_passed"] == 2


def test_screen_gte_keeps_high_pe():
    rules = [ScreenRule(field="pe", op="gte", threshold=25.0, reason="PE too low")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert set(result.passed) == {"B", "C"}


def test_screen_between():
    rules = [ScreenRule(field="pe", op="between", threshold=(8.0, 30.0), reason="PE out of range")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert set(result.passed) == {"A", "B"}


def test_screen_flag_excludes_st():
    rules = [ScreenRule(field="is_st", op="flag", threshold=False, reason="ST excluded")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert set(result.passed) == {"A", "B", "D"}
    assert len(result.rejected) == 1
    assert result.rejected[0]["symbol"] == "C"


def test_screen_empty_rules_all_pass():
    result = apply_screen_filter(candidates=_df(), rules=[])
    assert len(result.passed) == 4
    assert len(result.rejected) == 0


def test_screen_all_rejected():
    rules = [ScreenRule(field="pe", op="lt", threshold=1.0, reason="impossible")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert len(result.passed) == 0
    assert len(result.rejected) == 4


def test_screen_rejected_contains_reason():
    rules = [ScreenRule(field="pe", op="lte", threshold=10.0, reason="PE cut")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    for rej in result.rejected:
        assert rej["reason"] == "PE cut"
        assert "symbol" in rej
        assert "failed_rule" in rej


def test_screen_no_symbol_column_raises():
    df = pd.DataFrame({"ticker": ["A"], "pe": [10.0]})
    with pytest.raises(OprimError):
        apply_screen_filter(candidates=df, rules=[])


def test_screen_eq_exact_match():
    rules = [ScreenRule(field="pe", op="eq", threshold=10.0, reason="not exactly 10")]
    result = apply_screen_filter(candidates=_df(), rules=rules)
    assert result.passed == ["A"]
