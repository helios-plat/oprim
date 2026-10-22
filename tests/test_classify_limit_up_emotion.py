"""Tests for oprim.classify_limit_up_emotion (ported verbatim from Tide, 7 tests)."""

from __future__ import annotations

from oprim.classify_limit_up_emotion import classify_limit_up_emotion


def _c(**kw):
    base = dict(
        n_limit_up=40, market_height=3, break_rate=0.2, promotion_rate=0.4, money_effect=1.0
    )
    base.update(kw)
    return classify_limit_up_emotion(**base)


def test_gaochao_phase() -> None:
    out = _c(n_limit_up=90, market_height=8, money_effect=4.0)
    assert out["phase"] == "高潮"
    assert out["sentiment_score"] > 70


def test_bingdian_phase() -> None:
    out = _c(n_limit_up=8, market_height=1, money_effect=-1.0, promotion_rate=0.0)
    assert out["phase"] == "冰点"


def test_tuichao_on_high_break_rate() -> None:
    out = _c(break_rate=0.6, money_effect=-0.5)
    assert out["phase"] == "退潮"


def test_tuichao_on_negative_money_effect() -> None:
    out = _c(break_rate=0.2, money_effect=-3.0)
    assert out["phase"] == "退潮"


def test_fajiao_phase() -> None:
    out = _c(money_effect=2.0, promotion_rate=0.5, market_height=4, break_rate=0.1)
    assert out["phase"] == "发酵"


def test_fenqi_default() -> None:
    out = _c(money_effect=0.1, promotion_rate=0.1, break_rate=0.3, market_height=2)
    assert out["phase"] == "分歧"


def test_money_effect_none_safe() -> None:
    out = _c(money_effect=None)
    assert out["phase"] in {"冰点", "退潮", "分歧", "发酵", "高潮"}
    assert 0.0 <= out["sentiment_score"] <= 100.0
