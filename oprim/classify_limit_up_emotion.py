"""涨停结构情绪联动 — 连板高度 / 炸板率 / 晋级率 / 赚钱效应 → 情绪周期相位.

Ported verbatim from Tide's ``domain.limit_up.emotion_service.classify_limit_up_emotion``
(pure function only — the DB-bound ``LimitUpEmotionService``/ladder-lookup shell stays
in Tide). The market-emotion state machine keys off a single ``n_limit_up`` count,
divorced from the涨停 ladder it should be reading. This derives the structural signals
a 短线 trader actually reads the cycle from — most importantly 赚钱效应 (how yesterday's
涨停 stocks performed today) — and maps them to the canonical 冰点 / 退潮 / 分歧 / 发酵 /
高潮 phases.
"""

from __future__ import annotations

from typing import Any


def classify_limit_up_emotion(
    *,
    n_limit_up: int,
    market_height: int,
    break_rate: float,
    promotion_rate: float | None,
    money_effect: float | None,
) -> dict[str, Any]:
    """Map涨停 structural signals to an emotion phase. Pure, deterministic.

    Args:
        n_limit_up: 当日涨停家数.
        market_height: 最高连板高度.
        break_rate: 炸板率 [0, 1].
        promotion_rate: 晋级率 [0, 1], 或 None (视为 0).
        money_effect: 赚钱效应 —— 昨日涨停股今日平均涨跌幅(%), 或 None (视为 0).
            这是主导信号: 正赚钱效应 + 高度抬升是发酵, 负值 + 高炸板率是退潮.

    Returns:
        dict with phase (冰点/退潮/分歧/发酵/高潮), sentiment_score (0-100), reasons.
    """
    me = money_effect if money_effect is not None else 0.0
    pr = promotion_rate if promotion_rate is not None else 0.0
    reasons: list[str] = []

    # 高潮: 空间打开 + 强赚钱效应.
    if market_height >= 7 and me > 2.0 and n_limit_up >= 40:
        phase = "高潮"
        reasons.append(f"空间高度{market_height}板, 赚钱效应+{me:.1f}%")
    # 冰点: 极少涨停 + 负赚钱效应.
    elif n_limit_up < 20 and me <= 0:
        phase = "冰点"
        reasons.append(f"涨停仅{n_limit_up}家, 赚钱效应{me:.1f}%")
    # 退潮: 高炸板率 或 明显亏钱效应.
    elif break_rate >= 0.5 or me <= -2.0:
        phase = "退潮"
        reasons.append(f"炸板率{break_rate:.0%}, 赚钱效应{me:.1f}%")
    # 发酵: 正赚钱效应 + 晋级顺畅 + 高度抬升.
    elif me > 0.5 and pr >= 0.3 and market_height >= 3:
        phase = "发酵"
        reasons.append(f"赚钱效应+{me:.1f}%, 晋级率{pr:.0%}, 高度{market_height}板")
    # 分歧: 其余(涨停尚可但赚钱效应/晋级走弱).
    else:
        phase = "分歧"
        reasons.append(f"赚钱效应{me:.1f}%, 晋级率{pr:.0%}, 炸板率{break_rate:.0%}")

    # A 0-100 sentiment score: 赚钱效应 + 高度 + 晋级 lift it, 炸板 drags it.
    score = 50.0 + me * 4.0 + market_height * 2.0 + pr * 20.0 - break_rate * 40.0
    score = round(max(0.0, min(100.0, score)), 1)

    return {"phase": phase, "sentiment_score": score, "reasons": reasons}
