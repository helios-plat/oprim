"""Cognitive modeling atomic operations (BKT, FSRS)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Dict

from fsrs import Card, Rating, Scheduler

@dataclass
class KCState:
    """贝叶斯知识追踪状态（知识组件级）。
    kc_id: 知识组件标识符
    p_init: 初始掌握概率先验
    p_transit: 一次练习后从未掌握→掌握的转移概率
    p_guess: 未掌握状态下答对的概率（猜对）
    p_slip: 已掌握状态下答错的概率（失误）
    p_mastery: 当前掌握概率 P(L)，None 表示使用先验
    long_term_mastery: 去遗忘平滑的长期掌握度
    last_interaction_ts: 上次交互 unix 时间戳
    n_attempts: 累计交互次数
    """
    kc_id: str
    p_init: float = 0.20
    p_transit: float = 0.20
    p_guess: float = 0.15
    p_slip: float = 0.12
    p_mastery: Optional[float] = None
    p_recognition: Optional[float] = None       # 识别维度掌握概率（M-G）
    p_recognition_init: float = 0.20             # 识别维度先验
    long_term_mastery: Optional[float] = None
    last_interaction_ts: Optional[float] = None
    n_attempts: int = 0

    def current(self) -> float:
        return self.p_mastery if self.p_mastery is not None else self.p_init

def bkt_update(
    *,
    state: KCState,
    is_correct: bool,
    retrievability: float | None = None,
    days_since: float | None = None,
) -> KCState:
    """Forgetting-aware 贝叶斯知识追踪更新（in-place + return）。
    掌握度封顶 0.97，long_term_mastery 用 EMA(α=0.4) 平滑。
    """
    # 0. 准备参数
    p_l = state.current()
    p_g = state.p_guess
    p_s = state.p_slip
    p_t = state.p_transit
    alpha = 0.4
    cap = 0.97

    # 1. Forgetting decay
    r = 1.0
    if retrievability is not None:
        r = retrievability
    elif days_since is not None:
        r = exp_forgetting(days_since=days_since)
    
    p_eff = p_l * r

    # 2. Bayesian update (Observation)
    if is_correct:
        p_obs = (p_eff * (1 - p_s)) / (p_eff * (1 - p_s) + (1 - p_eff) * p_g)
    else:
        p_obs = (p_eff * p_s) / (p_eff * p_s + (1 - p_eff) * (1 - p_g))

    # 3. Learning transition
    p_new = p_obs + (1 - p_obs) * p_t
    p_new = min(p_new, cap)

    # 4. Update state (in-place)
    state.p_mastery = p_new
    
    if state.long_term_mastery is None:
        state.long_term_mastery = p_new
    else:
        state.long_term_mastery = alpha * p_new + (1 - alpha) * state.long_term_mastery

    state.n_attempts += 1
    # Note: last_interaction_ts update is left to the caller or handled via oskill
    return state

def bkt_classify_error(*, state: KCState) -> str:
    """答错时判定根因：'careless' 或 'dontknow'。"""
    p_l = state.current()
    p_g = state.p_guess
    p_s = state.p_slip

    # careless ∝ P(L) * P(S)
    # dontknow ∝ (1-P(L)) * (1-P(G))
    careless_weight = p_l * p_s
    dontknow_weight = (1 - p_l) * (1 - p_g)

    return "careless" if careless_weight >= dontknow_weight else "dontknow"

def bkt_predict_correct(*, state: KCState, retrievability: float | None = None) -> float:
    """预测下一次答对概率。"""
    p_l = state.current()
    r = retrievability if retrievability is not None else 1.0
    p_eff = p_l * r
    return p_eff * (1 - state.p_slip) + (1 - p_eff) * state.p_guess

def exp_forgetting(*, days_since: float, halflife_days: float = 7.0) -> float:
    """指数遗忘近似 R = 0.5^(days/halflife)。"""
    if halflife_days <= 0:
        raise ValueError("halflife_days must be positive")
    if days_since < 0:
        raise ValueError("days_since must be non-negative")
    return 0.5 ** (days_since / halflife_days)

def bkt_new_state(*, kc_id: str, prior: dict) -> KCState:
    """从先验参数字典创建 KCState。"""
    return KCState(
        kc_id=kc_id,
        p_init=prior.get("p_init", 0.2),
        p_transit=prior.get("p_transit", 0.2),
        p_guess=prior.get("p_guess", 0.15),
        p_slip=prior.get("p_slip", 0.12),
    )

# FSRS Helpers (dict-based serialization)

def _card_to_dict(card: Card) -> dict:
    return card.to_dict()

def _dict_to_card(d: dict) -> Card:
    return Card.from_dict(d)


def fsrs_new_card() -> dict:
    """创建新 FSRS 记忆卡片。"""
    return _card_to_dict(Card())

def fsrs_review(*, card_dict: dict, rating: str, now: datetime | None = None) -> dict:
    """对卡片做一次复习。"""
    card = _dict_to_card(card_dict)
    scheduler = Scheduler()
    if now is None:
        now = datetime.now(timezone.utc)
    
    rating_map = {
        "Again": Rating.Again,
        "Hard": Rating.Hard,
        "Good": Rating.Good,
        "Easy": Rating.Easy,
    }
    r = rating_map.get(rating, Rating.Good)
    
    new_card, _ = scheduler.review_card(card, r, now)
    return _card_to_dict(new_card)

def fsrs_retrievability(*, card_dict: dict, now: datetime | None = None) -> float:
    """当前可提取性 R ∈ [0,1]。"""
    if card_dict.get("last_review") is None:
        return 1.0
    card = _dict_to_card(card_dict)
    scheduler = Scheduler()
    if now is None:
        now = datetime.now(timezone.utc)
    return float(scheduler.get_card_retrievability(card, now))

def fsrs_map_rating(
    *, is_correct: bool, used_answer: bool = False,
    struggled: bool = False, effortless: bool = False
) -> str:
    """表现映射为 FSRS Rating 字符串。"""
    if used_answer or not is_correct:
        return "Again"
    if struggled:
        return "Hard"
    if effortless:
        return "Easy"
    return "Good"

def fsrs_due_date(*, card_dict: dict) -> str | None:
    """下次复习日期 ISO 字符串。"""
    return card_dict.get("due")
