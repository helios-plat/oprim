"""BKT+IRT 难度感知回归测试（Phase 0：零行为变化 + 难度方向性）。

守护红线：
- difficulty=None / 0.5 → 与旧实现逐位等价（R1）
- 难度方向性正确（R2）
- P(L)∈(0,0.97]、数值有限（R3）
- 固定难度连续答对单调不降且封顶（R6）
覆盖生产实现 oprim.bkt 与孪生 oprim.cognitive。
"""
import copy

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from oprim import KCState
from oprim.bkt import bkt_update, predict_correct, classify_error, _item_adjust, new_state_from_prior
from oprim.cognitive import (
    bkt_update as bkt_update_c,
    bkt_predict_correct,
)

_MASTERY_CAP = 0.97


def mk(p_mastery=None, p_init=0.3, p_transit=0.2, p_guess=0.2, p_slip=0.1):
    return KCState(
        kc_id="t", p_init=p_init, p_transit=p_transit,
        p_guess=p_guess, p_slip=p_slip, p_mastery=p_mastery,
    )


_CASES = [
    dict(p_mastery=None, p_guess=0.2, p_slip=0.1),
    dict(p_mastery=0.5, p_guess=0.25, p_slip=0.15),
    dict(p_mastery=0.9, p_guess=0.1, p_slip=0.05),
    dict(p_mastery=0.05, p_guess=0.3, p_slip=0.2),
]


# ── R1: 向后兼容（difficulty=None == 省略 == 0.5，逐位等价）─────────────
@pytest.mark.parametrize("params", _CASES)
@pytest.mark.parametrize("is_correct", [True, False])
def test_r1_bkt_update_backward_compat(params, is_correct):
    omitted = bkt_update(state=mk(**params), is_correct=is_correct).p_mastery
    none = bkt_update(state=mk(**params), is_correct=is_correct, difficulty=None).p_mastery
    half = bkt_update(state=mk(**params), is_correct=is_correct, difficulty=0.5).p_mastery
    assert omitted == none == half


@pytest.mark.parametrize("params", _CASES)
def test_r1_predict_and_classify_backward_compat(params):
    s = mk(**params)
    assert predict_correct(state=s) == predict_correct(state=s, difficulty=None) == predict_correct(state=s, difficulty=0.5)
    assert classify_error(state=s) == classify_error(state=s, difficulty=None) == classify_error(state=s, difficulty=0.5)


def test_r1_twin_cognitive_backward_compat():
    """孪生 oprim.cognitive 也保持 None==0.5==省略。"""
    for params in _CASES:
        for is_correct in (True, False):
            a = bkt_update_c(state=mk(**params), is_correct=is_correct).p_mastery
            b = bkt_update_c(state=mk(**params), is_correct=is_correct, difficulty=0.5).p_mastery
            assert a == b
        s = mk(**params)
        assert bkt_predict_correct(state=s) == bkt_predict_correct(state=s, difficulty=0.5)


def test_r1_item_adjust_identity():
    assert _item_adjust(0.2, 0.1, None) == (0.2, 0.1)
    assert _item_adjust(0.2, 0.1, 0.5) == (0.2, 0.1)


# ── R2: 难度方向性 ─────────────────────────────────────────────────────
def test_r2_correct_harder_means_stronger_evidence():
    """答对：越难的题后验涨得越多。"""
    easy = bkt_update(state=mk(p_mastery=0.5), is_correct=True, difficulty=0.2).p_mastery
    mid = bkt_update(state=mk(p_mastery=0.5), is_correct=True, difficulty=0.5).p_mastery
    hard = bkt_update(state=mk(p_mastery=0.5), is_correct=True, difficulty=0.8).p_mastery
    assert hard > mid > easy


def test_r2_wrong_easier_means_stronger_evidence():
    """答错：越简单的题掉得越多（hard 答错没那么伤）。"""
    easy = bkt_update(state=mk(p_mastery=0.5), is_correct=False, difficulty=0.2).p_mastery
    hard = bkt_update(state=mk(p_mastery=0.5), is_correct=False, difficulty=0.8).p_mastery
    assert hard > easy  # 答错时难题后验更高（掉得少）


def test_r2_item_adjust_monotone():
    g_easy, s_easy = _item_adjust(0.2, 0.1, 0.2)
    g_hard, s_hard = _item_adjust(0.2, 0.1, 0.8)
    assert s_hard > 0.1 > s_easy   # 难题 slip↑
    assert g_hard < 0.2 < g_easy   # 难题 guess↓


# ── R3: 边界与数值安全 ─────────────────────────────────────────────────
@pytest.mark.parametrize("difficulty", [0.0, 0.25, 0.5, 0.75, 1.0, None])
@pytest.mark.parametrize("is_correct", [True, False])
@pytest.mark.parametrize("params", _CASES)
def test_r3_bounds(params, is_correct, difficulty):
    p = bkt_update(state=mk(**params), is_correct=is_correct, difficulty=difficulty).p_mastery
    assert p == p  # 非 NaN
    assert 0.0 < p <= _MASTERY_CAP


@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_r3_adjusted_params_in_unit_interval(difficulty):
    g, s = _item_adjust(0.2, 0.1, difficulty)
    assert 0.0 < g < 1.0 and 0.0 < s < 1.0


# ── R4: 难度感知 AUC ≥ 难度盲（价值证明）─────────────────────────────────
def test_r4_difficulty_aware_auc_not_worse():
    """生成含题目难度依赖的作答序列，对比"难度盲 BKT"与"难度感知 BKT"的预测 AUC。
    数据生成过程的发射参数依难度变化（与模型假设一致），故难度感知应 ≥ 难度盲。"""
    rng = np.random.RandomState(7)
    prior = {"p_init": 0.2, "p_transit": 0.15, "p_guess": 0.2, "p_slip": 0.1}
    n_students, n_steps = 200, 25

    items, outcomes = [], []          # 共享的 (难度, 结果) 序列，独立于预测器
    for _ in range(n_students):
        mastered = rng.rand() < prior["p_init"]
        for _ in range(n_steps):
            b = float(rng.rand())     # 题目难度 ∈[0,1]
            g_b, s_b = _item_adjust(prior["p_guess"], prior["p_slip"], b)
            p_correct = (1 - s_b) if mastered else g_b
            outcome = int(rng.rand() < p_correct)
            items.append(b); outcomes.append(outcome)
            if not mastered and rng.rand() < prior["p_transit"]:
                mastered = True

    # 两个预测器各自维护状态，同序消费 (难度, 结果)
    blind_state = new_state_from_prior(kc_id="sim", prior=prior)
    aware_state = new_state_from_prior(kc_id="sim", prior=prior)
    blind_pred, aware_pred = [], []
    # 注意：状态在学生间不重置，模拟同一长序列；仅为 AUC 对比，足够
    for b, outcome in zip(items, outcomes):
        blind_pred.append(predict_correct(state=blind_state, difficulty=None))
        aware_pred.append(predict_correct(state=aware_state, difficulty=b))
        bkt_update(state=blind_state, is_correct=bool(outcome), difficulty=None)
        bkt_update(state=aware_state, is_correct=bool(outcome), difficulty=b)

    auc_blind = roc_auc_score(outcomes, blind_pred)
    auc_aware = roc_auc_score(outcomes, aware_pred)
    assert auc_aware >= 0.65
    assert auc_aware >= auc_blind - 1e-9   # 难度感知不劣于难度盲


# ── R6: 固定难度连续答对单调不降且封顶 ─────────────────────────────────
def test_r6_monotone_nondecreasing_capped():
    s = mk(p_mastery=0.1)
    prev = s.current()
    for _ in range(30):
        bkt_update(state=s, is_correct=True, difficulty=0.8)
        assert s.p_mastery >= prev - 1e-12
        assert s.p_mastery <= _MASTERY_CAP
        prev = s.p_mastery
    assert s.p_mastery == pytest.approx(_MASTERY_CAP, abs=1e-6)
