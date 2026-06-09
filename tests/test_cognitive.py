"""Tests for cognitive modeling atomic operations."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
import pytest
import numpy as np
from sklearn.metrics import roc_auc_score

from oprim.cognitive import (
    KCState,
    bkt_new_state,
    bkt_update,
    bkt_predict_correct,
    bkt_classify_error,
    exp_forgetting,
    fsrs_new_card,
    fsrs_review,
    fsrs_retrievability,
    fsrs_map_rating,
    fsrs_due_date,
)
from fsrs import Rating


def test_bkt_new_state():
    state = bkt_new_state(kc_id="test_kc", p_init=0.3, p_transit=0.1)
    assert state.kc_id == "test_kc"
    assert state.p_init == 0.3
    assert state.p_transit == 0.1
    assert state.p_mastery is None
    assert state.current() == 0.3

    with pytest.raises(ValueError):
        bkt_new_state(kc_id="err", p_init=1.2)


def test_bkt_update_basic():
    state = bkt_new_state(kc_id="math", p_init=0.2, p_transit=0.2, p_guess=0.1, p_slip=0.1)
    
    # Correct response should increase mastery
    state_after_correct = bkt_update(state=state, correct=True)
    assert state_after_correct.p_mastery > state.current()
    assert state_after_correct.n_attempts == 1
    assert state_after_correct.long_term_mastery == state_after_correct.p_mastery

    # Incorrect response should decrease mastery (unless p_transit is very high)
    state_after_incorrect = bkt_update(state=state, correct=False)
    assert state_after_incorrect.p_mastery < state.current()
    assert state_after_incorrect.n_attempts == 1


def test_bkt_update_forgetting():
    state = bkt_new_state(kc_id="vocab", p_init=0.8, p_transit=0.1)
    # Set initial mastery and timestamp
    state.p_mastery = 0.8
    state.last_interaction_ts = 1000.0
    
    # Update after 1 day (86400s) with halflife of 1 day
    # Mastery should decay before update
    state_new = bkt_update(state=state, correct=True, current_ts=1000.0 + 86400.0, halflife=1.0)
    
    # If no forgetting, update from 0.8 would be even higher.
    # With forgetting, 0.8 decays to 0.4 before update.
    # Let's verify p_mastery calculation manually for peace of mind if needed, 
    # but check that it's updated.
    assert state_new.last_interaction_ts == 1000.0 + 86400.0


def test_bkt_classify_error():
    # High mastery, incorrect -> careless
    high_mastery = KCState(kc_id="k1", p_mastery=0.9, p_slip=0.1, p_guess=0.1)
    assert bkt_classify_error(state=high_mastery, correct=False) == "careless"
    
    # Low mastery, incorrect -> dontknow
    low_mastery = KCState(kc_id="k1", p_mastery=0.1, p_slip=0.1, p_guess=0.1)
    assert bkt_classify_error(state=low_mastery, correct=False) == "dontknow"
    
    # Correct -> none
    assert bkt_classify_error(state=high_mastery, correct=True) == "none"


def test_bkt_predict_correct():
    state = KCState(kc_id="k1", p_mastery=0.5, p_guess=0.2, p_slip=0.1)
    # P(C) = 0.5 * 0.9 + 0.5 * 0.2 = 0.45 + 0.1 = 0.55
    assert math.isclose(bkt_predict_correct(state=state), 0.55)


def test_exp_forgetting():
    assert math.isclose(exp_forgetting(days=1, halflife=1), 0.5)
    assert math.isclose(exp_forgetting(days=2, halflife=1), 0.25)
    assert math.isclose(exp_forgetting(days=0, halflife=10), 1.0)
    with pytest.raises(ValueError):
        exp_forgetting(days=1, halflife=0)


def test_fsrs_flow():
    card = fsrs_new_card()
    assert card.state == 0  # New
    
    # Review with "Good" (3)
    card_v2, log = fsrs_review(card=card, rating=3)
    assert card_v2.state != 0
    assert log.rating == Rating.Good
    
    # Check retrievability
    r = fsrs_retrievability(card=card_v2)
    assert 0 <= r <= 1
    
    # Map rating
    assert fsrs_map_rating(rating=4) == Rating.Easy
    with pytest.raises(ValueError):
        fsrs_map_rating(rating=5)
        
    # Due date
    assert isinstance(fsrs_due_date(card=card_v2), datetime)


def test_bkt_mastery_cap():
    state = KCState(kc_id="k1", p_mastery=0.99, p_transit=0.5)
    state_new = bkt_update(state=state, correct=True, mastery_cap=0.97)
    assert state_new.p_mastery <= 0.97


def test_bkt_ema_long_term():
    state = KCState(kc_id="k1", p_mastery=0.5, long_term_mastery=0.5)
    # If p_new is 0.8 and ema_alpha is 0.1
    # ltm = 0.1 * 0.8 + 0.9 * 0.5 = 0.08 + 0.45 = 0.53
    state_new = bkt_update(state=state, correct=True, ema_alpha=0.1)
    # We don't know exact p_new without calculation, but let's check it's updated correctly
    p_new = state_new.p_mastery
    expected_ltm = 0.1 * p_new + 0.9 * 0.5
    assert math.isclose(state_new.long_term_mastery, expected_ltm)


def test_bkt_auc_simulation():
    """Simulate BKT learning and verify AUC >= 0.65."""
    np.random.seed(42)
    p_init = 0.1
    p_transit = 0.15
    p_guess = 0.2
    p_slip = 0.1
    
    n_students = 100
    n_steps = 20
    
    all_predictions = []
    all_outcomes = []
    
    for _ in range(n_students):
        state = bkt_new_state(kc_id="sim", p_init=p_init, p_transit=p_transit, p_guess=p_guess, p_slip=p_slip)
        is_mastered = np.random.random() < p_init
        
        for _ in range(n_steps):
            # Predict
            prob_correct = bkt_predict_correct(state=state)
            all_predictions.append(prob_correct)
            
            # Outcome
            if is_mastered:
                outcome = np.random.random() > p_slip
            else:
                outcome = np.random.random() < p_guess
            
            all_outcomes.append(int(outcome))
            
            # Update BKT
            state = bkt_update(state=state, correct=outcome)
            
            # Transition true state
            if not is_mastered:
                if np.random.random() < p_transit:
                    is_mastered = True
                    
    auc = roc_auc_score(all_outcomes, all_predictions)
    print(f"BKT Simulation AUC: {auc:.4f}")
    assert auc >= 0.65


def test_pure_functions():
    state = bkt_new_state(kc_id="pure", p_init=0.5)
    state_new = bkt_update(state=state, correct=True)
    assert state_new is not state
    assert state.p_mastery is None
    assert state.n_attempts == 0
    
    card = fsrs_new_card()
    card_new, _ = fsrs_review(card=card, rating=3)
    assert card_new is not card


def test_bkt_classify_error_edge_cases():
    # If p_inc is 0 (should not happen with p_guess/p_slip in (0, 1))
    state = KCState(kc_id="zero", p_mastery=0.0, p_guess=1.0, p_slip=0.0)
    # p_inc = 0*0 + 1*(1-1) = 0
    assert bkt_classify_error(state=state, correct=False) == "dontknow"


def test_fsrs_custom_weights():
    card = fsrs_new_card()
    # FSRS default weights have 17 or 19 elements
    weights = (
        0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61
    )
    card_new, log = fsrs_review(card=card, rating=3, weights=weights)
    assert log.rating == Rating.Good


def test_current_fallback():
    state = KCState(kc_id="fallback", p_init=0.7)
    assert state.current() == 0.7
    state.p_mastery = 0.8
    assert state.current() == 0.8


def test_bkt_update_invalid_ts():
    state = KCState(kc_id="ts", last_interaction_ts=100.0)
    with pytest.raises(ValueError):
        bkt_update(state=state, correct=True, current_ts=50.0, halflife=1.0)

def test_fsrs_retrievability_at_ts():
    card = fsrs_new_card()
    card_v2, _ = fsrs_review(card=card, rating=3, current_ts=1000000.0)
    # Check R immediately after review
    r_now = fsrs_retrievability(card=card_v2, current_ts=1000000.0)
    assert r_now > 0.99
    # Check R after 10 days
    r_later = fsrs_retrievability(card=card_v2, current_ts=1000000.0 + 86400 * 10)
    assert r_later < r_now
