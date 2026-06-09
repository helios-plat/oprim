"""Cognitive modeling atomic operations (BKT, FSRS)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
from fsrs import Card, Rating, Scheduler


@dataclass
class KCState:
    """Knowledge Component State for Bayesian Knowledge Tracing.

    Attributes
    ----------
    kc_id : str
        Unique identifier for the knowledge component.
    p_init : float
        Initial probability of mastery.
    p_transit : float
        Probability of transitioning from non-mastery to mastery after an attempt.
    p_guess : float
        Probability of guessing correctly despite not mastering.
    p_slip : float
        Probability of slipping (incorrectly responding) despite mastering.
    p_mastery : float | None
        Current estimated probability of mastery.
    long_term_mastery : float | None
        EMA-smoothed long-term mastery estimate.
    last_interaction_ts : float | None
        Timestamp of the last interaction.
    n_attempts : int
        Total number of attempts.
    """

    kc_id: str
    p_init: float = 0.20
    p_transit: float = 0.20
    p_guess: float = 0.15
    p_slip: float = 0.12
    p_mastery: Optional[float] = None
    long_term_mastery: Optional[float] = None
    last_interaction_ts: Optional[float] = None
    n_attempts: int = 0

    def current(self) -> float:
        """Return the current mastery probability, falling back to p_init."""
        return self.p_mastery if self.p_mastery is not None else self.p_init


def bkt_new_state(
    *,
    kc_id: str,
    p_init: float = 0.20,
    p_transit: float = 0.20,
    p_guess: float = 0.15,
    p_slip: float = 0.12,
) -> KCState:
    """Create a new KCState with given prior parameters.

    Parameters
    ----------
    kc_id : str
        Knowledge component ID.
    p_init : float
        Initial mastery probability.
    p_transit : float
        Learning/transition probability.
    p_guess : float
        Guessing probability.
    p_slip : float
        Slipping probability.

    Returns
    -------
    KCState
        Initialized state.
    """
    if not all(0 <= p <= 1 for p in [p_init, p_transit, p_guess, p_slip]):
        raise ValueError("Probabilities must be in [0, 1]")

    return KCState(
        kc_id=kc_id,
        p_init=p_init,
        p_transit=p_transit,
        p_guess=p_guess,
        p_slip=p_slip,
    )


def bkt_update(
    *,
    state: KCState,
    correct: bool,
    current_ts: float | None = None,
    halflife: float | None = None,
    ema_alpha: float = 0.1,
    mastery_cap: float = 0.97,
) -> KCState:
    """Forgetting-aware BKT state update.

    Updates mastery probability based on a new observation (correct/incorrect).
    Includes mastery capping and EMA smoothing for long-term mastery.

    Parameters
    ----------
    state : KCState
        Current KC state.
    correct : bool
        Whether the response was correct.
    current_ts : float | None
        Current timestamp. Required for forgetting.
    halflife : float | None
        Half-life in days for mastery decay. If None, no forgetting is applied.
    ema_alpha : float
        Smoothing factor for long_term_mastery (default 0.1).
    mastery_cap : float
        Maximum allowed mastery probability (default 0.97).

    Returns
    -------
    KCState
        Updated state (pure function).
    """
    p_mastery = state.current()

    # 1. Forgetting (decay)
    if halflife is not None and state.last_interaction_ts is not None and current_ts is not None:
        if current_ts < state.last_interaction_ts:
            raise ValueError("current_ts cannot be earlier than last_interaction_ts")
        days = (current_ts - state.last_interaction_ts) / 86400.0
        # Simple exponential decay of mastery probability
        p_mastery *= exp_forgetting(days=days, halflife=halflife)

    # 2. Bayesian Update
    p_guess = state.p_guess
    p_slip = state.p_slip
    p_transit = state.p_transit

    if correct:
        p_known_given_obs = (p_mastery * (1 - p_slip)) / (
            p_mastery * (1 - p_slip) + (1 - p_mastery) * p_guess
        )
    else:
        p_known_given_obs = (p_mastery * p_slip) / (
            p_mastery * p_slip + (1 - p_mastery) * (1 - p_guess)
        )

    p_new = p_known_given_obs + (1 - p_known_given_obs) * p_transit

    # 3. Apply Cap
    p_new = min(p_new, mastery_cap)

    # 4. Long Term Mastery EMA
    ltm = state.long_term_mastery
    if ltm is None:
        ltm = p_new
    else:
        ltm = ema_alpha * p_new + (1 - ema_alpha) * ltm

    return replace(
        state,
        p_mastery=p_new,
        long_term_mastery=ltm,
        last_interaction_ts=current_ts if current_ts is not None else state.last_interaction_ts,
        n_attempts=state.n_attempts + 1,
    )


def bkt_classify_error(
    *,
    state: KCState,
    correct: bool,
) -> Literal["careless", "dontknow", "none"]:
    """Classify the type of error based on BKT state.

    If correct is False, evaluates if it's a 'careless' mistake (high mastery)
    or 'dontknow' (low mastery).

    Parameters
    ----------
    state : KCState
        Current KC state.
    correct : bool
        Whether the response was correct.

    Returns
    -------
    {"careless", "dontknow", "none"}
        Error classification.
    """
    if correct:
        return "none"

    p_mastery = state.current()
    p_slip = state.p_slip
    p_guess = state.p_guess

    # Probability of mistake being a slip: P(Slip | Incorrect)
    p_inc = p_mastery * p_slip + (1 - p_mastery) * (1 - p_guess)
    if p_inc == 0:
        return "dontknow"

    p_slip_given_inc = (p_mastery * p_slip) / p_inc
    p_dontknow_given_inc = ((1 - p_mastery) * (1 - p_guess)) / p_inc

    return "careless" if p_slip_given_inc > p_dontknow_given_inc else "dontknow"


def bkt_predict_correct(*, state: KCState) -> float:
    """Predict the probability of a correct response.

    P(Correct) = P(Mastery) * (1 - P(Slip)) + (1 - P(Mastery)) * P(Guess)

    Parameters
    ----------
    state : KCState
        Current KC state.

    Returns
    -------
    float
        Probability of correct response in [0, 1].
    """
    p_l = state.current()
    return p_l * (1 - state.p_slip) + (1 - p_l) * state.p_guess


def exp_forgetting(*, days: float, halflife: float) -> float:
    """Exponential forgetting approximation.

    R = 0.5^(days/halflife)

    Parameters
    ----------
    days : float
        Elapsed time in days.
    halflife : float
        Time in days when retention drops to 0.5.

    Returns
    -------
    float
        Retention (retrievability) in [0, 1].
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    if days < 0:
        raise ValueError("days must be non-negative")
    return 0.5 ** (days / halflife)


def fsrs_new_card() -> Card:
    """Create a new FSRS card.

    Returns
    -------
    fsrs.Card
        New initialized card.
    """
    return Card()


def fsrs_review(
    *,
    card: Card,
    rating: int | Rating,
    current_ts: float | None = None,
    weights: tuple[float, ...] | None = None,
) -> tuple[Card, Any]:
    """Review an FSRS card and return the updated card and review log.

    Parameters
    ----------
    card : Card
        The FSRS card being reviewed.
    rating : int | Rating
        Rating (1-4). 1: Again, 2: Hard, 3: Good, 4: Easy.
    current_ts : float | None
        Timestamp of the review. Defaults to current time.
    weights : tuple[float, ...] | None
        FSRS model weights. If None, uses default weights.

    Returns
    -------
    tuple[Card, ReviewLog]
        Updated card and the review log.
    """
    if isinstance(rating, int):
        rating = fsrs_map_rating(rating=rating)

    scheduler = Scheduler(w=weights) if weights else Scheduler()
    dt = datetime.fromtimestamp(current_ts, tz=timezone.utc) if current_ts else datetime.now(timezone.utc)

    return scheduler.review_card(card=card, rating=rating, review_datetime=dt)


def fsrs_retrievability(
    *,
    card: Card,
    current_ts: float | None = None,
) -> float:
    """Compute the current retrievability (R) of an FSRS card.

    Parameters
    ----------
    card : Card
        The FSRS card.
    current_ts : float | None
        Timestamp to evaluate at. Defaults to current time.

    Returns
    -------
    float
        Retrievability in [0, 1].
    """
    scheduler = Scheduler()
    dt = datetime.fromtimestamp(current_ts, tz=timezone.utc) if current_ts else datetime.now(timezone.utc)
    return float(scheduler.get_card_retrievability(card, dt))


def fsrs_map_rating(*, rating: int) -> Rating:
    """Map integer rating (1-4) to FSRS Rating enum.

    Parameters
    ----------
    rating : int
        Rating value. 1: Again, 2: Hard, 3: Good, 4: Easy.

    Returns
    -------
    fsrs.Rating
        Corresponding Rating enum.
    """
    mapping = {
        1: Rating.Again,
        2: Rating.Hard,
        3: Rating.Good,
        4: Rating.Easy,
    }
    if rating not in mapping:
        raise ValueError(f"Invalid FSRS rating: {rating}. Must be 1, 2, 3, or 4.")
    return mapping[rating]


def fsrs_due_date(*, card: Card) -> datetime:
    """Return the due date of an FSRS card.

    Parameters
    ----------
    card : Card
        The FSRS card.

    Returns
    -------
    datetime
        The next due date.
    """
    return card.due
