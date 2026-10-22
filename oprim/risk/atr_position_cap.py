"""ATR-adaptive position-size cap — inversely scaled to normalized volatility."""

from __future__ import annotations


def atr_position_cap(
    atr_pct: float,
    *,
    risk_budget: float,
    min_position: float,
    max_position: float,
) -> float:
    """Position-size cap that shrinks as normalized volatility (ATR/close) rises.

    ``cap = clip(risk_budget / atr_pct, min_position, max_position)`` — a
    fixed fraction of equity is put "at risk" per trade (`risk_budget`), and
    the allowed position fraction is backed out from the current ATR so that
    a wider average true range yields a proportionally smaller position.

    Parameters
    ----------
    atr_pct : float
        Average True Range normalized by price (ATR / close), must be > 0.
    risk_budget : float
        Target risk fraction per trade, e.g. 0.01 for a 1% risk budget.
    min_position, max_position : float
        Hard floor/ceiling on the returned cap (position fraction, 0..1).

    Returns
    -------
    float
        Position-fraction cap in ``[min_position, max_position]``.

    Raises
    ------
    ValueError
        If `atr_pct` <= 0, `risk_budget` <= 0, or `min_position` > `max_position`.

    References
    ----------
    .. [1] Extraction source: helixa project,
       services/risk-engine/src/position_manager.py:_get_atr_cap
    """
    if atr_pct <= 0:
        raise ValueError(f"atr_pct must be > 0, got {atr_pct!r}")
    if risk_budget <= 0:
        raise ValueError(f"risk_budget must be > 0, got {risk_budget!r}")
    if min_position > max_position:
        raise ValueError(f"min_position ({min_position}) must be <= max_position ({max_position})")

    cap = risk_budget / atr_pct
    return max(min_position, min(max_position, cap))
