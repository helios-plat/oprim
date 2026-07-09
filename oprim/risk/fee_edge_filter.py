"""Fee/edge viability filter — reject trades whose expected gross can't clear costs."""

from __future__ import annotations


def fee_edge_filter(
    notional_usd: float,
    *,
    atr_pct: float,
    taker_fee_rate: float,
    edge_multiple: float,
) -> dict:
    """Reject a trade whose expected gross move can't clear round-trip fees × margin.

    ``fee = notional * taker_fee_rate * 2`` (both legs);
    ``expected_gross = atr_pct * notional`` (one ATR of favourable move as the
    edge proxy); the trade passes only if
    ``expected_gross >= fee * edge_multiple``.

    Parameters
    ----------
    notional_usd : float
        Proposed trade notional in USD.
    atr_pct : float
        Normalized ATR (ATR / price) as the expected-move proxy, must be > 0.
    taker_fee_rate : float
        Per-leg taker fee rate (e.g. 0.0005 for 5 bps).
    edge_multiple : float
        Required ratio of expected gross to round-trip fees (e.g. 1.5).

    Returns
    -------
    dict
        ``{"passes": bool, "expected_gross": float, "fee": float,
        "min_gross": float}``.

    Raises
    ------
    ValueError
        If `atr_pct` <= 0 or `taker_fee_rate` < 0 or `edge_multiple` <= 0.

    References
    ----------
    .. [1] Extraction source: helixa services/risk-engine/src/checks.py fee/edge gate.
    """
    if atr_pct <= 0:
        raise ValueError(f"atr_pct must be > 0, got {atr_pct}")
    if taker_fee_rate < 0:
        raise ValueError(f"taker_fee_rate must be >= 0, got {taker_fee_rate}")
    if edge_multiple <= 0:
        raise ValueError(f"edge_multiple must be > 0, got {edge_multiple}")

    fee = notional_usd * taker_fee_rate * 2.0
    expected_gross = atr_pct * notional_usd
    min_gross = fee * edge_multiple
    return {
        "passes": expected_gross >= min_gross,
        "expected_gross": expected_gross,
        "fee": fee,
        "min_gross": min_gross,
    }
