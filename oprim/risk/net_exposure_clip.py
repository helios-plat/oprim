"""Correlation-aware net-exposure clip for a proposed position."""

from __future__ import annotations


def net_exposure_clip(
    proposed: float,
    *,
    correlated_positions: list[tuple[float, float]],
    max_net_exposure: float,
) -> dict:
    """Clip a proposed position so correlation-weighted net exposure stays bounded.

    ``net = proposed + sum(other_position * correlation for other_position,
    correlation in correlated_positions)``. If ``|net| > max_net_exposure``,
    `proposed` is clipped down to exactly hit the bound (or to 0 if the
    already-used exposure alone exceeds the bound).

    Parameters
    ----------
    proposed : float
        Proposed new position size (signed, same unit as `correlated_positions`
        sizes — typically position fraction or notional).
    correlated_positions : list[tuple[float, float]]
        ``[(other_position_size, correlation), ...]`` for every other symbol
        this instrument is correlated with.
    max_net_exposure : float
        Maximum allowed absolute net exposure. Must be > 0.

    Returns
    -------
    dict
        ``{"allowed_size": float, "net_exposure": float, "was_clipped": bool,
        "was_rejected": bool}``.

    Raises
    ------
    ValueError
        If `max_net_exposure` <= 0.

    References
    ----------
    .. [1] Extraction source: helixa project,
       services/risk-engine/src/position_manager.py:_corr_net_exposure
    """
    if max_net_exposure <= 0:
        raise ValueError(f"max_net_exposure must be > 0, got {max_net_exposure!r}")

    already_used = sum(pos * corr for pos, corr in correlated_positions)
    net = proposed + already_used

    if abs(net) <= max_net_exposure:
        return {
            "allowed_size": proposed,
            "net_exposure": net,
            "was_clipped": False,
            "was_rejected": False,
        }

    sign = 1.0 if proposed >= 0 else -1.0
    max_allowed = max(0.0, max_net_exposure - abs(already_used)) * sign

    return {
        "allowed_size": max_allowed,
        "net_exposure": already_used + max_allowed,
        "was_clipped": True,
        "was_rejected": max_allowed == 0.0,
    }
