"""Triple-barrier labeling (López de Prado) for supervised trading models."""

from __future__ import annotations

import numpy as np


def triple_barrier_label(
    closes: np.ndarray | list,
    *,
    take_profit_pct: float,
    stop_loss_pct: float,
    horizon: int,
) -> np.ndarray:
    """Label each bar by which barrier its forward path hits first.

    For bar ``i``, scan the next ``horizon`` bars: whichever of the
    +``take_profit_pct`` (upper) or -``stop_loss_pct`` (lower) barrier the price
    crosses first sets the label (+1 / -1). If neither is hit within the
    horizon, the vertical (time) barrier gives label 0. The final ``horizon``
    bars are unlabelable and set to 0.

    Parameters
    ----------
    closes : array-like
        Close prices (chronological).
    take_profit_pct : float
        Upper barrier as a positive fraction, e.g. 0.015 for +1.5%.
    stop_loss_pct : float
        Lower barrier as a positive fraction, e.g. 0.010 for -1.0%.
    horizon : int
        Vertical (time) barrier in bars.

    Returns
    -------
    np.ndarray
        Integer labels in {-1, 0, 1}, same length as `closes`.

    Raises
    ------
    ValueError
        If `take_profit_pct` <= 0, `stop_loss_pct` <= 0, or `horizon` < 1.

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning, ch. 3.
    .. [2] Extraction source: helixa services/qlib-v2/src/label_engine.py.
    """
    if take_profit_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("take_profit_pct and stop_loss_pct must be > 0")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    c = np.asarray(closes, dtype=float)
    n = len(c)
    labels = np.zeros(n, dtype=int)

    for i in range(n - horizon):
        entry = c[i]
        up = entry * (1.0 + take_profit_pct)
        dn = entry * (1.0 - stop_loss_pct)
        window = c[i + 1 : i + 1 + horizon]
        hit_up = np.argmax(window >= up) if (window >= up).any() else -1
        hit_dn = np.argmax(window <= dn) if (window <= dn).any() else -1
        if hit_up == -1 and hit_dn == -1:
            labels[i] = 0
        elif hit_dn == -1:
            labels[i] = 1
        elif hit_up == -1:
            labels[i] = -1
        else:
            labels[i] = 1 if hit_up <= hit_dn else -1
    return labels
