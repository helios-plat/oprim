"""Technical signal detection primitives.

All functions are pure math on numeric arrays, no business concepts.
Market-agnostic: works on stocks, crypto, FX, commodities.

References:
- Murphy, J. J. (1999). Technical Analysis of the Financial Markets.
- Pring, M. J. (2014). Technical Analysis Explained, 5th ed.
- Nison, S. (1991). Japanese Candlestick Charting Techniques.
- Donchian, R. (1960). N-day high breakout rule.
"""

from __future__ import annotations

from typing import Literal, Optional

STABILITY = "experimental"


def _simple_ma(values: list[float], end_idx: int, period: int) -> Optional[float]:
    """Compute simple moving average ending at end_idx (exclusive)."""
    if end_idx < period or period <= 0:
        return None
    window = values[end_idx - period : end_idx]
    if any(v != v for v in window):  # NaN guard
        return None
    return sum(window) / period


def detect_ma_cross(
    closes: list[float],
    fast_period: int,
    slow_period: int,
    direction: Literal["golden", "death"],
) -> Optional[dict]:
    """Detect MA cross at the last bar.

    Returns {"crossed": bool, "fast_ma": float, "slow_ma": float,
             "prev_fast_ma": float, "prev_slow_ma": float}
    or None if insufficient data.

    Reference: Murphy (1999), Ch. 9 "Moving Averages".

    Parameters
    ----------
    closes : list of close prices, len >= slow_period + 1
    fast_period : fast MA window (e.g. 5)
    slow_period : slow MA window (e.g. 20)
    direction : "golden" (fast crosses above slow) or "death" (fast crosses below slow)
    """
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("fast_period and slow_period must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")
    if direction not in ("golden", "death"):
        raise ValueError('direction must be "golden" or "death"')

    n = len(closes)
    if n < slow_period + 1:
        return None

    fast_ma = _simple_ma(closes, n, fast_period)
    slow_ma = _simple_ma(closes, n, slow_period)
    prev_fast_ma = _simple_ma(closes, n - 1, fast_period)
    prev_slow_ma = _simple_ma(closes, n - 1, slow_period)

    if any(v is None for v in (fast_ma, slow_ma, prev_fast_ma, prev_slow_ma)):
        return None

    if direction == "golden":
        crossed = bool(prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma)
    else:
        crossed = bool(prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma)

    return {
        "crossed": crossed,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "prev_fast_ma": prev_fast_ma,
        "prev_slow_ma": prev_slow_ma,
    }


def detect_price_breakout(
    highs: list[float],
    closes: list[float],
    window: int,
) -> Optional[dict]:
    """Detect price breakout above N-bar high.

    Returns {"broke_out": bool, "current_close": float, "prior_max_high": float} or None.

    Reference: Donchian (1960), N-day high breakout rule.

    Parameters
    ----------
    highs : list of high prices, len >= window + 1
    closes : list of close prices, same len as highs
    window : lookback window (e.g. 60)
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if len(highs) != len(closes):
        raise ValueError("highs and closes must have the same length")

    n = len(highs)
    if n < window + 1:
        return None

    current_close = closes[-1]
    prior_max_high = max(highs[-(window + 1) : -1])
    broke_out = current_close > prior_max_high

    return {
        "broke_out": broke_out,
        "current_close": current_close,
        "prior_max_high": prior_max_high,
    }


def detect_volume_breakout(
    volumes: list[float],
    closes: list[float],
    highs: list[float],
    vol_ratio_threshold: float = 2.0,
) -> Optional[dict]:
    """Detect volume-confirmed breakout: current volume > N×MA + price > N-bar high.

    Returns {"broke_out": bool, "vol_ratio": float, "current_close": float} or None.

    Reference: Pring (2014), Ch. 13 "Volume Confirmation".

    Parameters
    ----------
    volumes : list of volume values, len >= 2
    closes : list of close prices, same len as volumes
    highs : list of high prices, same len as volumes
    vol_ratio_threshold : current volume / prior MA threshold (default 2.0)
    """
    n = len(volumes)
    if n < 2:
        return None
    if len(closes) != n or len(highs) != n:
        raise ValueError("volumes, closes, and highs must have the same length")

    prior_volumes = volumes[:-1]
    vol_ma = sum(prior_volumes) / len(prior_volumes)
    if vol_ma <= 0:
        return None

    vol_ratio = volumes[-1] / vol_ma
    prior_max_high = max(highs[:-1])
    broke_out = vol_ratio >= vol_ratio_threshold and closes[-1] > prior_max_high

    return {
        "broke_out": broke_out,
        "vol_ratio": vol_ratio,
        "current_close": closes[-1],
    }


def detect_ma_support_bounce(
    opens: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    ma_period: int,
    tolerance_pct: float = 0.015,
) -> Optional[dict]:
    """Detect MA support bounce: price touches MA within tolerance + closes higher + volume confirms.

    Returns {"bounced": bool, "ma_value": float, "touch_low": float} or None.

    Reference: Murphy (1999), Ch. 4 "Trends and Trendlines".

    Parameters
    ----------
    opens : list of open prices
    lows : list of low prices
    closes : list of close prices
    volumes : list of volume values
    ma_period : MA lookback period
    tolerance_pct : fractional tolerance for MA touch (default 1.5%)
    """
    if ma_period <= 0:
        raise ValueError("ma_period must be positive")

    n = len(closes)
    if n < ma_period + 1:
        return None
    if len(opens) != n or len(lows) != n or len(volumes) != n:
        raise ValueError("opens, lows, closes, and volumes must have the same length")

    ma_value = sum(closes[-(ma_period + 1) : -1]) / ma_period
    touch_low = lows[-1]

    if ma_value <= 0:
        return None

    touched = abs(touch_low - ma_value) / ma_value <= tolerance_pct
    closed_higher = closes[-1] > opens[-1]
    vol_confirm = volumes[-1] > volumes[-2] if n >= 2 else True

    bounced = touched and closed_higher and vol_confirm

    return {
        "bounced": bounced,
        "ma_value": ma_value,
        "touch_low": touch_low,
    }


def detect_volume_stagnation(
    opens: list[float],
    highs: list[float],
    closes: list[float],
    volumes: list[float],
    vol_ratio_threshold: float = 2.0,
) -> Optional[dict]:
    """Detect volume stagnation pattern: high volume + small body + long upper shadow.

    Returns {"stagnated": bool, "body_ratio": float, "vol_ratio": float} or None.

    Reference: Nison, S. (1991). Japanese Candlestick Charting Techniques.

    Parameters
    ----------
    opens : list of open prices
    highs : list of high prices
    closes : list of close prices
    volumes : list of volume values
    vol_ratio_threshold : current volume / prior MA threshold (default 2.0)
    """
    if not (len(opens) == len(highs) == len(closes) == len(volumes)):
        raise ValueError("opens, highs, closes, and volumes must have the same length")

    n = len(closes)
    if n < 2:
        return None

    prior_volumes = volumes[:-1]
    vol_ma = sum(prior_volumes) / len(prior_volumes)
    if vol_ma <= 0:
        return None

    vol_ratio = volumes[-1] / vol_ma

    body = abs(closes[-1] - opens[-1])
    candle_base = min(opens[-1], closes[-1])
    total_span = highs[-1] - candle_base
    body_ratio = (body / total_span) if total_span > 0 else 0.0

    stagnated = vol_ratio >= vol_ratio_threshold and body_ratio <= 0.3

    return {
        "stagnated": stagnated,
        "body_ratio": body_ratio,
        "vol_ratio": vol_ratio,
    }


def detect_bullish_divergence(
    prices: list[float],
    indicator: list[float],
) -> Optional[dict]:
    """Detect bullish divergence: price makes lower low, indicator makes higher low.

    Returns {"diverged": bool, "price_low_1": float, "price_low_2": float,
             "indicator_low_1": float, "indicator_low_2": float} or None.

    Reference: Murphy (1999), Ch. 10 "Oscillators and Divergences".

    Parameters
    ----------
    prices : list of prices (e.g. closes), len >= 4
    indicator : oscillator values aligned with prices, same len
    """
    if len(prices) < 4:
        return None
    if len(prices) != len(indicator):
        raise ValueError("prices and indicator must have the same length")

    n = len(prices)
    mid = n // 2

    first_prices = prices[:mid]
    second_prices = prices[mid:]
    first_ind = indicator[:mid]
    second_ind = indicator[mid:]

    idx1 = first_prices.index(min(first_prices))
    idx2 = second_prices.index(min(second_prices))

    price_low_1 = prices[idx1]
    price_low_2 = prices[mid + idx2]
    ind_low_1 = indicator[idx1]
    ind_low_2 = indicator[mid + idx2]

    diverged = price_low_2 < price_low_1 and ind_low_2 > ind_low_1

    return {
        "diverged": diverged,
        "price_low_1": price_low_1,
        "price_low_2": price_low_2,
        "indicator_low_1": ind_low_1,
        "indicator_low_2": ind_low_2,
    }


def consecutive_event_count(events: list[bool]) -> int:
    """Count consecutive True events at the tail of the list.

    Returns: number of trailing True values, e.g. [T,F,T,T,T] -> 3.

    Reference: standard run-length counting.

    Parameters
    ----------
    events : list of bool values
    """
    count = 0
    for v in reversed(events):
        if v:
            count += 1
        else:
            break
    return count
