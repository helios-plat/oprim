"""Technical indicator submodule."""

from oprim.technical.bands import bollinger_bands, donchian_channel, keltner_channels
from oprim.technical.exits import chandelier_exit
from oprim.technical.moving_averages import ema, macd, sma, vwap
from oprim.technical.oscillators import (
    rsi_normalized,
    stochastic_oscillator,
    cci,
    williams_r,
)
from oprim.technical.adaptive import kama
from oprim.technical.volume import obv, mfi
from oprim.technical.signals import (
    detect_ma_cross,
    detect_price_breakout,
    detect_volume_breakout,
    detect_ma_support_bounce,
    detect_volume_stagnation,
    detect_bullish_divergence,
    consecutive_event_count,
)

__all__ = [
    "sma",
    "ema",
    "vwap",
    "macd",
    "rsi_normalized",
    "stochastic_oscillator",
    "cci",
    "williams_r",
    "bollinger_bands",
    "donchian_channel",
    "keltner_channels",
    "chandelier_exit",
    "kama",
    "obv",
    "mfi",
    "detect_ma_cross",
    "detect_price_breakout",
    "detect_volume_breakout",
    "detect_ma_support_bounce",
    "detect_volume_stagnation",
    "detect_bullish_divergence",
    "consecutive_event_count",
]
