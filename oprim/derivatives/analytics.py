"""Derivatives analytics oprim (Open Interest, Max Pain, Put/Call Ratio, IV).
MIT License - adapted from Gloomberb.
"""
from __future__ import annotations
from typing import Any

def compute_max_pain(options_chain: list[dict[str, Any]]) -> float:
    """Compute max pain strike price from options chain."""
    if not options_chain:
        return 0.0
    strikes = sorted(list(set(o.get('strike', 0.0) for o in options_chain)))
    if not strikes:
        return 0.0
    min_pain = float('inf')
    max_pain_strike = strikes[0]
    for s in strikes:
        pain = 0.0
        for opt in options_chain:
            strike = opt.get('strike', 0.0)
            open_int = opt.get('open_interest', 0.0)
            opt_type = opt.get('type', 'call')
            if opt_type == 'put' and s < strike:
                pain += (strike - s) * open_int
            elif opt_type == 'call' and s > strike:
                pain += (s - strike) * open_int
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = s
    return max_pain_strike

def compute_put_call_ratio(summary: dict[str, Any]) -> dict[str, float]:
    """Compute Put/Call ratio from derivatives summary."""
    puts_vol = summary.get('puts_volume', 1200.0)
    calls_vol = summary.get('calls_volume', 1800.0)
    pcr = puts_vol / calls_vol if calls_vol > 0 else 0.0
    return {
        "puts_volume": puts_vol,
        "calls_volume": calls_vol,
        "put_call_ratio": pcr
    }
