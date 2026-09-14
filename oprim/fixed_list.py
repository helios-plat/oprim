"""Build a validated immutable market universe specification."""
from __future__ import annotations

VALID_INSTRUMENT_TYPES = {"spot", "perpetual", "futures", "option", "index_future"}


def fixed_list(symbols: list[str], venue: str, instrument_type: str, *, market_metadata: dict | None = None) -> dict:
    if not symbols:
        raise ValueError("symbols must be non-empty")
    if instrument_type not in VALID_INSTRUMENT_TYPES:
        raise ValueError(f"instrument_type must be one of {sorted(VALID_INSTRUMENT_TYPES)}, got {instrument_type!r}")
    return {"venue": venue, "instrument_type": instrument_type, "symbols": list(symbols), "instrument_ids": [f"{sym}.{venue}" for sym in symbols], "metadata": dict(market_metadata) if market_metadata else {}}
