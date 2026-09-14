"""Compatibility module for the retired realtime quote primitive.

The implementation was removed from the canonical oprim surface; callers
should migrate to their project transport.  The symbols remain importable so
old test and integration modules fail at invocation rather than collection.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuoteResult:
    symbol: str
    price: float
    source: str
    ts: Any = None


class QuoteFetchError(RuntimeError):
    pass


async def realtime_quote_redis_fetch(**kwargs: Any) -> QuoteResult:
    raise QuoteFetchError("realtime_quote_redis_fetch is a compatibility-only API")


__all__ = ["QuoteResult", "QuoteFetchError", "realtime_quote_redis_fetch"]
