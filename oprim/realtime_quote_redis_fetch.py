"""Redis 实时行情拉取 + EOD 兜底 (oprim B8)."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from oprim._exceptions import OprimError
from oprim._protocols import CacheClient


class QuoteResult(BaseModel):
    """行情查询结果.

    Attributes:
        symbol:    标的代码.
        price:     最新价格.
        source:    数据来源 — ``"redis"`` / ``"eod_fallback"`` / ``"none"``.
        ts:        行情时间戳 (UTC); Redis 中存有时则填入, 否则为 ``None``.
    """

    symbol: str
    price: float
    source: Literal["redis", "eod_fallback", "none"]
    ts: datetime | None = None


class QuoteFetchError(OprimError):
    """Raised when both Redis and EOD fallback fail."""


async def realtime_quote_redis_fetch(
    *,
    symbol: str,
    redis_client: CacheClient,
    key_prefix: str = "tide:quote:",
    fallback_eod_fn: Callable[[str], Awaitable[float]] | None = None,
) -> QuoteResult:
    """Fetch real-time quote from Redis, falling back to an EOD price function.

    Redis key format: ``{key_prefix}{symbol}`` (default: ``"tide:quote:{symbol}"``).
    The value is expected to be either a plain float string (e.g. ``"12.34"``) or a
    JSON object with ``"price"`` and optional ``"ts"`` (ISO-8601 UTC string) fields.

    Args:
        symbol:         Canonical ticker symbol.
        redis_client:   Any object satisfying :class:`~oprim._protocols.CacheClient`
                        (async ``get(key) → str | None``).
        key_prefix:     Redis key prefix.  Defaults to ``"tide:quote:"``.
        fallback_eod_fn: Optional async callable ``(symbol) → float`` invoked when
                         Redis returns nothing.  ``None`` = no fallback.

    Returns:
        :class:`QuoteResult` with ``source="redis"`` on hit,
        ``"eod_fallback"`` on fallback, or ``"none"`` if both miss.

    Raises:
        QuoteFetchError: If Redis raises an unexpected exception.

    Example:
        >>> result = await realtime_quote_redis_fetch(symbol="600519", redis_client=r)
        >>> result.source
        'redis'
    """
    key = f"{key_prefix}{symbol}"
    try:
        raw = await redis_client.get(key)
    except Exception as exc:
        raise QuoteFetchError(f"Redis GET {key!r} failed: {exc}") from exc

    if raw is not None:
        price, ts = _parse_raw(raw)
        if price is not None:
            return QuoteResult(symbol=symbol, price=price, source="redis", ts=ts)

    if fallback_eod_fn is not None:
        try:
            eod_price = await fallback_eod_fn(symbol)
            return QuoteResult(symbol=symbol, price=float(eod_price), source="eod_fallback")
        except Exception:
            pass

    return QuoteResult(symbol=symbol, price=0.0, source="none")


def _parse_raw(raw: str | bytes) -> tuple[float | None, datetime | None]:
    """Parse a Redis value into (price, timestamp).  Returns (None, None) on failure."""
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    try:
        price = float(text)
        return price, None
    except ValueError:
        pass
    try:
        obj = json.loads(text)
        price = float(obj["price"])
        ts_str = obj.get("ts")
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc) if ts_str else None
        return price, ts
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, None
