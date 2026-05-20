"""Exponential-backoff retry with jitter."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from oprim._logging import log

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    name: str = "operation",
) -> T:
    """Run *fn* up to policy.max_attempts times with exponential back-off."""
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except policy.retryable_exceptions as exc:
            last_exc = exc
            if attempt == policy.max_attempts:
                log.error(
                    "retry_exhausted", name=name, attempts=attempt, error=str(exc)
                )
                raise

            delay = min(
                policy.base_delay_sec * (policy.exponential_base ** (attempt - 1)),
                policy.max_delay_sec,
            )
            if policy.jitter:
                delay *= 0.5 + random.random()

            log.warning(
                "retry_attempt",
                name=name,
                attempt=attempt,
                delay_sec=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]
