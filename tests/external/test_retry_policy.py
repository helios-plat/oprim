"""Tests for oprim.external.retry_policy."""
from __future__ import annotations

import asyncio
import time

import pytest

from oprim.external.retry_policy import RetryPolicy, with_retry


class TestRetryPolicy:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        calls = []

        async def ok():
            calls.append(1)
            return "result"

        result = await with_retry(ok, RetryPolicy(max_attempts=3))
        assert result == "result"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        calls = []

        async def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("temporary")
            return "ok"

        result = await with_retry(
            flaky,
            RetryPolicy(max_attempts=3, base_delay_sec=0.01, jitter=False),
        )
        assert result == "ok"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        calls = []

        async def always_fails():
            calls.append(1)
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await with_retry(
                always_fails,
                RetryPolicy(max_attempts=3, base_delay_sec=0.01, jitter=False),
            )
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_without_jitter(self):
        timestamps: list[float] = []

        async def fail():
            timestamps.append(time.monotonic())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await with_retry(
                fail,
                RetryPolicy(
                    max_attempts=3,
                    base_delay_sec=0.05,
                    exponential_base=2.0,
                    jitter=False,
                ),
            )

        assert len(timestamps) == 3
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        assert gap1 >= 0.04  # ~0.05s
        assert gap2 >= 0.08  # ~0.10s
        assert gap2 > gap1   # exponential

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        calls = []

        async def raises_stop():
            calls.append(1)
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await with_retry(
                raises_stop,
                RetryPolicy(
                    max_attempts=3,
                    base_delay_sec=0.01,
                    retryable_exceptions=(ValueError,),
                ),
            )
        assert len(calls) == 1
