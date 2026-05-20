"""Tests for oprim.external.http_client."""
from __future__ import annotations

import pytest
import respx
import httpx

from oprim.external.circuit_breaker import CircuitBreaker, CircuitState
from oprim.external.errors import CircuitBreakerOpen, ExternalToolError
from oprim.external.http_client import HttpToolClient
from oprim.external.retry_policy import RetryPolicy


BASE = "http://test-tool"


class TestHttpToolClientHealthCheck:
    @pytest.mark.asyncio
    @respx.mock
    async def test_health_ok(self):
        respx.get(f"{BASE}/health").mock(
            return_value=httpx.Response(200, json={"version": "1.0", "uptime_seconds": 42})
        )
        client = HttpToolClient(name="test", base_url=BASE)
        status = await client.health_check()
        await client.close()
        assert status.healthy
        assert status.version == "1.0"
        assert status.uptime_seconds == 42

    @pytest.mark.asyncio
    @respx.mock
    async def test_health_fail(self):
        respx.get(f"{BASE}/health").mock(return_value=httpx.Response(503))
        client = HttpToolClient(name="test", base_url=BASE)
        status = await client.health_check()
        await client.close()
        assert not status.healthy

    @pytest.mark.asyncio
    async def test_health_network_error(self):
        client = HttpToolClient(name="test", base_url="http://nowhere-xyz")
        status = await client.health_check()
        await client.close()
        assert not status.healthy


class TestHttpToolClientInvoke:
    @pytest.mark.asyncio
    @respx.mock
    async def test_invoke_success(self):
        respx.post(f"{BASE}/invoke").mock(
            return_value=httpx.Response(200, json={"result": "ok"})
        )
        client = HttpToolClient(
            name="test",
            base_url=BASE,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        resp = await client.invoke({"action": "test"})
        await client.close()
        assert resp.success
        assert resp.result == {"result": "ok"}
        assert resp.elapsed_ms >= 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_invoke_server_error_raises(self):
        respx.post(f"{BASE}/invoke").mock(return_value=httpx.Response(500))
        client = HttpToolClient(
            name="test",
            base_url=BASE,
            retry_policy=RetryPolicy(max_attempts=1, base_delay_sec=0),
        )
        with pytest.raises(ExternalToolError):
            await client.invoke({})
        await client.close()

    @pytest.mark.asyncio
    async def test_invoke_rejected_when_circuit_open(self):
        cb = CircuitBreaker(name="test_cb", failure_threshold=1, recovery_timeout_sec=9999)
        cb.record_failure()  # → OPEN
        assert cb.state == CircuitState.OPEN

        client = HttpToolClient(name="test", base_url=BASE, circuit_breaker=cb)
        with pytest.raises(CircuitBreakerOpen):
            await client.invoke({})
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_invoke_records_success_on_circuit(self):
        respx.post(f"{BASE}/invoke").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        cb = CircuitBreaker(name="test_cb")
        client = HttpToolClient(
            name="test",
            base_url=BASE,
            circuit_breaker=cb,
            retry_policy=RetryPolicy(max_attempts=1),
        )
        await client.invoke({})
        await client.close()
        assert cb._consecutive_failures == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_circuit_opens_after_repeated_failure(self):
        respx.post(f"{BASE}/invoke").mock(return_value=httpx.Response(503))
        cb = CircuitBreaker(name="test_cb", failure_threshold=3)
        client = HttpToolClient(
            name="test",
            base_url=BASE,
            circuit_breaker=cb,
            retry_policy=RetryPolicy(max_attempts=1, base_delay_sec=0),
        )
        for _ in range(3):
            with pytest.raises(ExternalToolError):
                await client.invoke({})
        assert cb.state == CircuitState.OPEN
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_bearer_token_sent(self):
        route = respx.post(f"{BASE}/invoke").mock(
            return_value=httpx.Response(200, json={})
        )
        client = HttpToolClient(
            name="test",
            base_url=BASE,
            token="secret",
            retry_policy=RetryPolicy(max_attempts=1),
        )
        await client.invoke({})
        await client.close()
        assert route.called
        req = route.calls[0].request
        assert req.headers["Authorization"] == "Bearer secret"
