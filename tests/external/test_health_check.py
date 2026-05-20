"""Tests for oprim.external.health_check."""
from __future__ import annotations

import pytest

from oprim.external.health_check import aggregate_health
from oprim.external.protocol import HealthStatus


class _FakeClient:
    def __init__(self, name: str, healthy: bool):
        self.name = name
        self.timeout_seconds = 30
        self._healthy = healthy

    async def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self._healthy)

    async def invoke(self, payload):
        raise NotImplementedError

    async def stream(self, payload):
        raise NotImplementedError

    async def close(self):
        pass


class _BrokenClient(_FakeClient):
    async def health_check(self) -> HealthStatus:
        raise RuntimeError("boom")


class TestAggregateHealth:
    @pytest.mark.asyncio
    async def test_all_healthy(self):
        clients = {"a": _FakeClient("a", True), "b": _FakeClient("b", True)}
        result = await aggregate_health(clients)
        assert result["a"].healthy
        assert result["b"].healthy

    @pytest.mark.asyncio
    async def test_one_unhealthy(self):
        clients = {"a": _FakeClient("a", True), "b": _FakeClient("b", False)}
        result = await aggregate_health(clients)
        assert result["a"].healthy
        assert not result["b"].healthy

    @pytest.mark.asyncio
    async def test_exception_returns_unhealthy(self):
        clients = {"ok": _FakeClient("ok", True), "broken": _BrokenClient("broken", True)}
        result = await aggregate_health(clients)
        assert result["ok"].healthy
        assert not result["broken"].healthy

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        result = await aggregate_health({})
        assert result == {}
