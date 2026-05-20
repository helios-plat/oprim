"""HTTP REST client for external tools with circuit-breaker + retry."""
from __future__ import annotations

import time

import httpx

from oprim._logging import log

from .circuit_breaker import CircuitBreaker
from .errors import CircuitBreakerOpen, ExternalToolError
from .protocol import ExternalToolResponse, HealthStatus
from .retry_policy import RetryPolicy, with_retry


class HttpToolClient:
    """Async HTTP REST client for external tools.

    Wraps httpx with circuit breaker and exponential-backoff retry.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        timeout_seconds: int = 300,
        token: str | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name=f"{name}_cb")
        self.retry_policy = retry_policy or RetryPolicy()
        self._client = httpx.AsyncClient(timeout=float(timeout_seconds))

    async def health_check(self) -> HealthStatus:
        try:
            r = await self._client.get(f"{self.base_url}/health")
            data: dict = {}
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    pass
            return HealthStatus(
                healthy=(r.status_code == 200),
                version=data.get("version"),
                uptime_seconds=int(data.get("uptime_seconds", 0)),
                metadata=data,
            )
        except Exception as exc:
            log.warning(f"{self.name}_health_check_failed", error=str(exc))
            return HealthStatus(healthy=False)

    async def invoke(self, payload: dict) -> ExternalToolResponse:
        if not self.circuit_breaker.call_allowed():
            raise CircuitBreakerOpen(f"{self.name} circuit is OPEN — call rejected")

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async def _do() -> ExternalToolResponse:
            t0 = time.monotonic()
            r = await self._client.post(
                f"{self.base_url}/invoke",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            elapsed = int((time.monotonic() - t0) * 1000)
            return ExternalToolResponse(
                success=True,
                result=r.json(),
                elapsed_ms=elapsed,
            )

        try:
            result = await with_retry(_do, self.retry_policy, name=self.name)
            self.circuit_breaker.record_success()
            return result
        except Exception as exc:
            self.circuit_breaker.record_failure()
            raise ExternalToolError(f"{self.name} invoke failed: {exc}") from exc

    async def stream(self, payload: dict):
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        async with self._client.stream(
            "POST", f"{self.base_url}/stream", json=payload, headers=headers
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield {"data": line[6:]}

    async def close(self) -> None:
        await self._client.aclose()
