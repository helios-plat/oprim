"""MCP protocol client — Phase 11A implements stdio transport framework.

SSE / WebSocket transport left for Phase 11B (network-attached GPU tools).
"""
from __future__ import annotations

import asyncio
import time

from oprim._logging import log

from .circuit_breaker import CircuitBreaker
from .errors import ExternalToolError, ExternalToolUnavailable
from .protocol import ExternalToolResponse, HealthStatus
from .retry_policy import RetryPolicy, with_retry


class McpToolClient:
    """MCP protocol client for external tool servers.

    transport="stdio": launch local process, communicate via stdin/stdout.
    transport="sse"/"websocket": Phase 11B.
    """

    def __init__(
        self,
        name: str,
        transport: str,          # "stdio" | "sse" | "websocket"
        endpoint: str,           # command string (stdio) or URL (sse/ws)
        timeout_seconds: int = 300,
        circuit_breaker: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.name = name
        self.transport = transport
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name=f"{name}_cb")
        self.retry_policy = retry_policy or RetryPolicy()
        self._session = None
        self._connected = False

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self.transport == "stdio":
            await self._connect_stdio()
        elif self.transport in ("sse", "websocket"):
            raise ExternalToolUnavailable(
                f"{self.name}: {self.transport} transport reserved for Phase 11B"
            )
        else:
            raise ExternalToolUnavailable(f"{self.name}: unknown transport '{self.transport}'")

    async def _connect_stdio(self) -> None:
        """Connect to MCP server via stdio (local process)."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ExternalToolUnavailable(
                f"{self.name}: mcp SDK not available: {exc}"
            ) from exc

        import shlex
        cmd_parts = shlex.split(self.endpoint)
        params = StdioServerParameters(command=cmd_parts[0], args=cmd_parts[1:])

        # Create and store the session — caller is responsible for lifecycle
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        self._connected = True
        log.info("mcp_client_connected", name=self.name, transport=self.transport)

    async def health_check(self) -> HealthStatus:
        try:
            await self._ensure_connected()
            result = await self._session.list_tools()  # type: ignore[union-attr]
            tool_count = len(result.tools) if result else 0
            return HealthStatus(
                healthy=True,
                metadata={"tool_count": tool_count},
            )
        except Exception as exc:
            log.warning(f"{self.name}_mcp_health_failed", error=str(exc))
            return HealthStatus(healthy=False)

    async def invoke(self, payload: dict) -> ExternalToolResponse:
        if not self.circuit_breaker.call_allowed():
            from .errors import CircuitBreakerOpen
            raise CircuitBreakerOpen(f"{self.name} circuit is OPEN")

        async def _do() -> ExternalToolResponse:
            await self._ensure_connected()
            tool_name = payload.get("tool")
            tool_args = payload.get("args", {})
            if not tool_name:
                raise ExternalToolError(f"{self.name}: payload must contain 'tool' key")
            t0 = time.monotonic()
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, tool_args),  # type: ignore[union-attr]
                timeout=self.timeout_seconds,
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            content = result.content if result else []
            return ExternalToolResponse(
                success=True,
                result={"content": [c.model_dump() if hasattr(c, "model_dump") else str(c) for c in content]},
                elapsed_ms=elapsed,
            )

        try:
            resp = await with_retry(_do, self.retry_policy, name=self.name)
            self.circuit_breaker.record_success()
            return resp
        except Exception as exc:
            self.circuit_breaker.record_failure()
            raise ExternalToolError(f"{self.name} mcp invoke failed: {exc}") from exc

    async def stream(self, payload: dict):
        # MCP streaming not supported in Phase 11A
        raise NotImplementedError(f"{self.name}: MCP streaming reserved for Phase 11B")

    async def close(self) -> None:
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
            self._connected = False
