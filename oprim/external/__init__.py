"""oprim.external — external tool client infrastructure.

Provides a uniform Protocol + implementations for reaching external tools
(HTTP REST, MCP stdio) with circuit breaker and exponential-backoff retry.

Phase 11A: http_client + mcp_client (stdio).
Phase 11B: mcp_client (sse/websocket), whisper/tts/sd/searxng clients.
"""
from oprim.external.circuit_breaker import CircuitBreaker, CircuitState
from oprim.external.errors import (
    CircuitBreakerOpen,
    ExternalToolError,
    ExternalToolTimeout,
    ExternalToolUnavailable,
)
from oprim.external.health_check import aggregate_health
from oprim.external.http_client import HttpToolClient
from oprim.external.mcp_client import McpToolClient
from oprim.external.protocol import (
    ExternalToolClient,
    ExternalToolRequest,
    ExternalToolResponse,
    HealthStatus,
)
from oprim.external.retry_policy import RetryPolicy, with_retry

__all__ = [
    # Protocol + dataclasses
    "ExternalToolClient",
    "ExternalToolRequest",
    "ExternalToolResponse",
    "HealthStatus",
    # Implementations
    "HttpToolClient",
    "McpToolClient",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpen",
    # Retry
    "RetryPolicy",
    "with_retry",
    # Health
    "aggregate_health",
    # Errors
    "ExternalToolError",
    "ExternalToolTimeout",
    "ExternalToolUnavailable",
]
