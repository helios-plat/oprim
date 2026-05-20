"""ExternalToolClient Protocol + shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class HealthStatus:
    healthy: bool
    version: str | None = None
    uptime_seconds: int = 0
    metadata: dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExternalToolRequest:
    method: str          # "invoke" | "stream"
    payload: dict
    timeout_seconds: int = 300
    metadata: dict = field(default_factory=dict)


@dataclass
class ExternalToolResponse:
    success: bool
    result: dict | None = None
    error: str | None = None
    elapsed_ms: int = 0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ExternalToolClient(Protocol):
    """Uniform interface implemented by all external-tool clients."""

    name: str
    timeout_seconds: int

    async def health_check(self) -> HealthStatus: ...

    async def invoke(self, payload: dict) -> ExternalToolResponse: ...

    async def stream(self, payload: dict) -> AsyncIterator[dict]: ...

    async def close(self) -> None: ...
