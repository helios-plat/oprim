"""Aggregate health check across multiple external tool clients."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from oprim._logging import log

from .protocol import HealthStatus

if TYPE_CHECKING:
    from .protocol import ExternalToolClient


async def aggregate_health(
    clients: dict[str, "ExternalToolClient"],
) -> dict[str, HealthStatus]:
    """Run health_check on all clients concurrently and return results."""

    async def _check(name: str, client: "ExternalToolClient") -> tuple[str, HealthStatus]:
        try:
            status = await client.health_check()
            return name, status
        except Exception as exc:
            log.error(f"{name}_health_check_exception", error=str(exc))
            return name, HealthStatus(healthy=False)

    results = await asyncio.gather(*[_check(n, c) for n, c in clients.items()])
    return dict(results)
