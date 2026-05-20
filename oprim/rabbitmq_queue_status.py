"""oprim.rabbitmq_queue_status — Query a single RabbitMQ queue's live status."""
from __future__ import annotations

from urllib.parse import quote

import httpx
from obase.tool_registry import register_tool
from pydantic import BaseModel, Field

from oprim._exceptions import OprimError


class RabbitMQQueueStatus(BaseModel):
    """Subset of RabbitMQ Management API /api/queues/{vhost}/{name} response."""

    queue_name: str
    vhost: str
    messages_ready: int = Field(description="Messages awaiting consumer pickup")
    messages_unacknowledged: int = Field(description="Delivered but unacked")
    consumers: int = Field(description="Number of active consumers")
    state: str = Field(description="running / idle / flow / down")
    memory_bytes: int = Field(description="Queue memory consumption")
    publish_rate_per_sec: float = Field(description="Publish rate")
    deliver_rate_per_sec: float = Field(description="Deliver rate")


@register_tool(  # type: ignore[untyped-decorator]
    permission="read",
    requires_secrets=["aegis/{env}/rabbitmq_mgmt_password"],
)
def rabbitmq_queue_status(
    *,
    mgmt_url: str,
    queue_name: str,
    vhost: str = "/",
    username: str = "guest",
    password: str = "guest",
    timeout_seconds: float = 10.0,
) -> RabbitMQQueueStatus:
    """Query a single RabbitMQ queue's live status via Management API.

    Args:
        mgmt_url: RabbitMQ Management base URL (e.g. http://rabbit.internal:15672)
        queue_name: Queue name to query
        vhost: Virtual host (default "/")
        username: RabbitMQ Management user
        password: RabbitMQ Management password
        timeout_seconds: HTTP timeout in seconds

    Returns:
        RabbitMQQueueStatus with key metrics.

    Raises:
        OprimError: HTTP non-2xx, connection error, timeout, or queue not found.
    """
    vhost_encoded = quote(vhost, safe="")
    url = f"{mgmt_url.rstrip('/')}/api/queues/{vhost_encoded}/{quote(queue_name, safe='')}"
    try:
        with httpx.Client(timeout=timeout_seconds, auth=(username, password)) as client:
            response = client.get(url)
    except httpx.TimeoutException as exc:
        raise OprimError(
            f"RabbitMQ Management API timeout after {timeout_seconds}s: {exc}"
        ) from exc
    except httpx.ConnectError as exc:
        raise OprimError(f"Failed to connect to RabbitMQ Management API: {exc}") from exc

    if response.status_code == 404:
        raise OprimError(f"Queue not found: vhost={vhost!r}, name={queue_name!r}")
    if response.status_code == 401:
        raise OprimError("RabbitMQ Management API authentication failed")
    if not response.is_success:
        raise OprimError(
            f"RabbitMQ Management API returned {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    stats = data.get("message_stats", {})
    return RabbitMQQueueStatus(
        queue_name=data["name"],
        vhost=data["vhost"],
        messages_ready=data.get("messages_ready", 0),
        messages_unacknowledged=data.get("messages_unacknowledged", 0),
        consumers=data.get("consumers", 0),
        state=data.get("state", "unknown"),
        memory_bytes=data.get("memory", 0),
        publish_rate_per_sec=stats.get("publish_details", {}).get("rate", 0.0),
        deliver_rate_per_sec=stats.get("deliver_get_details", {}).get("rate", 0.0),
    )
