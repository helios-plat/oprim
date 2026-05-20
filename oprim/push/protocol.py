"""Push notification protocol — PushChannel interface + PushResult dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


@dataclass
class PushResult:
    channel: str
    success: bool
    recipient: str
    error_message: str | None = None
    sent_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class PushChannel(Protocol):
    """Interface for a single notification channel (web push, email, etc.)."""

    name: str

    async def send(
        self,
        recipient: str,
        title: str,
        body: str,
        deep_link: str | None = None,
        metadata: dict | None = None,
    ) -> PushResult: ...

    async def health_check(self) -> bool: ...
