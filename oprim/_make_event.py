"""make_event — construct a validated Event with a generated id and timestamp."""

from __future__ import annotations

import time
from typing import Any

from ._hicode_types import Event


def make_event(*, event_type: str, payload: dict[str, Any]) -> Event:
    """Return a new :class:`Event` with a generated uuid7 id and current timestamp.

    Raises :class:`ValueError` if *event_type* is empty.
    """
    if not event_type:
        raise ValueError("event type must not be empty")

    from obase import uuid7

    return Event(
        id=str(uuid7()),
        type=event_type,
        payload=payload,
        timestamp=time.time(),
    )
