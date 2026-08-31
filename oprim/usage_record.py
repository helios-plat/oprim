"""Usage recording atomic."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


async def usage_record(record: Any, *, sink: Callable[[Any], Any] | None = None) -> Any:
    """Send one normalized usage record to the injected persistence sink."""
    if sink is None:
        return record.to_dict() if hasattr(record, "to_dict") else record
    result = sink(record)
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["usage_record"]
