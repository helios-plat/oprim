"""Provider call atomic."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


async def provider_call(request: Any, *, caller: Callable[[Any], Any]) -> Any:
    """Execute exactly one provider request through an injected adapter."""
    result = caller(request)
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["provider_call"]
