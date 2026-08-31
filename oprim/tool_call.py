"""Atomic native tool invocation."""

from __future__ import annotations

import inspect
from typing import Any

from obase.tool_governance import ToolCallRequest


async def tool_call(request: ToolCallRequest, *, executor: Any) -> Any:
    """Invoke one injected native physical executor exactly once."""
    result = executor(request)
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["tool_call"]
