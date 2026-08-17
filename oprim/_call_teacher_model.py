"""oprim.call_teacher_model — one teacher LLM call. Caller injected."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

TeacherCaller = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


async def call_teacher_model(
    messages: list[dict[str, Any]],
    *,
    caller: TeacherCaller,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Forward one chat to the injected teacher. Does not pick a vendor."""
    result = caller(messages=messages, max_tokens=max_tokens)
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[misc]
    if not isinstance(result, dict):
        return {"ok": False, "error": "teacher caller returned non-dict", "content": result}
    out = dict(result)
    out.setdefault("ok", True)
    return out
