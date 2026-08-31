"""Atomic secret read used only inside a trusted execution boundary."""

from __future__ import annotations

from typing import Any

from obase.tool_governance import SecretRef


class SecretReadError(RuntimeError):
    """Safe error for an unavailable secret."""


async def secret_read(ref: SecretRef | str, *, reader: Any) -> str:
    """Read one secret reference through an injected reader."""
    identifier = ref if isinstance(ref, str) else ref.id
    try:
        if hasattr(reader, "get_secret"):
            value = reader.get_secret(identifier)
        elif hasattr(reader, "read"):
            value = reader.read(ref)
        else:
            value = reader(ref)
        if hasattr(value, "__await__"):
            value = await value
    except Exception as exc:
        raise SecretReadError(f"secret read failed ({type(exc).__name__})") from exc
    if not isinstance(value, str) or not value:
        raise SecretReadError("secret reference is missing or empty")
    return value


__all__ = ["SecretReadError", "secret_read"]
