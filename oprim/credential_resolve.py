"""Atomic, late-bound credential resolution."""

from __future__ import annotations

from typing import Any

from obase.tool_governance import CredentialRef, SecretRef


class CredentialResolutionError(RuntimeError):
    """Safe error for unavailable credentials; never includes a secret value."""


def _ref_id(ref: CredentialRef | SecretRef | str) -> str:
    if isinstance(ref, str):
        return ref
    return ref.id


async def credential_resolve(
    ref: CredentialRef | SecretRef | str,
    *,
    resolver: Any,
) -> str:
    """Resolve a reference through one injected vault callback."""
    identifier = _ref_id(ref)
    try:
        if hasattr(resolver, "resolve"):
            value = resolver.resolve(ref)
        elif hasattr(resolver, "get_secret"):
            value = resolver.get_secret(identifier)
        else:
            value = resolver(ref)
        if hasattr(value, "__await__"):
            value = await value
    except Exception as exc:
        raise CredentialResolutionError(
            f"credential resolution failed ({type(exc).__name__})"
        ) from exc
    if not isinstance(value, str) or not value:
        raise CredentialResolutionError("credential reference is missing or empty")
    return value


__all__ = ["CredentialResolutionError", "credential_resolve"]
