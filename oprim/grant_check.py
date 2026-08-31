"""Atomic grant-to-request comparison."""

from __future__ import annotations

from datetime import datetime

from obase.tool_governance import Grant


def grant_check(
    grant: Grant | None,
    *,
    tool_identity: str,
    actor: str,
    effect: str,
    version: str,
    resource: str = "*",
    now: datetime | None = None,
) -> bool:
    """Return whether one grant authorizes one exact tool request."""
    if grant is None or not grant.is_valid(now=now):
        return False
    if grant.tool != tool_identity:
        return False
    if grant.subject not in {"", "*", actor}:
        return False
    if grant.tool_version not in {None, "", version}:
        return False
    if effect not in grant.allowed_effects and "*" not in grant.allowed_effects:
        return False
    return grant.resource in {"", "*", resource}


__all__ = ["grant_check"]
