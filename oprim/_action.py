"""Atomic Action Gateway operations.

The functions in this module only invoke injected callables.  Policy
composition, approval, and transaction sequencing belong to oskill/omodul.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from obase.action import ActionRequest, AuditRecord, PolicyRule


def policy_match(rule: PolicyRule, request: ActionRequest) -> bool:
    """Match one rule against one request; no priority or policy composition."""
    return all(
        expected in {"", "*"} or expected == actual
        for expected, actual in (
            (rule.action, request.action),
            (rule.effect, request.effect),
            (rule.resource, request.resource),
        )
    )


async def audit_append(
    record: AuditRecord,
    writer: Callable[[AuditRecord], Any] | Callable[[AuditRecord], Awaitable[Any]],
) -> Any:
    """Append one audit record through the injected persistence writer."""
    result = writer(record)
    if inspect.isawaitable(result):
        return await result
    return result


async def tool_invoke(
    request: ActionRequest,
    executor: Callable[[ActionRequest], Any] | Callable[[ActionRequest], Awaitable[Any]],
) -> Any:
    """Invoke one injected physical executor."""
    result = executor(request)
    if inspect.isawaitable(result):
        return await result
    return result


async def side_effect_record(
    *,
    request: ActionRequest,
    operation_key: str,
    operation_type: str,
    target_ref: str,
    provider: Callable[[], Any] | Callable[[], Awaitable[Any]],
    recorder: Callable[..., Any] | Callable[..., Awaitable[Any]],
    capability: str = "manual_only",
) -> Any:
    """Record one side effect through an injected ledger adapter.

    The recorder owns durable idempotency.  This primitive never stores or
    duplicates a ledger, and never calls another oprim.
    """
    result = recorder(
        request=request,
        operation_key=operation_key,
        operation_type=operation_type,
        target_ref=target_ref,
        provider=provider,
        capability=capability,
    )
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["audit_append", "policy_match", "side_effect_record", "tool_invoke"]
