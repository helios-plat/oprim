"""Atomic computer lifecycle operations.

These operations reuse the existing sandbox runtime and ``obase.docker``
adapters.  They do not create a second process/container implementation or a
remote-worker abstraction.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from obase.computer import ComputerHandle, ComputerProfile, ComputerState

from oprim._sandbox_env import _create_sandbox, _destroy_sandbox, _get

_ISOLATION_BY_BACKEND: dict[str, str] = {"local": "process", "docker": "docker"}
_DOCKER_STATES: dict[str, ComputerState] = {
    "created": "created",
    "running": "running",
    "restarting": "running",
}


@dataclass
class _ComputerRecord:
    profile: ComputerProfile
    handle: ComputerHandle


_LOCK = threading.RLock()
_COMPUTERS: dict[str, _ComputerRecord] = {}


def _profile(value: ComputerProfile | Mapping[str, Any]) -> ComputerProfile:
    if isinstance(value, ComputerProfile):
        return value
    if isinstance(value, Mapping):
        return ComputerProfile(**dict(value))
    raise TypeError("profile must be a ComputerProfile or mapping")


def _handle(value: ComputerHandle | Mapping[str, Any]) -> ComputerHandle:
    if isinstance(value, ComputerHandle):
        return value
    if isinstance(value, Mapping):
        raw = dict(value)
        raw.pop("computer", None)
        return ComputerHandle(**raw)
    raise TypeError("handle must be a ComputerHandle or mapping")


def _record(value: ComputerHandle | Mapping[str, Any]) -> _ComputerRecord | None:
    candidate = _handle(value)
    with _LOCK:
        return _COMPUTERS.get(candidate.computer_id)


def _result(
    handle: ComputerHandle,
    *,
    operation: str,
    physical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "operation": operation,
        "computer": handle.to_dict(),
        "handle": handle.to_dict(),
        "status": handle.state,
    }
    if physical is not None:
        result["physical"] = dict(physical)
    return result


def _failure(operation: str, error: str) -> dict[str, Any]:
    return {"ok": False, "operation": operation, "status": "failed", "error": error}


def _docker_operation(
    operation: Literal["start", "stop", "restart", "inspect"],
    record: Any,
) -> Any:
    """Call the canonical obase Docker adapter; no Docker SDK lives here."""
    from obase.docker import (
        docker_container_inspect,
        docker_container_restart,
        docker_container_start,
        docker_container_stop,
    )

    container_id = str(record.container_id or "")
    if not container_id:
        raise RuntimeError("docker computer has no container handle")
    if operation == "start":
        return docker_container_start(container_id=container_id)
    if operation == "stop":
        return docker_container_stop(container_id=container_id)
    if operation == "restart":
        return docker_container_restart(container_id=container_id)
    return docker_container_inspect(container_id=container_id)


def _physical_status(computer: _ComputerRecord) -> tuple[ComputerState, dict[str, Any]]:
    handle = computer.handle
    if handle.backend == "local":
        return handle.state, {"backend": "local", "sandbox_id": handle.sandbox_id}
    inspected = _docker_operation("inspect", _get(handle.sandbox_id))
    state = _DOCKER_STATES.get(str(inspected.state), "stopped")
    if state == "running" and handle.attached:
        state = "attached"
    return state, inspected.model_dump() if hasattr(inspected, "model_dump") else inspected.dict()


def _store_handle(handle: ComputerHandle) -> None:
    with _LOCK:
        record = _COMPUTERS.get(handle.computer_id)
        if record is not None:
            _COMPUTERS[handle.computer_id] = replace(record, handle=handle)


def computer_create(
    profile: ComputerProfile | Mapping[str, Any],
) -> dict[str, Any]:
    """Create one local or Docker computer without starting user work."""
    operation = "create"
    try:
        selected = _profile(profile)
        isolation = _ISOLATION_BY_BACKEND[selected.backend]
        created = _create_sandbox(
            isolation=isolation,
            image=selected.image or "",
            block_network=selected.block_network,
            cpu=selected.cpu,
            memory=selected.memory,
            workspace=selected.workspace,
            owner_id=selected.owner_id,
            start=False,
        )
        if not created.get("ok"):
            return _failure(operation, str(created.get("error") or "computer create failed"))
        sandbox_id = str(created["sandbox_id"])
        sandbox_record = _get(sandbox_id)
        if sandbox_record is None:
            return _failure(operation, "sandbox create returned no runtime record")
        handle = ComputerHandle(
            computer_id=f"computer-{sandbox_id}",
            profile_id=selected.id,
            backend=selected.backend,
            workspace=selected.workspace,
            sandbox_id=sandbox_id,
            container_id=str(sandbox_record.container_id or ""),
            state="created",
            owner_id=selected.owner_id,
        )
        with _LOCK:
            _COMPUTERS[handle.computer_id] = _ComputerRecord(selected, handle)
        return _result(handle, operation=operation)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def computer_start(handle: ComputerHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Start a created or stopped computer; repeated starts are idempotent."""
    operation = "start"
    try:
        computer = _record(handle)
        if computer is None:
            return _failure(operation, "unknown computer handle")
        current = computer.handle
        if current.state not in {"running", "attached"}:
            sandbox_record = _get(current.sandbox_id)
            if sandbox_record is None:
                return _failure(operation, "underlying sandbox no longer exists")
            if current.backend == "docker":
                _docker_operation("start", sandbox_record)
            current = replace(current, state="running", attached=False)
            _store_handle(current)
        return _result(current, operation=operation)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def computer_status(handle: ComputerHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Read logical and physical status without changing the computer."""
    operation = "status"
    try:
        computer = _record(handle)
        if computer is None:
            return _failure(operation, "unknown computer handle")
        state, physical = _physical_status(computer)
        current = replace(
            computer.handle,
            state=state,
            attached=computer.handle.attached if state in {"running", "attached"} else False,
        )
        _store_handle(current)
        return _result(current, operation=operation, physical=physical)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def computer_attach(handle: ComputerHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Attach a local/client view to a running computer; no remote worker is created."""
    operation = "attach"
    try:
        computer = _record(handle)
        if computer is None:
            return _failure(operation, "unknown computer handle")
        state, physical = _physical_status(computer)
        if state not in {"running", "attached"}:
            return _failure(operation, f"computer is not running: {state}")
        current = replace(computer.handle, state="attached", attached=True)
        _store_handle(current)
        return _result(current, operation=operation, physical=physical)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def computer_stop(handle: ComputerHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Stop a computer while retaining its caller-owned workspace."""
    operation = "stop"
    try:
        computer = _record(handle)
        if computer is None:
            return _failure(operation, "unknown computer handle")
        current = computer.handle
        sandbox_record = _get(current.sandbox_id)
        if sandbox_record is None:
            return _failure(operation, "underlying sandbox no longer exists")
        if current.backend == "docker" and current.state in {"running", "attached"}:
            _docker_operation("stop", sandbox_record)
        current = replace(current, state="stopped", attached=False)
        _store_handle(current)
        return _result(current, operation=operation)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def computer_reset(handle: ComputerHandle | Mapping[str, Any]) -> dict[str, Any]:
    """Reinitialize runtime state without deleting or rewriting the workspace."""
    operation = "reset"
    try:
        computer = _record(handle)
        if computer is None:
            return _failure(operation, "unknown computer handle")
        current = computer.handle
        sandbox_record = _get(current.sandbox_id)
        if sandbox_record is None:
            return _failure(operation, "underlying sandbox no longer exists")
        if current.backend == "docker":
            if current.state in {"running", "attached"}:
                _docker_operation("restart", sandbox_record)
            else:
                _docker_operation("start", sandbox_record)
        current = replace(current, state="running", attached=False)
        _store_handle(current)
        return _result(current, operation=operation)
    except Exception as exc:
        return _failure(operation, f"{type(exc).__name__}: {exc}")


def _reset_computer_runtime() -> None:
    """Test-only cleanup for the process-local computer registry."""
    with _LOCK:
        records = list(_COMPUTERS.values())
        _COMPUTERS.clear()
    for record in records:
        _destroy_sandbox(record.handle.sandbox_id, owner_id=record.handle.owner_id)


__all__ = [
    "computer_attach",
    "computer_create",
    "computer_reset",
    "computer_start",
    "computer_status",
    "computer_stop",
]
