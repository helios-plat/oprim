"""oprim sandbox contract — create / exec / files / patch / destroy.

One registry, multiple backends. Backends do not call each other.
Unknown isolation or a missing runtime fails honestly (no silent downgrade).
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from oprim._opensandbox import OpenSandboxBackend, reset_opensandbox_driver
from oprim._sandbox_backends import (
    DockerBackend,
    MemoryBackend,
    NetnsBackend,
    ProcessBackend,
    SandboxRecord,
)
from oprim._sandbox_profile import hosted_forbids_process

_ISOLATIONS = frozenset({"memory", "process", "netns", "docker", "opensandbox"})
_LOCK = threading.Lock()
_STORE: dict[str, SandboxRecord] = {}
_BACKENDS = {
    "memory": MemoryBackend(),
    "process": ProcessBackend(),
    "netns": NetnsBackend(),
    "docker": DockerBackend(),
    "opensandbox": OpenSandboxBackend(),
}


def _fail(**extra: Any) -> dict[str, Any]:
    rec = {"ok": False, "sandbox_id": "", "isolation": extra.pop("isolation", "")}
    rec.update(extra)
    return rec


def _ok(record: SandboxRecord, **extra: Any) -> dict[str, Any]:
    rec = {
        "ok": True,
        "sandbox_id": record.sandbox_id,
        "isolation": record.isolation,
        "block_network": record.block_network,
        "owner_id": record.owner_id,
    }
    rec.update(extra)
    return rec


def _owner_denied(record: SandboxRecord, owner_id: str) -> bool:
    if not owner_id or not record.owner_id:
        return False
    return owner_id != record.owner_id


def _get(sandbox_id: str) -> SandboxRecord | None:
    with _LOCK:
        return _STORE.get(sandbox_id)


def _jail(relpath: str) -> str | None:
    if not relpath or relpath.startswith("/") or Path(relpath).is_absolute():
        return None
    norm = Path(relpath).as_posix()
    if norm.startswith("../") or "/../" in f"/{norm}/" or norm == "..":
        return None
    if any(part == ".." for part in Path(relpath).parts):
        return None
    return norm


def _create_sandbox(
    *,
    isolation: str = "process",
    image: str = "",
    block_network: bool = True,
    cpu: str = "1",
    memory: str = "512m",
    timeout_s: int = 3600,
    workspace: str | None = None,
    env: dict[str, str] | None = None,
    owner_id: str = "",
    start: bool = True,
) -> dict[str, Any]:
    """Internal sandbox creation shared by sandbox and computer atomics."""
    if isolation not in _ISOLATIONS:
        return _fail(
            isolation=isolation,
            error=f"unknown isolation {isolation!r}; expected one of {sorted(_ISOLATIONS)}",
        )
    if isolation == "process" and hosted_forbids_process():
        return _fail(
            isolation=isolation,
            error="hosted profile forbids process isolation",
        )
    backend = _BACKENDS[isolation]
    sandbox_id = uuid.uuid4().hex[:16]
    created = backend.create(
        sandbox_id=sandbox_id,
        image=image,
        block_network=block_network,
        cpu=cpu,
        memory=memory,
        timeout_s=timeout_s,
        workspace=workspace,
        env=env or {},
        owner_id=owner_id,
        start=start,
    )
    if not created.get("ok"):
        return _fail(isolation=isolation, error=created.get("error") or "create failed")
    record = created["record"]
    with _LOCK:
        _STORE[sandbox_id] = record
    return _ok(record)


def sandbox_create(
    *,
    isolation: str = "process",
    image: str = "",
    block_network: bool = True,
    cpu: str = "1",
    memory: str = "512m",
    timeout_s: int = 3600,
    workspace: str | None = None,
    env: dict[str, str] | None = None,
    owner_id: str = "",
) -> dict[str, Any]:
    """Create and start a sandbox using the existing sandbox contract."""
    return _create_sandbox(
        isolation=isolation,
        image=image,
        block_network=block_network,
        cpu=cpu,
        memory=memory,
        timeout_s=timeout_s,
        workspace=workspace,
        env=env,
        owner_id=owner_id,
        start=True,
    )


def sandbox_exec(
    sandbox_id: str,
    argv: list[str],
    *,
    cwd: str = "",
    env: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    pty: bool = False,
    owner_id: str = "",
) -> dict[str, Any]:
    """Run ``argv`` inside an existing sandbox. ``pty=True`` is one-shot TTY, not attach."""
    record = _get(sandbox_id)
    if record is None:
        return {
            "ok": False,
            "sandbox_id": sandbox_id,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "error": f"unknown sandbox_id {sandbox_id}",
            "pty": False,
        }
    if _owner_denied(record, owner_id):
        return {
            "ok": False,
            "sandbox_id": sandbox_id,
            "isolation": record.isolation,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "error": "sandbox not owned by this user",
            "pty": False,
        }
    if not argv:
        return {
            "ok": False,
            "sandbox_id": sandbox_id,
            "isolation": record.isolation,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "error": "argv is empty",
            "pty": False,
        }
    jail_cwd = ""
    if cwd:
        jail_cwd = _jail(cwd) or ""
        if not jail_cwd:
            return {
                "ok": False,
                "sandbox_id": sandbox_id,
                "isolation": record.isolation,
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": f"cwd escapes workspace: {cwd}",
                "pty": False,
            }
    backend = _BACKENDS[record.isolation]
    return backend.exec_cmd(
        record, list(argv), cwd=jail_cwd, env=env or {}, timeout_s=timeout_s, pty=pty
    )


def sandbox_put_file(
    sandbox_id: str, path: str, content: str | bytes, *, owner_id: str = ""
) -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    if _owner_denied(record, owner_id):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error="sandbox not owned by this user",
        )
    rel = _jail(path)
    if rel is None:
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=f"path escapes workspace: {path}",
        )
    data = content.encode("utf-8") if isinstance(content, str) else content
    backend = _BACKENDS[record.isolation]
    written = backend.put_file(record, rel, data)
    if not written.get("ok"):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=written.get("error") or "put_file failed",
        )
    return _ok(record, path=rel, size=len(data))


def sandbox_get_file(sandbox_id: str, path: str, *, owner_id: str = "") -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    if _owner_denied(record, owner_id):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error="sandbox not owned by this user",
        )
    rel = _jail(path)
    if rel is None:
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=f"path escapes workspace: {path}",
        )
    backend = _BACKENDS[record.isolation]
    got = backend.get_file(record, rel)
    if not got.get("ok"):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=got.get("error") or "get_file failed",
        )
    return _ok(record, path=rel, content=got["content"], size=len(got["content"]))


def sandbox_list(sandbox_id: str, path: str = ".", *, owner_id: str = "") -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    if _owner_denied(record, owner_id):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error="sandbox not owned by this user",
        )
    rel = "." if path in {"", "."} else _jail(path)
    if rel is None:
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=f"path escapes workspace: {path}",
        )
    backend = _BACKENDS[record.isolation]
    listed = backend.list_dir(record, rel)
    if not listed.get("ok"):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=listed.get("error") or "list failed",
        )
    return _ok(record, path=rel, files=listed["files"])


def sandbox_apply_patch(sandbox_id: str, patch: str, *, owner_id: str = "") -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    if _owner_denied(record, owner_id):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error="sandbox not owned by this user",
        )
    backend = _BACKENDS[record.isolation]
    applied = backend.apply_patch(record, patch)
    if not applied.get("ok"):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error=applied.get("error") or "apply_patch failed",
        )
    return _ok(record, changed=applied.get("changed") or [])


def _destroy_sandbox(sandbox_id: str, *, owner_id: str = "") -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    if _owner_denied(record, owner_id):
        return _fail(
            isolation=record.isolation,
            sandbox_id=sandbox_id,
            error="sandbox not owned by this user",
        )
    with _LOCK:
        record = _STORE.pop(sandbox_id, None)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    backend = _BACKENDS[record.isolation]
    backend.destroy(record)
    return {
        "ok": True,
        "sandbox_id": sandbox_id,
        "isolation": record.isolation,
        "status": "deleted",
    }


def sandbox_destroy(sandbox_id: str, *, owner_id: str = "") -> dict[str, Any]:
    """Destroy a sandbox."""
    return _destroy_sandbox(sandbox_id, owner_id=owner_id)


def sandbox_heartbeat(sandbox_id: str) -> dict[str, Any]:
    record = _get(sandbox_id)
    if record is None:
        return _fail(error=f"unknown sandbox_id {sandbox_id}")
    return _ok(record, status="alive")


def reset_sandbox_runtime() -> None:
    """Test helper: destroy every live sandbox and clear the registry."""
    with _LOCK:
        ids = list(_STORE.keys())
    for sandbox_id in ids:
        _destroy_sandbox(sandbox_id)
    reset_opensandbox_driver()
