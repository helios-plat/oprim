"""OpenSandbox isolation backend + pluggable driver.

The oprim sandbox contract is synchronous. Drivers may wrap the OpenSandbox
SDK or a loopback (tests / local without a daemon). Hosted production refuses
the loopback unless a driver was explicitly injected.
"""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol

from oprim._sandbox_backends import (
    SandboxRecord,
    _apply_patch_to_store,
    _base_env,
    _run_argv,
    _sync_dir_to_store,
    _sync_store_to_dir,
    _text,
)
from oprim._sandbox_profile import sandbox_profile

_UNSET = object()
_INJECTED: Any = _UNSET


class OpenSandboxDriver(Protocol):
    def create(self, spec: dict[str, Any]) -> dict[str, Any]: ...
    def exec_cmd(
        self,
        remote_id: str,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout_s: float,
        pty: bool,
        workspace: Path,
    ) -> dict[str, Any]: ...
    def put_file(self, remote_id: str, path: str, data: bytes, workspace: Path) -> dict[str, Any]: ...
    def get_file(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]: ...
    def list_dir(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]: ...
    def apply_patch(self, remote_id: str, patch: str, workspace: Path) -> dict[str, Any]: ...
    def destroy(self, remote_id: str, workspace: Path, owned: bool) -> None: ...


class LoopbackOpenSandboxDriver:
    """Isolated tempdir + subprocess. Tests and local-without-daemon only."""

    def __init__(self) -> None:
        self._boxes: dict[str, dict[str, Any]] = {}

    def create(self, spec: dict[str, Any]) -> dict[str, Any]:
        given = spec.get("workspace")
        owned = not bool(given)
        root = Path(given) if given else Path(tempfile.mkdtemp(prefix="o3-osb-"))
        root.mkdir(parents=True, exist_ok=True)
        remote_id = uuid.uuid4().hex[:16]
        self._boxes[remote_id] = {
            "workspace": root,
            "owner_id": str(spec.get("owner_id") or ""),
            "owned": owned,
            "block_network": bool(spec.get("block_network", True)),
        }
        return {"ok": True, "remote_id": remote_id, "workspace": root, "owned": owned}

    def exec_cmd(
        self,
        remote_id: str,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout_s: float,
        pty: bool,
        workspace: Path,
    ) -> dict[str, Any]:
        box = self._boxes.get(remote_id)
        if box is None:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": f"unknown opensandbox remote_id {remote_id}",
                "pty": False,
            }
        work = workspace / cwd if cwd else workspace
        work.mkdir(parents=True, exist_ok=True)
        rec = _run_argv(argv, cwd=work, env=env, timeout_s=timeout_s, pty=pty)
        rec["block_network"] = bool(box.get("block_network", True))
        return rec

    def put_file(self, remote_id: str, path: str, data: bytes, workspace: Path) -> dict[str, Any]:
        if remote_id not in self._boxes:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        dest = workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"ok": True}

    def get_file(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]:
        if remote_id not in self._boxes:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        dest = workspace / path
        if not dest.is_file():
            return {"ok": False, "error": f"not found: {path}"}
        return {"ok": True, "content": dest.read_text(encoding="utf-8")}

    def list_dir(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]:
        if remote_id not in self._boxes:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        root = workspace if path == "." else workspace / path
        if not root.is_dir():
            return {"ok": False, "error": f"not a directory: {path}"}
        files = []
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            files.append(
                {
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else 0,
                }
            )
        return {"ok": True, "files": files}

    def apply_patch(self, remote_id: str, patch: str, workspace: Path) -> dict[str, Any]:
        if remote_id not in self._boxes:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        store: dict[str, bytes] = {}
        _sync_dir_to_store(store, workspace)
        result = _apply_patch_to_store(store, patch)
        if result.get("ok"):
            _sync_store_to_dir(store, workspace)
        return result

    def destroy(self, remote_id: str, workspace: Path, owned: bool) -> None:
        self._boxes.pop(remote_id, None)
        if owned:
            shutil.rmtree(workspace, ignore_errors=True)


class SdkOpenSandboxDriver:
    """Thin sync wrapper around the OpenSandbox Python SDK."""

    def __init__(self, client_factory: Any) -> None:
        self._client_factory = client_factory
        self._live: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> SdkOpenSandboxDriver | None:
        try:
            from datetime import timedelta

            from opensandbox import SandboxSync
            from opensandbox.config import ConnectionConfigSync
        except ImportError:
            return None
        domain = os.environ.get("OPEN_SANDBOX_DOMAIN", "localhost:8080").strip()
        api_key = os.environ.get("OPEN_SANDBOX_API_KEY", "").strip()
        protocol = os.environ.get("OPEN_SANDBOX_PROTOCOL", "http").strip() or "http"

        def factory(spec: dict[str, Any]) -> Any:
            kwargs: dict[str, Any] = {
                "domain": domain,
                "protocol": protocol,
            }
            if api_key:
                kwargs["api_key"] = api_key
            config = ConnectionConfigSync(**kwargs)
            create_kw: dict[str, Any] = {
                "connection_config": config,
                "timeout": timedelta(seconds=int(spec.get("timeout_s") or 600)),
                "resource": {
                    "cpu": spec.get("cpu") or "1",
                    "memory": spec.get("memory") or "512m",
                },
            }
            owner = str(spec.get("owner_id") or "")
            if owner:
                create_kw["metadata"] = {"veya.owner_id": owner}
            if spec.get("block_network", True):
                try:
                    from opensandbox.models.sandboxes import NetworkPolicy

                    create_kw["network_policy"] = NetworkPolicy(defaultAction="deny")
                except Exception:
                    pass
            image = spec.get("image") or "python:3.11-slim"
            return SandboxSync.create(image, **create_kw)

        return cls(factory)

    def create(self, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            sbx = self._client_factory(spec)
        except Exception as exc:
            return {"ok": False, "error": f"opensandbox create failed: {exc}"}
        remote_id = str(getattr(sbx, "id", None) or uuid.uuid4().hex[:16])
        given = spec.get("workspace")
        owned = not bool(given)
        root = Path(given) if given else Path(tempfile.mkdtemp(prefix="o3-osb-"))
        root.mkdir(parents=True, exist_ok=True)
        self._live[remote_id] = sbx
        return {"ok": True, "remote_id": remote_id, "workspace": root, "owned": owned}

    def exec_cmd(
        self,
        remote_id: str,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout_s: float,
        pty: bool,
        workspace: Path,
    ) -> dict[str, Any]:
        sbx = self._live.get(remote_id)
        if sbx is None:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": f"unknown opensandbox remote_id {remote_id}",
                "pty": False,
            }
        cmd = shlex.join(argv)
        if cwd:
            cmd = f"cd {shlex.quote(cwd)} && {cmd}"
        try:
            execution = sbx.commands.run(cmd)
            stdout_parts = getattr(getattr(execution, "logs", None), "stdout", None) or []
            stderr_parts = getattr(getattr(execution, "logs", None), "stderr", None) or []
            stdout = "".join(getattr(p, "text", str(p)) for p in stdout_parts)
            stderr = "".join(getattr(p, "text", str(p)) for p in stderr_parts)
            exit_code = int(getattr(execution, "exit_code", 0) or 0)
            return {
                "ok": exit_code == 0,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
                "error": "" if exit_code == 0 else stderr,
                "pty": bool(pty),
            }
        except Exception as exc:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error": str(exc),
                "pty": False,
            }

    def put_file(self, remote_id: str, path: str, data: bytes, workspace: Path) -> dict[str, Any]:
        dest = workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        sbx = self._live.get(remote_id)
        if sbx is None:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        try:
            from opensandbox.models import WriteEntry

            sbx.files.write_files(
                [WriteEntry(path=f"/workspace/{path}", data=_text(data), mode=644)]
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def get_file(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]:
        dest = workspace / path
        if dest.is_file():
            return {"ok": True, "content": dest.read_text(encoding="utf-8")}
        sbx = self._live.get(remote_id)
        if sbx is None:
            return {"ok": False, "error": f"unknown opensandbox remote_id {remote_id}"}
        try:
            content = sbx.files.read_file(f"/workspace/{path}")
            return {"ok": True, "content": content if isinstance(content, str) else _text(content)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def list_dir(self, remote_id: str, path: str, workspace: Path) -> dict[str, Any]:
        root = workspace if path == "." else workspace / path
        if root.is_dir():
            files = []
            for child in sorted(root.iterdir(), key=lambda p: p.name):
                files.append(
                    {
                        "name": child.name,
                        "type": "directory" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else 0,
                    }
                )
            return {"ok": True, "files": files}
        return {"ok": False, "error": f"not a directory: {path}"}

    def apply_patch(self, remote_id: str, patch: str, workspace: Path) -> dict[str, Any]:
        store: dict[str, bytes] = {}
        _sync_dir_to_store(store, workspace)
        result = _apply_patch_to_store(store, patch)
        if result.get("ok"):
            _sync_store_to_dir(store, workspace)
            for rel in result.get("changed") or []:
                data = store.get(rel)
                if data is not None:
                    self.put_file(remote_id, rel, data, workspace)
        return result

    def destroy(self, remote_id: str, workspace: Path, owned: bool) -> None:
        sbx = self._live.pop(remote_id, None)
        if sbx is not None:
            try:
                destroy = getattr(sbx, "destroy", None) or getattr(sbx, "kill", None)
                if destroy is not None:
                    destroy()
            except Exception:
                pass
        if owned:
            shutil.rmtree(workspace, ignore_errors=True)


def set_opensandbox_driver(driver: OpenSandboxDriver | None) -> None:
    global _INJECTED
    _INJECTED = driver


def reset_opensandbox_driver() -> None:
    global _INJECTED
    _INJECTED = _UNSET


def get_opensandbox_driver() -> OpenSandboxDriver | None:
    if _INJECTED is not _UNSET:
        return _INJECTED
    flag = os.environ.get("VEYA_OPENSANDBOX_DRIVER", "").strip().lower()
    if flag in {"loopback", "fake", "local"}:
        if sandbox_profile() == "hosted":
            return None
        return LoopbackOpenSandboxDriver()
    return SdkOpenSandboxDriver.from_env()


class OpenSandboxBackend:
    def create(self, **spec: Any) -> dict[str, Any]:
        driver = get_opensandbox_driver()
        if driver is None:
            return {
                "ok": False,
                "error": (
                    "opensandbox unavailable: no driver (install the opensandbox "
                    "SDK + server, or inject a driver in tests)"
                ),
            }
        created = driver.create(spec)
        if not created.get("ok"):
            return {"ok": False, "error": created.get("error") or "opensandbox create failed"}
        root = Path(created.get("workspace") or spec.get("workspace") or tempfile.mkdtemp(prefix="o3-osb-"))
        root.mkdir(parents=True, exist_ok=True)
        owned = bool(created.get("owned", not bool(spec.get("workspace"))))
        record = SandboxRecord(
            sandbox_id=spec["sandbox_id"],
            isolation="opensandbox",
            block_network=bool(spec.get("block_network", True)),
            workspace=root,
            image=str(spec.get("image") or "python:3.11-slim"),
            container_id=str(created.get("remote_id") or ""),
            env=_base_env(root, spec.get("env") or {}),
            cpu=spec.get("cpu") or "1",
            memory=spec.get("memory") or "512m",
            owned=owned,
            owner_id=str(spec.get("owner_id") or ""),
            handle=driver,
        )
        return {"ok": True, "record": record}

    def exec_cmd(
        self,
        record: SandboxRecord,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout_s: float,
        pty: bool = False,
    ) -> dict[str, Any]:
        driver: OpenSandboxDriver = record.handle or get_opensandbox_driver()
        if driver is None:
            return {
                "ok": False,
                "sandbox_id": record.sandbox_id,
                "isolation": "opensandbox",
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": "opensandbox driver missing",
                "pty": False,
            }
        merged = dict(record.env)
        merged.update(env)
        rec = driver.exec_cmd(
            record.container_id,
            argv,
            cwd=cwd,
            env=merged,
            timeout_s=timeout_s,
            pty=pty,
            workspace=record.workspace,
        )
        rec["sandbox_id"] = record.sandbox_id
        rec["isolation"] = "opensandbox"
        rec.setdefault("block_network", record.block_network)
        return rec

    def put_file(self, record: SandboxRecord, path: str, data: bytes) -> dict[str, Any]:
        driver: OpenSandboxDriver = record.handle or get_opensandbox_driver()
        if driver is None:
            return {"ok": False, "error": "opensandbox driver missing"}
        return driver.put_file(record.container_id, path, data, record.workspace)

    def get_file(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        driver: OpenSandboxDriver = record.handle or get_opensandbox_driver()
        if driver is None:
            return {"ok": False, "error": "opensandbox driver missing"}
        return driver.get_file(record.container_id, path, record.workspace)

    def list_dir(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        driver: OpenSandboxDriver = record.handle or get_opensandbox_driver()
        if driver is None:
            return {"ok": False, "error": "opensandbox driver missing"}
        return driver.list_dir(record.container_id, path, record.workspace)

    def apply_patch(self, record: SandboxRecord, patch: str) -> dict[str, Any]:
        driver: OpenSandboxDriver = record.handle or get_opensandbox_driver()
        if driver is None:
            return {"ok": False, "error": "opensandbox driver missing"}
        return driver.apply_patch(record.container_id, patch, record.workspace)

    def destroy(self, record: SandboxRecord) -> None:
        driver: OpenSandboxDriver | None = record.handle or get_opensandbox_driver()
        if driver is not None:
            driver.destroy(record.container_id, record.workspace, record.owned)
