"""Sandbox backends for oprim._sandbox_env. One OS family each; no cross-calls."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oprim._apply_patch import apply_patch
from oprim._parse_unified_diff import parse_unified_diff


@dataclass
class SandboxRecord:
    sandbox_id: str
    isolation: str
    block_network: bool
    workspace: Path
    image: str = ""
    container_id: str = ""
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    cpu: str = "1"
    memory: str = "512m"
    owned: bool = True
    owner_id: str = ""
    handle: Any = None


def _text(data: bytes) -> str:
    return data.decode("utf-8")


def _apply_patch_to_store(store: dict[str, bytes], patch: str) -> dict[str, Any]:
    if not patch.strip():
        return {"ok": True, "changed": []}
    try:
        diffs = parse_unified_diff(patch)
    except Exception as exc:
        return {"ok": False, "error": f"parse patch failed: {exc}"}
    changed: list[str] = []
    for file_diff in diffs:
        raw_path = file_diff.new_path or file_diff.old_path
        rel = raw_path
        for prefix in ("a/", "b/", "./"):
            if rel.startswith(prefix):
                rel = rel[len(prefix) :]
        if not rel or rel == "/dev/null":
            continue
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            return {"ok": False, "error": f"patch path escapes workspace: {raw_path}"}
        original = _text(store.get(rel, b""))
        single = _file_patch_text(file_diff)
        try:
            store[rel] = apply_patch(original, patch=single).encode("utf-8")
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        changed.append(rel)
    return {"ok": True, "changed": changed}


def _file_patch_text(file_diff: Any) -> str:
    lines = [
        f"--- a/{file_diff.old_path}",
        f"+++ b/{file_diff.new_path}",
    ]
    for hunk in file_diff.hunks:
        suffix = f" {hunk.header}" if hunk.header else ""
        lines.append(
            f"@@ -{hunk.old_start},{hunk.old_count} "
            f"+{hunk.new_start},{hunk.new_count} @@{suffix}"
        )
        lines.extend(hunk.lines)
    return "\n".join(lines) + "\n"


def _sync_store_to_dir(store: dict[str, bytes], root: Path) -> None:
    for rel, data in store.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def _sync_dir_to_store(store: dict[str, bytes], root: Path) -> None:
    store.clear()
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_file():
            store[path.relative_to(root).as_posix()] = path.read_bytes()


def _run_argv(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    prefix: list[str] | None = None,
    pty: bool = False,
) -> dict[str, Any]:
    full = list(prefix or []) + list(argv)
    if pty:
        from oprim._hb_process import run_pty

        try:
            rec = run_pty(full, cwd=cwd, env=env, timeout_s=timeout_s)
        except OSError as exc:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error": str(exc),
                "pty": False,
            }
        except FileNotFoundError as exc:
            return {
                "ok": False,
                "exit_code": 127,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error": str(exc),
                "pty": False,
            }
        rec["pty"] = True
        return rec
    try:
        proc = subprocess.run(
            full,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": (proc.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (proc.stderr or b"").decode("utf-8", errors="replace"),
            "timed_out": False,
            "error": "",
            "pty": False,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "error": str(exc),
            "pty": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": "",
            "stderr": "timed out",
            "timed_out": True,
            "error": "timed out",
            "pty": False,
        }


def _base_env(workspace: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {
        "PATH": extra.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workspace),
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    env.update(extra)
    env["HOME"] = str(workspace)
    return env


class MemoryBackend:
    def create(self, **spec: Any) -> dict[str, Any]:
        given = spec.get("workspace")
        owned = not bool(given)
        root = Path(given) if given else Path(tempfile.mkdtemp(prefix="o3-mem-"))
        root.mkdir(parents=True, exist_ok=True)
        record = SandboxRecord(
            sandbox_id=spec["sandbox_id"],
            isolation="memory",
            block_network=False,
            workspace=root,
            env=_base_env(root, spec.get("env") or {}),
            owned=owned,
            owner_id=str(spec.get("owner_id") or ""),
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
        if pty:
            return {
                "ok": False,
                "sandbox_id": record.sandbox_id,
                "isolation": "memory",
                "block_network": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": "memory backend has no PTY",
                "pty": False,
            }
        _sync_store_to_dir(record.files, record.workspace)
        work = record.workspace / cwd if cwd else record.workspace
        work.mkdir(parents=True, exist_ok=True)
        merged = dict(record.env)
        merged.update(env)
        rec = _run_argv(argv, cwd=work, env=merged, timeout_s=timeout_s)
        _sync_dir_to_store(record.files, record.workspace)
        rec["sandbox_id"] = record.sandbox_id
        rec["isolation"] = record.isolation
        rec["block_network"] = False
        return rec

    def put_file(self, record: SandboxRecord, path: str, data: bytes) -> dict[str, Any]:
        record.files[path] = data
        dest = record.workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"ok": True}

    def get_file(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        if path in record.files:
            return {"ok": True, "content": _text(record.files[path])}
        dest = record.workspace / path
        if dest.is_file():
            return {"ok": True, "content": dest.read_text(encoding="utf-8")}
        return {"ok": False, "error": f"not found: {path}"}

    def list_dir(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        _sync_store_to_dir(record.files, record.workspace)
        root = record.workspace if path == "." else record.workspace / path
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

    def apply_patch(self, record: SandboxRecord, patch: str) -> dict[str, Any]:
        _sync_dir_to_store(record.files, record.workspace)
        result = _apply_patch_to_store(record.files, patch)
        if result.get("ok"):
            _sync_store_to_dir(record.files, record.workspace)
        return result

    def destroy(self, record: SandboxRecord) -> None:
        if record.owned:
            shutil.rmtree(record.workspace, ignore_errors=True)


class ProcessBackend:
    def create(self, **spec: Any) -> dict[str, Any]:
        given = spec.get("workspace")
        owned = not bool(given)
        root = Path(given) if given else Path(tempfile.mkdtemp(prefix="o3-proc-"))
        root.mkdir(parents=True, exist_ok=True)
        record = SandboxRecord(
            sandbox_id=spec["sandbox_id"],
            isolation="process",
            block_network=False,
            workspace=root,
            env=_base_env(root, spec.get("env") or {}),
            cpu=spec.get("cpu") or "1",
            memory=spec.get("memory") or "512m",
            owned=owned,
            owner_id=str(spec.get("owner_id") or ""),
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
        work = record.workspace / cwd if cwd else record.workspace
        work.mkdir(parents=True, exist_ok=True)
        merged = dict(record.env)
        merged.update(env)
        rec = _run_argv(argv, cwd=work, env=merged, timeout_s=timeout_s, pty=pty)
        rec["sandbox_id"] = record.sandbox_id
        rec["isolation"] = "process"
        rec["block_network"] = False
        return rec

    def put_file(self, record: SandboxRecord, path: str, data: bytes) -> dict[str, Any]:
        dest = record.workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"ok": True}

    def get_file(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        dest = record.workspace / path
        if not dest.is_file():
            return {"ok": False, "error": f"not found: {path}"}
        return {"ok": True, "content": dest.read_text(encoding="utf-8")}

    def list_dir(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        root = record.workspace if path == "." else record.workspace / path
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

    def apply_patch(self, record: SandboxRecord, patch: str) -> dict[str, Any]:
        store: dict[str, bytes] = {}
        _sync_dir_to_store(store, record.workspace)
        result = _apply_patch_to_store(store, patch)
        if result.get("ok"):
            _sync_store_to_dir(store, record.workspace)
        return result

    def destroy(self, record: SandboxRecord) -> None:
        if record.owned:
            shutil.rmtree(record.workspace, ignore_errors=True)


def unshare_available() -> bool:
    if shutil.which("unshare") is None:
        return False
    try:
        proc = subprocess.run(["unshare", "-Urn", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


class NetnsBackend(ProcessBackend):
    def create(self, **spec: Any) -> dict[str, Any]:
        if not unshare_available():
            return {
                "ok": False,
                "error": "netns unavailable: unshare -Urn failed or not installed",
            }
        created = super().create(**spec)
        if created.get("ok"):
            created["record"].isolation = "netns"
            created["record"].block_network = True
        return created

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
        work = record.workspace / cwd if cwd else record.workspace
        work.mkdir(parents=True, exist_ok=True)
        merged = dict(record.env)
        merged.update(env)
        rec = _run_argv(
            argv,
            cwd=work,
            env=merged,
            timeout_s=timeout_s,
            prefix=["unshare", "-Urn", "--"],
            pty=pty,
        )
        rec["sandbox_id"] = record.sandbox_id
        rec["isolation"] = "netns"
        rec["block_network"] = True
        return rec


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


class DockerBackend:
    def create(self, **spec: Any) -> dict[str, Any]:
        if not docker_available():
            return {"ok": False, "error": "docker unavailable: docker info failed"}
        image = spec.get("image") or "python:3.11-slim"
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=15,
        )
        if inspect.returncode != 0:
            return {"ok": False, "error": f"docker image not present: {image}"}
        given = spec.get("workspace")
        owned = not bool(given)
        root = Path(given) if given else Path(tempfile.mkdtemp(prefix="o3-dock-"))
        root.mkdir(parents=True, exist_ok=True)
        name = f"o3-sbx-{spec['sandbox_id']}"
        args = [
            "docker",
            "create",
            "--name",
            name,
            "--memory",
            spec.get("memory") or "512m",
            "--cpus",
            spec.get("cpu") or "1",
            "--workdir",
            "/workspace",
            "-v",
            f"{root}:/workspace",
        ]
        if spec.get("block_network", True):
            args.extend(["--network", "none"])
        args.extend([image, "sleep", "infinity"])
        created = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if created.returncode != 0:
            return {
                "ok": False,
                "error": (created.stderr or created.stdout or "docker create failed").strip(),
            }
        if spec.get("start", True):
            started = subprocess.run(
                ["docker", "start", name], capture_output=True, text=True, timeout=30
            )
            if started.returncode != 0:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
                return {
                    "ok": False,
                    "error": (started.stderr or started.stdout or "docker start failed").strip(),
                }
        record = SandboxRecord(
            sandbox_id=spec["sandbox_id"],
            isolation="docker",
            block_network=bool(spec.get("block_network", True)),
            workspace=root,
            image=image,
            container_id=name,
            env=_base_env(root, spec.get("env") or {}),
            cpu=spec.get("cpu") or "1",
            memory=spec.get("memory") or "512m",
            owned=owned,
            owner_id=str(spec.get("owner_id") or ""),
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
        work = "/workspace" if not cwd else f"/workspace/{cwd}"
        cmd = ["docker", "exec", "-w", work]
        if pty:
            cmd.append("-t")
        merged = dict(record.env)
        merged.update(env)
        for key, value in merged.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(record.container_id)
        cmd.extend(argv)
        rec = _run_argv(cmd, cwd=record.workspace, env=merged, timeout_s=timeout_s)
        rec["pty"] = bool(pty)
        rec["sandbox_id"] = record.sandbox_id
        rec["isolation"] = "docker"
        rec["block_network"] = record.block_network
        return rec

    def put_file(self, record: SandboxRecord, path: str, data: bytes) -> dict[str, Any]:
        dest = record.workspace / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"ok": True}

    def get_file(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        dest = record.workspace / path
        if not dest.is_file():
            return {"ok": False, "error": f"not found: {path}"}
        return {"ok": True, "content": dest.read_text(encoding="utf-8")}

    def list_dir(self, record: SandboxRecord, path: str) -> dict[str, Any]:
        root = record.workspace if path == "." else record.workspace / path
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

    def apply_patch(self, record: SandboxRecord, patch: str) -> dict[str, Any]:
        store: dict[str, bytes] = {}
        _sync_dir_to_store(store, record.workspace)
        result = _apply_patch_to_store(store, patch)
        if result.get("ok"):
            _sync_store_to_dir(store, record.workspace)
        return result

    def destroy(self, record: SandboxRecord) -> None:
        if record.container_id:
            subprocess.run(
                ["docker", "rm", "-f", record.container_id],
                capture_output=True,
                timeout=20,
            )
        if record.owned:
            shutil.rmtree(record.workspace, ignore_errors=True)

