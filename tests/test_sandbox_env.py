"""G0/G1 sandbox contract tests."""

from __future__ import annotations

import sys

import pytest

from oprim._sandbox_backends import docker_available, unshare_available
from oprim._sandbox_env import (
    reset_sandbox_runtime,
    sandbox_apply_patch,
    sandbox_create,
    sandbox_destroy,
    sandbox_exec,
    sandbox_get_file,
    sandbox_list,
    sandbox_put_file,
)


@pytest.fixture(autouse=True)
def _clean_runtime() -> None:
    reset_sandbox_runtime()
    yield
    reset_sandbox_runtime()


def test_memory_create_put_get_list_destroy() -> None:
    created = sandbox_create(isolation="memory")
    assert created["ok"] is True
    sid = created["sandbox_id"]
    put = sandbox_put_file(sid, "src/hello.txt", "hi\n")
    assert put["ok"] is True
    got = sandbox_get_file(sid, "src/hello.txt")
    assert got["content"] == "hi\n"
    listed = sandbox_list(sid, "src")
    assert any(item["name"] == "hello.txt" for item in listed["files"])
    dead = sandbox_destroy(sid)
    assert dead["status"] == "deleted"
    again = sandbox_exec(sid, [sys.executable, "-c", "print(1)"])
    assert again["ok"] is False
    assert "unknown sandbox_id" in again["error"]


def test_path_jail_rejects_escape() -> None:
    sid = sandbox_create(isolation="memory")["sandbox_id"]
    for bad in ("../secret", "/etc/passwd", "foo/../../x"):
        rec = sandbox_put_file(sid, bad, "x")
        assert rec["ok"] is False
        assert "escapes" in rec["error"]


def test_apply_patch_and_exec() -> None:
    sid = sandbox_create(isolation="memory")["sandbox_id"]
    sandbox_put_file(sid, "hello.txt", "old\n")
    patch = """--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-old
+new
"""
    rec = sandbox_apply_patch(sid, patch)
    assert rec["ok"] is True
    assert "hello.txt" in rec["changed"]
    assert sandbox_get_file(sid, "hello.txt")["content"] == "new\n"
    ran = sandbox_exec(sid, [sys.executable, "-c", "print(open('hello.txt').read())"])
    assert ran["ok"] is True
    assert "new" in ran["stdout"]


def test_unknown_isolation_fails_honestly() -> None:
    rec = sandbox_create(isolation="k8s")
    assert rec["ok"] is False
    assert "unknown isolation" in rec["error"]


def test_process_reports_network_not_blocked() -> None:
    rec = sandbox_create(isolation="process")
    assert rec["ok"] is True
    assert rec["isolation"] == "process"
    assert rec["block_network"] is False
    ran = sandbox_exec(rec["sandbox_id"], [sys.executable, "-c", "print(2+2)"])
    assert ran["ok"] is True
    assert "4" in ran["stdout"]
    assert ran["block_network"] is False


def test_netns_create_fails_honestly_when_unavailable() -> None:
    if unshare_available():
        rec = sandbox_create(isolation="netns")
        assert rec["ok"] is True
        assert rec["block_network"] is True
        sandbox_destroy(rec["sandbox_id"])
        return
    rec = sandbox_create(isolation="netns")
    assert rec["ok"] is False
    assert "unshare" in rec["error"]


def test_caller_workspace_survives_destroy(tmp_path) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("stay\n", encoding="utf-8")
    rec = sandbox_create(isolation="process", workspace=str(tmp_path))
    assert rec["ok"] is True
    sandbox_destroy(rec["sandbox_id"])
    assert marker.read_text(encoding="utf-8") == "stay\n"


def test_process_pty_is_a_tty() -> None:
    rec = sandbox_create(isolation="process")
    assert rec["ok"] is True
    sid = rec["sandbox_id"]
    piped = sandbox_exec(
        sid,
        [sys.executable, "-c", "import sys; print('TTY' if sys.stdout.isatty() else 'PIPE')"],
    )
    assert piped["ok"] is True
    assert piped["pty"] is False
    assert "PIPE" in piped["stdout"]
    tty = sandbox_exec(
        sid,
        [sys.executable, "-c", "import sys; print('TTY' if sys.stdout.isatty() else 'PIPE')"],
        pty=True,
    )
    assert tty["ok"] is True
    assert tty["pty"] is True
    assert "TTY" in tty["stdout"]


def test_memory_pty_fails_honestly() -> None:
    sid = sandbox_create(isolation="memory")["sandbox_id"]
    rec = sandbox_exec(sid, [sys.executable, "-c", "print(1)"], pty=True)
    assert rec["ok"] is False
    assert rec["pty"] is False
    assert "PTY" in rec["error"]


def test_docker_create_fails_honestly_when_unavailable() -> None:
    if docker_available():
        rec = sandbox_create(isolation="docker", image="python:3.11-slim")
        if rec["ok"]:
            sandbox_destroy(rec["sandbox_id"])
            return
        assert "docker" in rec["error"].lower() or "image" in rec["error"].lower()
        return
    rec = sandbox_create(isolation="docker")
    assert rec["ok"] is False
    assert "docker" in rec["error"]


def test_hosted_profile_forbids_process(monkeypatch) -> None:
    monkeypatch.setenv("VEYA_SANDBOX_PROFILE", "hosted")
    rec = sandbox_create(isolation="process")
    assert rec["ok"] is False
    assert "hosted profile forbids process" in rec["error"]


def test_opensandbox_two_users_cannot_share(monkeypatch) -> None:
    from oprim._opensandbox import LoopbackOpenSandboxDriver, set_opensandbox_driver

    set_opensandbox_driver(LoopbackOpenSandboxDriver())
    a = sandbox_create(isolation="opensandbox", owner_id="alice")
    b = sandbox_create(isolation="opensandbox", owner_id="bob")
    assert a["ok"] and b["ok"]
    assert a["owner_id"] == "alice"
    put = sandbox_put_file(a["sandbox_id"], "secret.txt", "alice-only")
    assert put["ok"] is True
    stolen = sandbox_get_file(a["sandbox_id"], "secret.txt", owner_id="bob")
    assert stolen["ok"] is False
    assert "not owned" in stolen["error"]
    ran = sandbox_exec(
        a["sandbox_id"],
        [sys.executable, "-c", "print(1)"],
        owner_id="bob",
    )
    assert ran["ok"] is False
    assert "not owned" in ran["error"]
    own = sandbox_get_file(a["sandbox_id"], "secret.txt", owner_id="alice")
    assert own["content"] == "alice-only"
    sandbox_destroy(a["sandbox_id"], owner_id="alice")
    sandbox_destroy(b["sandbox_id"], owner_id="bob")


def test_opensandbox_unavailable_fails_honestly() -> None:
    from oprim._opensandbox import set_opensandbox_driver

    set_opensandbox_driver(None)
    rec = sandbox_create(isolation="opensandbox")
    assert rec["ok"] is False
    assert "opensandbox unavailable" in rec["error"]
