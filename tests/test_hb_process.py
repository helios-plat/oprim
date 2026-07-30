"""Tests — H-B B组: 进程控制 (spawn_pty / stream_stdout / kill_process / wait_with_timeout / run_background)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from oprim._hb_process import (
    ProcHandle,
    PtyHandle,
    _JOBS,
    kill_process,
    run_background,
    spawn_pty,
    stream_stdout,
    wait_with_timeout,
)


# ---------------------------------------------------------------------------
# Helper: create a ProcHandle wrapping a real subprocess
# ---------------------------------------------------------------------------

async def _make_proc(cmd: str, *, cwd: Path, stdout: bool = True) -> ProcHandle:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE if stdout else asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        # Isolate in its own process group so os.killpg() in kill_process()
        # does not accidentally signal the pytest runner.
        start_new_session=True,
    )
    return ProcHandle(pid=proc.pid, _proc=proc)


# ---------------------------------------------------------------------------
# wait_with_timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_normal(tmp_path: Path) -> None:
    handle = await _make_proc("exit 0", cwd=tmp_path)
    code = await wait_with_timeout(handle, timeout=10)
    assert code == 0


@pytest.mark.asyncio
async def test_wait_nonzero_exit(tmp_path: Path) -> None:
    handle = await _make_proc("exit 42", cwd=tmp_path)
    code = await wait_with_timeout(handle, timeout=10)
    assert code == 42


@pytest.mark.asyncio
async def test_wait_already_finished(tmp_path: Path) -> None:
    handle = await _make_proc("exit 0", cwd=tmp_path)
    await handle._proc.wait()  # finish it
    code = await wait_with_timeout(handle, timeout=5)
    assert code == 0


@pytest.mark.asyncio
async def test_wait_timeout(tmp_path: Path) -> None:
    handle = await _make_proc("sleep 60", cwd=tmp_path)
    with pytest.raises(TimeoutError):
        await wait_with_timeout(handle, timeout=0.2)
    handle._proc.kill()
    # pytest-asyncio creates a per-test event loop; the ChildWatcher may be
    # attached to a different loop, so proc.wait() can deadlock.  SIGKILL is
    # enough — the OS reaps the child; we just yield briefly so any pending
    # loop callbacks can fire.
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_wait_zero_timeout(tmp_path: Path) -> None:
    handle = await _make_proc("exit 0", cwd=tmp_path)
    with pytest.raises(ValueError, match="timeout"):
        await wait_with_timeout(handle, timeout=0)
    handle._proc.kill()
    await handle._proc.wait()


@pytest.mark.asyncio
async def test_wait_negative_timeout(tmp_path: Path) -> None:
    handle = await _make_proc("exit 0", cwd=tmp_path)
    with pytest.raises(ValueError, match="timeout"):
        await wait_with_timeout(handle, timeout=-1)
    handle._proc.kill()
    await handle._proc.wait()


# ---------------------------------------------------------------------------
# kill_process
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_term(tmp_path: Path) -> None:
    handle = await _make_proc("sleep 60", cwd=tmp_path, stdout=False)
    await kill_process(handle, sig="TERM")
    # Use asyncio.wait() not wait_for() to avoid Python 3.12 cancel-hang bug.
    done, _ = await asyncio.wait({asyncio.ensure_future(handle._proc.wait())}, timeout=5)
    assert done, "process did not exit within 5s after SIGTERM"
    assert handle._proc.returncode is not None


@pytest.mark.asyncio
async def test_kill_already_exited_idempotent(tmp_path: Path) -> None:
    handle = await _make_proc("exit 0", cwd=tmp_path, stdout=False)
    await handle._proc.wait()
    # Should not raise
    await kill_process(handle, sig="KILL")


@pytest.mark.asyncio
async def test_kill_invalid_sig(tmp_path: Path) -> None:
    handle = await _make_proc("sleep 60", cwd=tmp_path, stdout=False)
    with pytest.raises(ValueError, match="signal"):
        await kill_process(handle, sig="BADNAME")
    handle._proc.kill()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_kill_kill_sig(tmp_path: Path) -> None:
    handle = await _make_proc("sleep 60", cwd=tmp_path, stdout=False)
    await kill_process(handle, sig="KILL")
    done, _ = await asyncio.wait({asyncio.ensure_future(handle._proc.wait())}, timeout=5)
    assert done, "process did not exit within 5s after SIGKILL"
    assert handle._proc.returncode is not None


# ---------------------------------------------------------------------------
# stream_stdout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_stdout_proc(tmp_path: Path) -> None:
    handle = await _make_proc('echo "hello from stream"', cwd=tmp_path, stdout=True)
    chunks: list[str] = []
    async for chunk in stream_stdout(handle):
        chunks.append(chunk)
    text = "".join(chunks)
    assert "hello from stream" in text


@pytest.mark.asyncio
async def test_stream_stdout_empty(tmp_path: Path) -> None:
    handle = await _make_proc("true", cwd=tmp_path, stdout=True)
    chunks: list[str] = []
    async for chunk in stream_stdout(handle):
        chunks.append(chunk)
    assert "".join(chunks) == ""


@pytest.mark.asyncio
async def test_stream_stdout_invalid_handle() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        async for _ in stream_stdout("bad"):  # type: ignore[arg-type]
            pass


# ---------------------------------------------------------------------------
# run_background
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_background_returns_job_id(tmp_path: Path) -> None:
    jid = await run_background("sleep 1", cwd=tmp_path)
    assert isinstance(jid, str) and len(jid) > 0
    assert jid in _JOBS
    _JOBS[jid].kill()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_run_background_does_not_block(tmp_path: Path) -> None:
    jid = await run_background("sleep 60", cwd=tmp_path)
    # If we got here quickly, it didn't block
    assert jid in _JOBS
    _JOBS[jid].kill()
    await asyncio.wait({asyncio.ensure_future(_JOBS[jid].wait())}, timeout=2)


@pytest.mark.asyncio
async def test_run_background_empty_cmd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        await run_background("", cwd=tmp_path)


@pytest.mark.asyncio
async def test_run_background_missing_cwd() -> None:
    with pytest.raises(FileNotFoundError):
        await run_background("echo hi", cwd=Path("/nonexistent_dir_xyz_12345"))


@pytest.mark.asyncio
async def test_run_background_multiple(tmp_path: Path) -> None:
    jids = [await run_background("sleep 60", cwd=tmp_path) for _ in range(3)]
    assert len(set(jids)) == 3  # unique IDs
    for jid in jids:
        _JOBS[jid].kill()
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# spawn_pty (only on Unix)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
@pytest.mark.asyncio
async def test_spawn_pty_basic(tmp_path: Path) -> None:
    handle = await spawn_pty("echo pty_test_ok", cwd=tmp_path)
    assert handle.pid > 0
    assert handle.master_fd > 0
    # Read output
    chunks: list[str] = []
    try:
        async with asyncio.timeout(5):
            async for chunk in stream_stdout(handle):
                chunks.append(chunk)
                if "pty_test_ok" in "".join(chunks):
                    break
    except (asyncio.TimeoutError, TimeoutError):
        pass
    import os
    try:
        os.close(handle.master_fd)
    except OSError:
        pass
    await handle._proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
@pytest.mark.asyncio
async def test_spawn_pty_empty_cmd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        await spawn_pty("", cwd=tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="PTY not available on Windows")
@pytest.mark.asyncio
async def test_spawn_pty_missing_cwd() -> None:
    with pytest.raises(FileNotFoundError):
        await spawn_pty("echo hi", cwd=Path("/nonexistent_xyz_99"))
