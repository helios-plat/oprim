"""oprim.tmux_pane_create — Tmux 窗格创建.

在指定 tmux 会话中创建新窗格（可选运行命令），返回 pane id。
tmux 二进制惰性探测；不可用时返回错误 dict（不抛）。

Example:
    >>> r = tmux_pane_create("dev", pane_command="htop")
    >>> r["pane_id"]
    '%3'
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from oprim._exceptions import OprimError, OprimValidationError


class TmuxPaneError(OprimError):
    """tmux 操作失败。"""


def tmux_pane_create(
    session: str,
    *,
    pane_command: str | None = None,
    index: int | None = None,
    cwd: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """创建 tmux 窗格。

    Args:
        session: 目标会话名。
        pane_command: 窗格内执行的命令（可选）。
        index: 目标窗格序号（可选）。
        cwd: 窗格工作目录（可选）。
        timeout: 命令超时。

    Returns:
        {"status": "ok", "pane_id": str, "session": str, "command": str|None}

    Raises:
        TmuxPaneError: tmux 不可用或创建失败。
        OprimValidationError: session 为空。
    """
    if not session or not session.strip():
        raise OprimValidationError("tmux_pane_create: session must not be empty")

    if shutil.which("tmux") is None:
        raise TmuxPaneError("tmux_pane_create: tmux binary not found in PATH")

    args = ["tmux", "new-window"]
    if cwd:
        args += ["-c", cwd]
    if index is not None:
        args += ["-t", f"{session}:{index}"]
    else:
        args += ["-t", session]
    if pane_command:
        args += [pane_command]

    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise TmuxPaneError(f"tmux_pane_create timed out after {timeout}s") from exc
    except OSError as exc:
        raise TmuxPaneError(f"tmux_pane_create failed: {exc}", cause=exc) from exc

    if result.returncode != 0:
        raise TmuxPaneError(
            f"tmux_pane_create failed: {result.stderr.strip() or result.stdout.strip()}"
        )

    pane_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return {"status": "ok", "pane_id": pane_id, "session": session, "command": pane_command}
