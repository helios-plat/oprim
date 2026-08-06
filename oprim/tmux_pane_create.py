"""oprim.tmux_pane_create — ClawTeam-style tmux tiled terminal monitoring.

Creates and manages tmux sessions/windows/panes for real-time agent monitoring.
Each agent gets its own pane; the leader has a multi-pane layout for overview.

3O element: ``oprim.tmux_pane_create``.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tmux_pane_create(
    action: str = "create",
    session_name: str = "veya-swarm",
    agent_name: str | None = None,
    command: str | None = None,
    layout: str = "tiled",
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Manage tmux sessions for agent swarm monitoring.

    Actions:
      * ``create`` — create a new session or window
      * ``pane`` — add a pane for an agent (runs a watch command)
      * ``list`` — list windows/panes
      * ``kill`` — terminate session
      * ``capture`` — capture pane content for web streaming

    Args:
        action: ``create`` | ``pane`` | ``list`` | ``kill`` | ``capture``
        session_name: tmux session name
        agent_name: Agent name (used as pane title)
        command: Shell command to run in the pane
        layout: tmux layout (``tiled``, ``even-vertical``, ``main-vertical``)
        context: Optional config.

    Returns:
        {status, session, panes: [...], output}
    """
    ctx = context or {}

    def _tmux(*args: str) -> str:
        try:
            return subprocess.check_output(["tmux"] + list(args), text=True, stderr=subprocess.DEVNULL, timeout=int(ctx.get("timeout", 10))).strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    if action == "create":
        _tmux("new-session", "-d", "-s", session_name)
        _tmux("rename-window", "-t", f"{session_name}:0", "overview")
        _tmux("select-layout", layout)
        return {"status": "created", "session": session_name, "panes": [_list_panes(session_name)]}

    elif action == "pane":
        if not agent_name:
            return {"status": "failed", "error": "agent_name required for pane action"}
        cmd = command or f'while true; do echo "[{agent_name}] $(date +%H:%M:%S)"; sleep 5; done'
        _tmux("split-window", "-t", session_name, "-d")
        _tmux("select-layout", layout)
        _tmux("send-keys", "-t", f"{session_name}:0", f'echo "--- {agent_name} ---" ; {cmd}', "Enter")
        # rename the pane for identification
        _tmux("send-keys", "-t", f"{session_name}:0", "C-b", ",")
        _tmux("send-keys", "-t", f"{session_name}:0", agent_name, "Enter")
        return {"status": "pane_added", "session": session_name, "agent": agent_name}

    elif action == "list":
        return {"status": "ok", "session": session_name, "panes": _list_panes(session_name)}

    elif action == "capture":
        # capture visible content for web streaming
        output = _tmux("capture-pane", "-t", session_name, "-p", "-S", "-200")
        return {"status": "captured", "session": session_name, "output": output, "lines": len(output.splitlines())}

    elif action == "kill":
        _tmux("kill-session", "-t", session_name)
        return {"status": "killed", "session": session_name}

    return {"status": "failed", "error": f"unknown action: {action}"}


def _list_panes(session_name: str) -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(["tmux", "list-panes", "-t", session_name, "-F", "#{pane_index} #{pane_title}"], text=True, stderr=subprocess.DEVNULL, timeout=5)
        panes = []
        for line in out.strip().splitlines():
            parts = line.split(" ", 1)
            panes.append({"index": parts[0], "title": parts[1] if len(parts) > 1 else ""})
        return panes
    except Exception:
        return []
