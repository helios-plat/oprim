"""oprim.diff_since — changed files + unified diff since a git ref.

Single subprocess family (git -C). Does not call other oprims.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_MAX_DIFF_CHARS = 200_000


def _git(repo: Path, *args: str, timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, "", "git executable not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"git {' '.join(args)} timed out"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def diff_since(
    *,
    repo: str | Path,
    since_ref: str = "HEAD",
    include_untracked: bool = True,
) -> dict[str, Any]:
    """List files changed since ``since_ref`` (working tree included).

    Args:
        repo: Git working tree root.
        since_ref: Compare against this ref (default HEAD = uncommitted + vs HEAD).
        include_untracked: Also list untracked files (they never appear in ``git diff``).

    Returns:
        ``{ok, since_ref, files:[{path,status}], changed:[str], diff, error?}``
    """
    root = Path(repo).resolve()
    if not root.is_dir():
        return {
            "ok": False,
            "since_ref": since_ref,
            "files": [],
            "changed": [],
            "diff": "",
            "error": f"not a directory: {repo}",
        }

    code, _, err = _git(root, "rev-parse", "--is-inside-work-tree")
    if code != 0:
        return {
            "ok": False,
            "since_ref": since_ref,
            "files": [],
            "changed": [],
            "diff": "",
            "error": err.strip() or "not a git repository",
        }

    files: list[dict[str, str]] = []
    seen: set[str] = set()

    code, out, err = _git(root, "diff", "--name-status", since_ref)
    if code != 0:
        return {
            "ok": False,
            "since_ref": since_ref,
            "files": [],
            "changed": [],
            "diff": "",
            "error": err.strip() or f"git diff --name-status {since_ref} failed",
        }
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0].strip()[:1] or "M"
        path = parts[-1].strip()
        if not path or path in seen:
            continue
        seen.add(path)
        files.append({"path": path, "status": status})

    if include_untracked:
        code, out, _ = _git(root, "ls-files", "--others", "--exclude-standard")
        if code == 0:
            for path in out.splitlines():
                path = path.strip()
                if not path or path in seen:
                    continue
                seen.add(path)
                files.append({"path": path, "status": "?"})

    code, diff, err = _git(root, "diff", since_ref)
    if code != 0:
        diff = ""
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n...[diff truncated]...\n"

    return {
        "ok": True,
        "since_ref": since_ref,
        "files": files,
        "changed": [item["path"] for item in files],
        "diff": diff,
    }
