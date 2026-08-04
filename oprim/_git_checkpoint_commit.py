"""oprim.git_checkpoint_commit — 单次将当前工作区提交为一个带时间戳的 Git Checkpoint.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: subprocess (git CLI).

例:
    >>> result = git_checkpoint_commit("auto-snapshot", repo_path=".")
    >>> result["status"]
    'success'
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any


def git_checkpoint_commit(
    commit_message: str = "checkpoint",
    *,
    repo_path: str = ".",
    add_all: bool = True,
) -> dict[str, Any]:
    """单次将当前工作区提交为一个带时间戳的 Git Checkpoint.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        commit_message: 提交信息
        repo_path: 仓库路径
        add_all: 是否 git add . 所有变更

    Returns:
        {
            "status": "success" | "failed",
            "commit_hash": str,
            "message": str,
        }
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full_message = f"[CHECKPOINT {timestamp}] {commit_message}"

    try:
        if add_all:
            subprocess.run(
                ["git", "add", "."],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

        commit_res = subprocess.run(
            ["git", "commit", "-m", full_message],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        if commit_res.returncode != 0:
            # 可能没有变更需要提交
            if "nothing to commit" in commit_res.stdout + commit_res.stderr:
                return {
                    "status": "success",
                    "commit_hash": "",
                    "message": "No changes to commit — clean working tree.",
                }
            return {
                "status": "failed",
                "commit_hash": "",
                "message": f"Commit failed: {commit_res.stderr.strip()[:500]}",
            }

        hash_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "status": "success",
            "commit_hash": hash_res.stdout.strip(),
            "message": full_message,
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "failed",
            "commit_hash": "",
            "message": f"Git command failed: {e.stderr[:500] if e.stderr else str(e)}",
        }
    except FileNotFoundError:
        return {
            "status": "failed",
            "commit_hash": "",
            "message": "Git CLI not found — is git installed?",
        }
    except Exception as e:
        return {
            "status": "failed",
            "commit_hash": "",
            "message": str(e)[:500],
        }
