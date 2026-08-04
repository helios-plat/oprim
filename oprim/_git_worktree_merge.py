"""oprim.git_worktree_merge — Worktree 分支合并回目标分支.

把 worktree 分支的提交合并进目标分支（fast-forward 优先，失败回退 merge
提交），返回合并结果与冲突信息。基于 git subprocess（obase.git 惰性）。

Example:
    >>> r = await git_worktree_merge("feat/x", repo="/repo")
    >>> r["merged"]
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._exceptions import GitOprimError, OprimValidationError


async def git_worktree_merge(
    branch: str,
    *,
    repo: str | Path,
    target: str = "main",
    commit_message: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """把 branch 合并进 target。

    Args:
        branch: 待合并的 worktree 分支。
        repo: 仓库根目录。
        target: 目标分支（默认 main）。
        commit_message: merge 提交信息；None 用默认。
        timeout: git 命令超时。

    Returns:
        {"status": "merged"|"conflict"|"error", "merged": bool, "branch": str,
         "target": str, "conflicts": [str], "detail": str}

    Raises:
        GitOprimError: git 命令失败（非冲突类）。
        OprimValidationError: branch 为空。
    """
    if not branch or not branch.strip():
        raise OprimValidationError("git_worktree_merge: branch must not be empty")

    try:
        from obase.git import run_git
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise GitOprimError("git_worktree_merge: obase unavailable", cause=exc) from exc

    repo_path = Path(repo).resolve()
    if not repo_path.is_dir():
        raise GitOprimError(f"git_worktree_merge: repo not found: {repo_path}")

    # 1) 切到 target 并更新
    checkout = await run_git(["checkout", target], cwd=repo_path, timeout=timeout)
    if not checkout.ok:
        return {"status": "error", "merged": False, "branch": branch, "target": target,
                "conflicts": [], "detail": checkout.stderr.strip()}

    # 2) fast-forward 尝试
    ff = await run_git(["merge", "--ff-only", branch], cwd=repo_path, timeout=timeout)
    if ff.ok:
        return {"status": "merged", "merged": True, "branch": branch, "target": target,
                "conflicts": [], "detail": "fast-forward"}

    # 3) 常规 merge（--no-edit 用默认信息）
    args = ["merge", "--no-edit"]
    if commit_message:
        args += ["-m", commit_message]
    args.append(branch)
    merge = await run_git(args, cwd=repo_path, timeout=timeout)
    if merge.ok:
        return {"status": "merged", "merged": True, "branch": branch, "target": target,
                "conflicts": [], "detail": "merge commit"}

    # 4) 冲突：列出冲突文件并中止
    combined = merge.stdout + "\n" + merge.stderr
    conflicts = [ln for ln in combined.splitlines() if "CONFLICT" in ln]
    await run_git(["merge", "--abort"], cwd=repo_path, timeout=timeout)
    return {"status": "conflict", "merged": False, "branch": branch, "target": target,
            "conflicts": conflicts, "detail": combined.strip()}
