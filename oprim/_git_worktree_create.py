"""oprim.git_worktree_create — 单次 Git Worktree 目录创建与切分支.

组合 obase.git_worktree.add_worktree（git worktree add 单次 subprocess），
为多 Agent 并行开发提供隔离文件树。

Example:
    >>> wt = await git_worktree_create("feat/parallel", repo="/repo")
    >>> wt
    PosixPath('/repo-worktrees/feat-parallel')
"""

from __future__ import annotations

from pathlib import Path

from oprim._exceptions import GitOprimError


async def git_worktree_create(
    branch: str,
    *,
    repo: str | Path,
    path: str | Path | None = None,
    create_branch: bool = True,
    timeout: float = 30.0,
) -> Path:
    """创建 worktree 并切到分支，返回 worktree 目录绝对路径。

    Args:
        branch: worktree 分支名；create_branch=True 时不存在则创建。
        repo: Git 仓库根目录。
        path: worktree 落盘位置；None 自动取 "<repo-parent>/<repo>-worktrees/<branch-safe>"。
        create_branch: 是否允许新建分支。
        timeout: git subprocess 超时秒数。

    Returns:
        worktree 目录绝对 Path。

    Raises:
        GitOprimError: git 命令失败 / 仓库目录不存在。
    """
    if not branch.strip():
        raise GitOprimError("git_worktree_create: branch must not be empty")
    try:
        from obase.git_worktree import GitWorktreeError, add_worktree
    except ImportError as exc:  # pragma: no cover
        raise GitOprimError("git_worktree_create: obase unavailable", cause=exc) from exc

    try:
        return await add_worktree(
            repo=Path(repo),
            branch=branch,
            path=Path(path) if path else None,
            create_branch=create_branch,
            timeout=timeout,
        )
    except (GitWorktreeError, FileNotFoundError) as exc:
        raise GitOprimError(f"git_worktree_create failed: {exc}", cause=exc) from exc
