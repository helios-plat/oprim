"""Git atomic operations backed exclusively by :func:`obase.run_git`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obase.git import run_git

from ._exceptions import GitOprimError


def _call(args: list[str], *, repo: str | Path, input_text: str | None = None) -> Any:
    """Synchronously expose the async obase Git primitive to legacy callers."""
    try:
        result = asyncio.run(run_git(args, cwd=Path(repo), input_text=input_text))
    except (FileNotFoundError, TimeoutError) as exc:
        raise GitOprimError(str(exc)) from exc
    if not result.ok:
        command = "git " + " ".join(args[:1])
        raise GitOprimError(f"{command} failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


@dataclass
class FileStatus:
    path: str
    index: str
    worktree: str
    renamed_from: str | None = None


@dataclass
class Commit:
    hash: str
    author: str
    date: str
    message: str


@dataclass
class BlameLine:
    lineno: int
    commit: str
    author: str
    content: str


def git_status(*, repo: str | Path) -> list[FileStatus]:
    out = _call(["status", "--porcelain=v1", "-u"], repo=repo)
    statuses: list[FileStatus] = []
    for line in out.splitlines():
        if not line:
            continue
        rest = line[3:]
        renamed_from = None
        if " -> " in rest:
            renamed_from, rest = rest.split(" -> ", 1)
        statuses.append(FileStatus(rest.strip(), line[0], line[1], renamed_from))
    return statuses


def git_diff(
    *,
    repo: str | Path,
    staged: bool = False,
    base: str | None = None,
    head: str | None = None,
    paths: list[str] | None = None,
    context_lines: int = 3,
) -> str:
    args = ["diff", f"-U{context_lines}"]
    if staged:
        args.append("--cached")
    if base is not None:
        args.append(base if head is None else f"{base}..{head}")
    if paths:
        args.extend(["--", *paths])
    return _call(args, repo=repo)


def git_show(ref: str, *, repo: str | Path, path: str | None = None) -> str:
    return _call(["show", f"{ref}:{path}" if path else ref], repo=repo)


def git_apply(patch: str, *, repo: str | Path, check: bool = False) -> str:
    args = ["apply"]
    if check:
        args.append("--check")
    return _call(args, repo=repo, input_text=patch)


def git_add(paths: list[str] | str, *, repo: str | Path) -> None:
    selected = [paths] if isinstance(paths, str) else paths
    _call(["add", "--", *selected], repo=repo)


def git_commit(*, repo: str | Path, message: str, allow_empty: bool = False) -> str:
    args = ["commit", "-m", message]
    if allow_empty:
        args.append("--allow-empty")
    _call(args, repo=repo)
    return _call(["rev-parse", "--short=8", "HEAD"], repo=repo).strip()


def git_branch(
    *, repo: str | Path, name: str | None = None, create: bool = False, delete: bool = False
) -> list[str] | str:
    if create and name:
        return _call(["checkout", "-b", name], repo=repo).strip()
    if delete and name:
        return _call(["branch", "-d", name], repo=repo).strip()
    return [
        line.strip().lstrip("* ")
        for line in _call(["branch", "--list"], repo=repo).splitlines()
        if line.strip()
    ]


def git_checkout(ref: str, *, repo: str | Path) -> None:
    _call(["checkout", *ref.split()], repo=repo)


def git_stash(*, repo: str | Path, pop: bool = False, message: str = "") -> str:
    args = ["stash", "pop"] if pop else ["stash", "push"]
    if message and not pop:
        args.extend(["-m", message])
    return _call(args, repo=repo).strip()


def git_worktree_create(
    branch: str,
    *,
    repo: str | Path,
    path: str | Path,
    create_branch: bool = True,
) -> Path:
    target = Path(path).resolve()
    args = ["worktree", "add"]
    if create_branch:
        args.extend(["-b", branch, str(target)])
    else:
        args.extend([str(target), branch])
    _call(args, repo=repo)
    return target


def git_worktree_add(
    branch: str,
    *,
    repo: str | Path,
    path: str | Path | None = None,
    create_branch: bool = True,
) -> Path:
    repo_path = Path(repo).resolve()
    if path is None:
        safe_branch = branch.replace("/", "-").replace(" ", "_")
        path = repo_path.parent / f"{repo_path.name}-worktrees" / safe_branch
    return git_worktree_create(branch, repo=repo_path, path=path, create_branch=create_branch)


def git_worktree_remove(path: str | Path, *, repo: str | Path, force: bool = False) -> None:
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(Path(path).resolve()))
    _call(args, repo=repo)


def git_worktree_list(*, repo: str | Path) -> list[dict[str, str]]:
    out = _call(["worktree", "list", "--porcelain"], repo=repo)
    worktrees: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line[9:].strip(), "branch": "", "commit": ""}
        elif line.startswith("HEAD "):
            current["commit"] = line[5:].strip()[:8]
        elif line.startswith("branch "):
            current["branch"] = line[7:].strip().replace("refs/heads/", "")
    if current:
        worktrees.append(current)
    return worktrees


def git_log(*, repo: str | Path, n: int = 20, path: str | None = None) -> list[Commit]:
    args = ["log", f"-{n}", "--format=%H%x1f%an%x1f%aI%x1f%s%x1e"]
    if path:
        args.extend(["--", path])
    out = _call(args, repo=repo)
    records: list[Commit] = []
    for item in out.split("\x1e"):
        fields = item.strip("\n").split("\x1f")
        if len(fields) == 4 and fields[0]:
            records.append(Commit(fields[0], fields[1], fields[2], fields[3]))
    return records


def git_blame(path: str, *, repo: str | Path) -> list[BlameLine]:
    out = _call(["blame", "--porcelain", "--", path], repo=repo)
    lines: list[BlameLine] = []
    lineno = 0
    commit = author = ""
    for raw in out.splitlines():
        if raw.startswith("\t"):
            lines.append(BlameLine(lineno, commit[:8], author, raw[1:]))
        elif len(raw) >= 41 and raw[40] == " " and all(c in "0123456789abcdef" for c in raw[:40]):
            fields = raw.split()
            commit = fields[0]
            lineno = int(fields[2])
        elif raw.startswith("author "):
            author = raw[7:]
    return lines


__all__ = [
    "BlameLine", "Commit", "FileStatus", "git_add", "git_apply", "git_blame",
    "git_branch", "git_checkout", "git_commit", "git_diff", "git_log", "git_show",
    "git_stash", "git_status", "git_worktree_add", "git_worktree_create",
    "git_worktree_list", "git_worktree_remove",
]
