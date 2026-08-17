"""oprim.extract_ast_delta — one file at one commit → AST nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from obase.cocoindex.parser import ASTParser
from obase.git import run_git


async def extract_ast_delta(
    repo_path: Path | str,
    *,
    target_file: Path | str,
    commit_hash: str = "",
) -> dict[str, Any]:
    """Parse the file at ``commit_hash``, or the worktree if commit is empty."""
    root = Path(repo_path)
    rel = Path(target_file)
    if rel.is_absolute():
        try:
            rel = rel.relative_to(root.resolve())
        except ValueError:
            return {
                "ok": False,
                "nodes": [],
                "commit": commit_hash,
                "path": str(target_file),
                "error": "target_file is outside repo_path",
            }
    source = ""
    if commit_hash:
        shown = await run_git(["show", f"{commit_hash}:{rel.as_posix()}"], cwd=root)
        if not shown.ok:
            return {
                "ok": False,
                "nodes": [],
                "commit": commit_hash,
                "path": rel.as_posix(),
                "error": shown.stderr.strip() or "git show failed",
            }
        source = shown.stdout
    else:
        path = root / rel
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return {
                "ok": False,
                "nodes": [],
                "commit": "",
                "path": rel.as_posix(),
                "error": str(exc),
            }
    parsed = ASTParser().parse_source(source)
    parsed["commit"] = commit_hash
    parsed["path"] = rel.as_posix()
    return parsed
