"""oprim.execute_leaf_with_constitution — jail + constitution prefix. No server import."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from obase.sandbox.path_jail import PathJail

LeafRunner = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


def compose_constitution_brief(*, instruction: str, constitution_text: str) -> str:
    constitution = constitution_text.strip()
    if not constitution:
        return instruction
    return f"## Project Constitution\n{constitution}\n\n## Task\n{instruction}"


async def execute_leaf_with_constitution(
    project_root: Path | str,
    *,
    instruction: str,
    constitution_text: str,
    assignee: str = "hicode",
    runner: LeafRunner | None = None,
) -> dict[str, Any]:
    """Pin constitution at the top of the brief. PathJail verifies the project root."""
    root = Path(project_root)
    jail = PathJail(root)
    jail.resolve_and_verify(".")
    brief = compose_constitution_brief(
        instruction=instruction, constitution_text=constitution_text
    )
    if runner is None:
        return {
            "ok": True,
            "status": "prepared",
            "brief": brief,
            "assignee": assignee,
            "jailed_root": str(jail.root),
        }
    result = runner(
        project_root=str(jail.root),
        instruction=brief,
        assignee=assignee,
    )
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[misc]
    if not isinstance(result, dict):
        return {"ok": False, "status": "blocked", "error": "runner returned non-dict"}
    out = dict(result)
    out.setdefault("ok", out.get("status") in {"completed", "prepared"})
    out["brief"] = brief
    return out
