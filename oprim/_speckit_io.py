"""oprim.speckit_io — read Spec Kit files, atomically write taskgraph.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from obase.veya_workspace import SpecKitPaths

_KNOWN = ("constitution.md", "tasks.md", "spec.md", "plan.md")


async def load_speckit_artifacts(
    paths: SpecKitPaths,
    *,
    artifact_types: list[str] | None = None,
) -> dict[str, str]:
    wanted = artifact_types or ["constitution.md", "tasks.md"]
    out: dict[str, str] = {}
    for name in wanted:
        if name not in _KNOWN and not name.endswith(".md"):
            continue
        path = paths.artifact(name)
        if path.is_file():
            out[name] = path.read_text(encoding="utf-8")
    return out


async def save_taskgraph(
    paths: SpecKitPaths,
    *,
    goal_id: str,
    graph_dict: dict[str, Any],
) -> Path:
    dest = paths.taskgraph_path(goal_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(graph_dict, ensure_ascii=False, indent=2)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, dest)
    return dest
