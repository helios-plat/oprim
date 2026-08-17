"""load_speckit_artifacts / save_taskgraph / jailed executor."""

from __future__ import annotations

import pytest

from obase.veya_workspace import SpecKitPaths
from oprim._jailed_executor import compose_constitution_brief, execute_leaf_with_constitution
from oprim._speckit_io import load_speckit_artifacts, save_taskgraph


@pytest.mark.asyncio
async def test_load_and_save(tmp_path) -> None:
    paths = SpecKitPaths(tmp_path)
    paths.speckit_dir.mkdir(parents=True)
    (paths.speckit_dir / "constitution.md").write_text("use fetch", encoding="utf-8")
    (paths.speckit_dir / "tasks.md").write_text("- [ ] T1 a", encoding="utf-8")
    arts = await load_speckit_artifacts(paths, artifact_types=["constitution.md", "tasks.md"])
    assert "constitution.md" in arts
    dest = await save_taskgraph(paths, goal_id="g1", graph_dict={"goal_id": "g1", "tasks": []})
    assert dest.is_file()
    assert "g1" in dest.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_leaf_prepares_brief(tmp_path) -> None:
    rec = await execute_leaf_with_constitution(
        tmp_path, instruction="add form", constitution_text="Do not use axios"
    )
    assert rec["status"] == "prepared"
    assert "Project Constitution" in rec["brief"]
    assert "add form" in rec["brief"]


def test_compose_without_constitution() -> None:
    assert compose_constitution_brief(instruction="x", constitution_text="  ") == "x"
