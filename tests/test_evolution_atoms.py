"""extract_ast_delta / write_fact_node / call_teacher_model."""

from __future__ import annotations

import pytest
from obase.graph_store.models import GraphDBPool

from oprim._call_teacher_model import call_teacher_model
from oprim._extract_ast_delta import extract_ast_delta
from oprim._write_fact_node import write_fact_node


@pytest.mark.asyncio
async def test_extract_worktree(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    rec = await extract_ast_delta(tmp_path, target_file="a.py", commit_hash="")
    assert rec["ok"] is True
    assert any(n["kind"] == "function" for n in rec["nodes"])


@pytest.mark.asyncio
async def test_write_fact_archives(tmp_path) -> None:
    pool = GraphDBPool()
    first = await write_fact_node(
        "mod.py",
        predicate="ast_delta",
        object_val="v0",
        evidence_chunk="e0",
        pool=pool,
    )
    second = await write_fact_node(
        "mod.py",
        predicate="ast_delta",
        object_val="v1",
        evidence_chunk="e1",
        pool=pool,
    )
    assert first != second
    assert pool.facts[first].status == "ARCHIVED"
    assert pool.find_active("mod.py", predicate="ast_delta").object_value == "v1"


@pytest.mark.asyncio
async def test_teacher_forwards() -> None:
    async def fake(*, messages, max_tokens):
        return {"content": "ok", "n": len(messages), "max_tokens": max_tokens}

    rec = await call_teacher_model([{"role": "user", "content": "hi"}], caller=fake)
    assert rec["ok"] is True
    assert rec["content"] == "ok"
    assert rec["max_tokens"] == 4096
