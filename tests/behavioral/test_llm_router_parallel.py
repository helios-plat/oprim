"""Behavioral tests for the LLM routing / parallel-dispatch primitives."""

from __future__ import annotations

import json

import pytest

from oprim._llm_router import DEFAULT_MATRIX, load_matrix, route_decision
from oprim._parallel_llm import aggregate_results, dispatch_parallel, split_prompt


def _msg(text: str, role: str = "user") -> dict:
    return {"role": role, "content": text}


class TestRouteDecision:
    def test_quick_route_for_short_text(self) -> None:
        d = route_decision([_msg("hi")], matrix=DEFAULT_MATRIX)
        assert d["route"] == "quick"
        assert d["provider"] == DEFAULT_MATRIX["routes"]["quick"]["provider"]

    def test_system_prompt_excluded_from_length(self) -> None:
        msgs = [_msg("x" * 50_000, role="system"), _msg("hello")]
        assert route_decision(msgs, matrix=DEFAULT_MATRIX)["route"] == "quick"

    def test_long_route(self) -> None:
        assert route_decision([_msg("a " * 20_000)], matrix=DEFAULT_MATRIX)["route"] == "long"

    def test_text_route(self) -> None:
        assert route_decision([_msg("a " * 1_000)], matrix=DEFAULT_MATRIX)["route"] == "text"

    def test_vision_route(self) -> None:
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        assert route_decision(msgs, matrix=DEFAULT_MATRIX)["route"] == "vision"
        assert route_decision([_msg("see data:image/png;base64,AAA")])["route"] == "vision"

    def test_reason_code_tool_routes(self) -> None:
        assert route_decision([_msg("请证明这个定理")], matrix=DEFAULT_MATRIX)["route"] == "reason"
        assert route_decision([_msg("```py\nx=1\n```")], matrix=DEFAULT_MATRIX)["route"] == "code"
        assert route_decision([_msg("weather now")], tools=[{}], matrix=DEFAULT_MATRIX)["route"] == "tool"

    def test_frontier_by_hint_and_priority(self) -> None:
        assert route_decision([_msg("做一次安全审计")], matrix=DEFAULT_MATRIX)["route"] == "frontier"
        d = route_decision([_msg("请证明")], matrix=DEFAULT_MATRIX, priority="high")
        assert d["route"] == "frontier"
        assert d["model"] == DEFAULT_MATRIX["frontier"]["model"]

    def test_low_priority_and_budget_downgrade(self) -> None:
        assert route_decision([_msg("请证明")], matrix=DEFAULT_MATRIX, priority="low")["route"] == "text"
        assert route_decision([_msg("请证明")], matrix=DEFAULT_MATRIX, budget=0.0001)["route"] == "text"

    def test_load_matrix_from_file_and_fallback(self, tmp_path) -> None:
        p = tmp_path / "router.json"
        p.write_text(json.dumps({"routes": {"quick": {"provider": "p", "model": "m"}}}))
        m = load_matrix(str(p))
        assert m["routes"]["quick"] == {"provider": "p", "model": "m"}
        assert "text" in m["routes"]
        assert isinstance(load_matrix(str(tmp_path / "missing.json")), dict)


class TestParallel:
    def test_split_short_passthrough(self) -> None:
        assert split_prompt("short") == ["short"]

    def test_split_paragraphs(self) -> None:
        paras = ["p" * 300 for _ in range(3)]
        assert split_prompt("\n\n".join(paras), max_chunks=4) == paras
        many = "\n\n".join(f"para {i} " + "x" * 100 for i in range(10))
        chunks = split_prompt(many, max_chunks=4)
        assert 1 < len(chunks) <= 4

    def test_split_sentences(self) -> None:
        text = " ".join(f"This is sentence number {i}." for i in range(60))
        chunks = split_prompt(text, max_chunks=3)
        assert 1 < len(chunks) <= 3

    def test_aggregate_marks_failures(self) -> None:
        out = aggregate_results([{"ok": True, "output": "A"}, {"ok": False, "error": "t/o"}], title="T")
        assert out.startswith("# T") and "A" in out and "1 个分片未完成" in out

    @pytest.mark.asyncio
    async def test_dispatch_short_is_not_parallel(self) -> None:
        async def caller(chunk: str, idx: int) -> dict:
            return {"ok": True, "output": chunk.upper()}

        r = await dispatch_parallel("hi", caller)
        assert r["parallel"] is False and r["output"] == "HI"

    @pytest.mark.asyncio
    async def test_dispatch_parallel_isolates_failures(self) -> None:
        async def caller(chunk: str, idx: int):
            if idx == 1:
                raise RuntimeError("boom")
            if idx == 2:
                return "not-a-dict"
            return {"ok": True, "output": f"chunk{idx}"}

        prompt = "\n\n".join("p" * 300 for _ in range(3))
        r = await dispatch_parallel(prompt, caller, max_parallel=4)
        assert r["parallel"] is True and r["chunks"] == 3
        assert r["partial"] == [True, False, False]
        assert r["ok"] is True and "RuntimeError" in r["output"]
