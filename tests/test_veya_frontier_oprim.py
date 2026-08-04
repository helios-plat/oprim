"""Tests for Veya 前沿探索元素 — oprim 层 (3 个).

Covers: fol_translate, causal_graph_build, invariant_extract.
"""

from __future__ import annotations

import pytest


class FakeFOLCaller:
    def __call__(self, prompt, *, temperature=0.0):
        return '{"declarations": [{"name": "p", "type": "Bool"}], "constraints": ["p"]}'


class BadFOLCaller:
    def __call__(self, prompt, *, temperature=0.0):
        return "not json at all"


class TestFolTranslate:
    def test_success(self):
        from oprim import fol_translate

        r = fol_translate("p 为真", llm_caller=FakeFOLCaller())
        assert r["status"] == "success"
        assert r["declarations"] == [{"name": "p", "type": "Bool"}]
        assert r["constraints"] == ["p"]

    def test_llm_invalid_json(self):
        from oprim import fol_translate

        r = fol_translate("x", llm_caller=BadFOLCaller())
        assert r["status"] == "error"
        assert "invalid JSON" in r["error"]

    def test_validation(self):
        from oprim import OprimValidationError, fol_translate

        with pytest.raises(OprimValidationError):
            fol_translate("", llm_caller=FakeFOLCaller())
        with pytest.raises(OprimValidationError):
            fol_translate("x", llm_caller=None)


class TestCausalGraphBuild:
    def test_chain_and_output_refs(self):
        from oprim import causal_graph_build

        trail = [
            {"type": "action", "action": "read", "status": "success"},
            {
                "type": "action", "action": "edit", "status": "failed",
                "output_refs": ["step_0_action"],
            },
        ]
        r = causal_graph_build(trail, strict_mode=True)
        assert r["node_count"] == 2 and r["edge_count"] == 2
        assert r["graph"]["nodes"][0]["id"] == "step_0_action"
        assert r["graph"]["edges"][0]["relation"] == "causes"
        assert r["graph"]["edges"][1]["relation"] == "depends_on"

    def test_empty_trail_rejected(self):
        from oprim import OprimValidationError, causal_graph_build

        with pytest.raises(OprimValidationError):
            causal_graph_build([])

    def test_non_list_rejected(self):
        from oprim import OprimValidationError, causal_graph_build

        with pytest.raises(OprimValidationError):
            causal_graph_build("not-a-list")


class TestInvariantExtract:
    def test_assert_and_guard(self):
        from oprim import invariant_extract

        code = (
            "def f(x):\n"
            "    assert x > 0\n"
            "    if x < 0:\n"
            "        raise ValueError()\n"
            "    return x\n"
        )
        r = invariant_extract(code)
        assert r["status"] == "success"
        assert "x > 0" in r["invariants"]
        assert "Not(x < 0)" in r["invariants"]

    def test_target_function_filter(self):
        from oprim import invariant_extract

        code = (
            "def good():\n"
            "    assert a == 1\n"
            "def bad():\n"
            "    assert b == 2\n"
        )
        r = invariant_extract(code, target_function="good")
        assert r["invariants"] == ["a == 1"]

    def test_syntax_error_failed(self):
        from oprim import invariant_extract

        r = invariant_extract("def broken(:\n")
        assert r["status"] == "failed" and r["error"]

    def test_empty_rejected(self):
        from oprim import OprimValidationError, invariant_extract

        with pytest.raises(OprimValidationError):
            invariant_extract("")
