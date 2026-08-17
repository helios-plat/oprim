"""boss LLM callers parse injected responses."""

from __future__ import annotations

import pytest

from oprim.boss_llm_callers import (
    call_llm_for_intent,
    call_llm_for_planning,
    call_llm_for_verification,
)


async def _plan_caller(*, messages, max_tokens):
    return {"ok": True, "content": '{"tasks": [{"id": "T1", "title": "A"}]}'}


async def _verify_caller(*, messages, max_tokens):
    return {"ok": True, "content": '{"passed": false, "reasoning": "no diff"}'}


async def _intent_caller(*, messages, max_tokens):
    return {
        "ok": True,
        "content": '{"action": "plan", "interpretation": "add foo to a.py"}',
    }


@pytest.mark.asyncio
async def test_intent_parses_action() -> None:
    rec = await call_llm_for_intent(
        [{"role": "user", "content": "x"}], caller=_intent_caller
    )
    assert rec["action"] == "plan"
    assert "foo" in rec["interpretation"]


@pytest.mark.asyncio
async def test_planning_parses_tasks() -> None:
    rec = await call_llm_for_planning([{"role": "user", "content": "x"}], caller=_plan_caller)
    assert rec["ok"] is True
    assert rec["tasks"][0]["id"] == "T1"


@pytest.mark.asyncio
async def test_verification_parses_passed() -> None:
    rec = await call_llm_for_verification(
        [{"role": "user", "content": "x"}], caller=_verify_caller
    )
    assert rec["passed"] is False
    assert "no diff" in rec["reasoning"]
