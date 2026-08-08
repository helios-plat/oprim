"""_sculpt_pipeline 状态机原语测试 (领域无关)。"""
import json

import pytest

from oprim import (
    SculptPipelineError,
    load_pipeline_state,
    new_pipeline_state,
    pipeline_mark,
    pipeline_status,
    record_refinement,
    save_pipeline_state,
    set_current_pass,
)

SETUP = [("image-analysis", "analyze {reference}"), ("spec-authoring", "write spec")]
PASS = [
    ("build-current-pass", "generate {pass_id}"),
    ("render-capture", "render {pass_id}"),
    ("gate-check", "gate {pass_id}"),
]
FINAL = [("part-coverage", "check coverage")]


def _state():
    return new_pipeline_state("ref.png", SETUP, PASS, FINAL, max_per_pass=2, max_total=3)


def test_initial_status_first_setup_step():
    st = pipeline_status(_state())
    assert st["currentStep"] == "image-analysis"
    assert "analyze ref.png" in st["nextCommand"]
    # 初始 pending: setup 两步 + pass 步骤 (currentPass 尚未 complete)
    assert "image-analysis" in st["pending"] and "spec-authoring" in st["pending"]
    assert "build-current-pass" in st["pending"]


def test_order_enforced_and_evidence_required():
    state = _state()
    with pytest.raises(SculptPipelineError):
        pipeline_mark(state, "spec-authoring", evidence=["x"])  # 越序
    with pytest.raises(SculptPipelineError):
        pipeline_mark(state, "image-analysis")  # done 无 evidence
    pipeline_mark(state, "image-analysis", evidence=["analysis.md"])
    assert pipeline_status(state)["currentStep"] == "spec-authoring"
    with pytest.raises(SculptPipelineError):
        pipeline_mark(state, "spec-authoring")  # 跳过无 reason
    pipeline_mark(state, "spec-authoring", status="skipped", reason="trivial subject")
    assert pipeline_status(state)["status"] == "active"


def test_pass_loop_and_hard_stop():
    state = _state()
    for step in ("image-analysis", "spec-authoring"):
        pipeline_mark(state, step, evidence=["e"])
    # setup 完成 → 进入 pass: currentPass 为空时标记 build-current-pass 会推进
    set_current_pass(state, "blockout")
    assert pipeline_status(state)["currentPass"] == "blockout"
    for step in ("build-current-pass", "render-capture", "gate-check"):
        pipeline_mark(state, step, evidence=["e"])
    # pass 步骤全 done → 下一个 pass 需 set_current_pass; 无 pending pass/final 前是 await
    st = pipeline_status(state)
    assert st["currentStep"] == "await-pass-transition"
    # 纠错循环: 2 次 refine 达到 maxPerPass=2 → hard stop
    record_refinement(state, "blockout", "refine-code")
    assert state["status"] == "active"
    record_refinement(state, "blockout", "refine-spec")
    assert state["status"] == "stopped"
    assert "max-correction-loops-reached:blockout:2/2" in state["stopReason"]
    with pytest.raises(SculptPipelineError):
        pipeline_mark(state, "image-analysis", evidence=["x"])  # 硬停止后禁止


def test_total_loop_stop():
    state = _state()
    for step in ("image-analysis", "spec-authoring"):
        pipeline_mark(state, step, evidence=["e"])
    set_current_pass(state, "p1")
    record_refinement(state, "p1", "refine-code")
    set_current_pass(state, "p2")
    record_refinement(state, "p2", "refine-code")
    set_current_pass(state, "p3")
    record_refinement(state, "p3", "refine-code")
    assert state["status"] == "stopped"
    assert "max-total-correction-loops-reached:3/3" in state["stopReason"]


def test_persist_roundtrip(tmp_path):
    state = _state()
    path = tmp_path / "state.json"
    save_pipeline_state(path, state)
    loaded = load_pipeline_state(path)
    assert loaded["checklist"][0]["id"] == "image-analysis"
    with pytest.raises(SculptPipelineError):
        load_pipeline_state(tmp_path / "missing.json")


def test_validate_rejects_bad_limits():
    with pytest.raises(SculptPipelineError):
        new_pipeline_state("r", SETUP, PASS, FINAL, max_per_pass=3, max_total=2)
