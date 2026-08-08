"""分阶段工作流状态机 (Sculpt Pipeline State Machine) — 通用原语。

从 img2threejs forge/state.py + workflow_state.py 提炼的领域无关骨架:
任何"分阶段生成 + 逐步证据验证 + 硬停止纠错循环"的任务 (3D 雕刻 / UI 复刻 /
视频分镜 / 长文档重构) 的可恢复执行索引。

核心语义 (与 img2threejs 对齐):
- 三段步骤: setup (一次性准备) → pass (每轮迭代, 循环重置) → final (收尾);
- 每个步骤强制 evidence (完成必须有证据) / reason (跳过必须有理由);
- 严格顺序: 只能标记 next_entry 期望的步骤;
- 纠错循环硬停止: 单 pass 修正次数 > max_per_pass 或总修正次数 > max_total
  即 hard-stop (status=stopped), 防无限自纠错烧 token;
- 状态可 JSON 序列化 → 跨会话/跨 agent 恢复 (resumable index)。

状态文件是"可恢复索引"而非证据本体: 渲染图/spec/审查历史仍是权威工件。

纯 stdlib, 无第三方依赖。
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = 1
STEP_STATUSES: Final = {"pending", "done", "skipped"}
REFINE_ACTIONS: Final = {"refine-spec", "refine-code"}
SCOPES: Final = {"setup", "pass", "final"}


class SculptPipelineError(ValueError):
    """状态机非法操作 (越序标记 / 硬停止后继续 / 未知步骤等)。"""


def _step(step_id: str, command: str, *, scope: str) -> dict[str, Any]:
    return {
        "id": step_id,
        "scope": scope,
        "status": "pending",
        "evidence": [],
        "reason": "",
        "command": command,
    }


def new_pipeline_state(
    reference: str,
    setup_steps: list[tuple[str, str]],
    pass_steps: list[tuple[str, str]],
    final_steps: list[tuple[str, str]] | None = None,
    *,
    profile: str = "generic",
    spec: str = "",
    max_per_pass: int = 3,
    max_total: int = 6,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建新状态机。

    Args:
        reference: 参考工件路径 (图片/文档/规格)。
        setup_steps / pass_steps / final_steps: (step_id, command) 元组列表。
            command 可含 {reference}/{spec}/{pass_id} 占位符 (自动填充)。
        max_per_pass: 单 pass 内最大纠错次数 (硬停止阈值)。
        max_total: 全流程最大纠错总次数。
        meta: 调用方附加元数据, 原样保留。
    """
    if max_per_pass < 1 or max_total < 1 or max_per_pass > max_total:
        raise SculptPipelineError("loop limits require 1 <= max-per-pass <= max-total")
    checklist = [_step(*item, scope="setup") for item in setup_steps]
    if pass_steps:
        checklist += [_step(*item, scope="pass") for item in pass_steps]
    if final_steps:
        checklist += [_step(*item, scope="final") for item in final_steps]
    if not checklist:
        raise SculptPipelineError("at least one step is required")
    state = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "active",
        "profile": profile,
        "currentStep": checklist[0]["id"],
        "currentPass": "",
        "checklist": checklist,
        "loops": {"perPass": {}, "total": 0, "maxPerPass": max_per_pass, "maxTotal": max_total},
        "artifacts": {"reference": reference, "spec": spec},
        "passHistory": [],
        "reviewCursor": 0,
        "iterationAction": "initial",
        "stopReason": "",
        "meta": meta or {},
    }
    _recompute(state)
    return state


def validate_pipeline_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise SculptPipelineError("state must be a JSON object")
    if state.get("schemaVersion") != SCHEMA_VERSION:
        raise SculptPipelineError(
            f"unsupported state schemaVersion: {state.get('schemaVersion')!r}"
        )
    checklist = state.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        raise SculptPipelineError("state checklist must be a non-empty list")
    seen: set[str] = set()
    for entry in checklist:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise SculptPipelineError("every checklist entry needs a string id")
        if entry["id"] in seen:
            raise SculptPipelineError(f"duplicate checklist step: {entry['id']}")
        seen.add(entry["id"])
        if entry.get("scope") not in SCOPES:
            raise SculptPipelineError(f"invalid checklist scope for {entry['id']}")
        if entry.get("status") not in STEP_STATUSES:
            raise SculptPipelineError(f"invalid checklist status for {entry['id']}")
    loops = state.get("loops")
    if not isinstance(loops, dict):
        raise SculptPipelineError("state loops must be an object")
    max_per_pass = loops.get("maxPerPass")
    max_total = loops.get("maxTotal")
    if not isinstance(max_per_pass, int) or not isinstance(max_total, int):
        raise SculptPipelineError("loop limits must be integers")
    if max_per_pass < 1 or max_total < 1 or max_per_pass > max_total:
        raise SculptPipelineError("loop limits require 1 <= maxPerPass <= maxTotal")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("reference"):
        raise SculptPipelineError("state artifacts.reference is required")
    return state


def load_pipeline_state(path: str | Path) -> dict[str, Any]:
    try:
        state = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SculptPipelineError(f"state file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SculptPipelineError(f"state file is not valid JSON: {path}") from exc
    return validate_pipeline_state(state)


def save_pipeline_state(path: str | Path, state: dict[str, Any]) -> None:
    """原子写 (tmp + os.replace), 避免半写损坏跨会话索引。"""
    validate_pipeline_state(state)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _entries(state: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    return [entry for entry in state["checklist"] if entry["scope"] == scope]


def _pending(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if entry["status"] == "pending"]


def _format_command(state: dict[str, Any], entry: dict[str, Any]) -> str:
    artifacts = state.get("artifacts", {})
    command = str(entry["command"])
    if entry["id"] == "build-current-pass":
        action = state.get("iterationAction")
        if action == "refine-code":
            return "Refine the existing output from the latest review; do not regenerate it"
        if action in {"new-pass", "refine-spec"}:
            command += " --force"
    return command.format(
        reference=shlex.quote(str(artifacts.get("reference") or "<reference>")),
        spec=shlex.quote(str(artifacts.get("spec") or "<spec>")),
        pass_id=shlex.quote(str(state.get("currentPass") or "<pass>")),
    )


def next_entry(state: dict[str, Any]) -> dict[str, Any] | None:
    setup_pending = _pending(_entries(state, "setup"))
    if setup_pending:
        return setup_pending[0]
    if state.get("currentPass") != "complete":
        pass_pending = _pending(_entries(state, "pass"))
        if pass_pending:
            return pass_pending[0]
        return {
            "id": "await-pass-transition",
            "scope": "pass",
            "status": "pending",
            "command": "advance to the next pass (record pass acceptance, then set_current_pass)",
        }
    final_pending = _pending(_entries(state, "final"))
    return final_pending[0] if final_pending else None


def _recompute(state: dict[str, Any]) -> None:
    entry = next_entry(state)
    if state.get("status") == "stopped":
        state["currentStep"] = "stopped"
    elif entry is None:
        state["status"] = "complete"
        state["currentStep"] = "complete"
        state["stopReason"] = ""
    else:
        state["status"] = "active"
        state["currentStep"] = entry["id"]
        state["stopReason"] = ""


def pipeline_mark(
    state: dict[str, Any],
    step_id: str,
    *,
    status: str = "done",
    evidence: list[str] | None = None,
    reason: str = "",
) -> None:
    """标记步骤完成/跳过/重置。严格顺序: 只能标记 next_entry 期望的步骤。

    - done 必须有 evidence; skipped 必须有 reason;
    - 硬停止后不允许再标记 (raise)。
    """
    if state.get("status") == "stopped":
        raise SculptPipelineError("state is hard-stopped; do not mark more work complete")
    if status not in {"done", "skipped", "pending"}:
        raise SculptPipelineError("mark status must be done, skipped, or pending")
    if status == "done" and not evidence:
        raise SculptPipelineError(
            "completing a mandatory step requires at least one evidence value"
        )
    if status == "skipped" and not reason.strip():
        raise SculptPipelineError("skipping a mandatory step requires a reason")
    by_id = {entry["id"]: entry for entry in state["checklist"]}
    if step_id not in by_id:
        raise SculptPipelineError(f"unknown checklist step: {step_id}")
    entry = by_id[step_id]
    if status in {"done", "skipped"}:
        expected = next_entry(state)
        if expected is None or expected["id"] != step_id:
            expected_id = expected["id"] if expected else "complete"
            raise SculptPipelineError(
                f"out-of-order checklist update: expected {expected_id}, received {step_id}"
            )
        entry["status"] = status
        entry["evidence"] = list(evidence or [])
        entry["reason"] = reason.strip()
        _recompute(state)
        return
    entry["status"] = status
    entry["evidence"] = list(evidence or [])
    entry["reason"] = reason.strip()
    _recompute(state)


def set_current_pass(state: dict[str, Any], pass_id: str) -> None:
    """推进到下一 pass: 记录 passHistory, 重置 pass 步骤为 pending。"""
    normalized = pass_id.strip()
    if not normalized:
        raise SculptPipelineError("current pass cannot be empty")
    previous = str(state.get("currentPass") or "")
    if previous and previous != normalized:
        state.setdefault("passHistory", []).append(
            {"passId": previous, "checklist": deepcopy(_entries(state, "pass"))}
        )
    if previous != normalized:
        for entry in _entries(state, "pass"):
            entry["status"] = "pending"
            entry["evidence"] = []
            entry["reason"] = ""
        state["iterationAction"] = "new-pass" if previous else "initial"
    state["currentPass"] = normalized
    _recompute(state)


def record_refinement(state: dict[str, Any], pass_id: str, action: str) -> None:
    """记录一次纠错迭代 (来自 spec 的 reviewHistory 或调用方)。

    循环计数驱动硬停止: 单 pass 纠错 >= maxPerPass 或总纠错 >= maxTotal → stopped。
    """
    if action not in REFINE_ACTIONS:
        raise SculptPipelineError(f"refinement action must be one of {REFINE_ACTIONS}")
    state.setdefault("passHistory", []).append(
        {"passId": pass_id, "iteration": "refine", "checklist": deepcopy(_entries(state, "pass"))}
    )
    for entry in _entries(state, "pass"):
        entry["status"] = "pending"
        entry["evidence"] = []
        entry["reason"] = ""
    state["iterationAction"] = action
    loops = state["loops"]
    loops["perPass"][pass_id] = loops["perPass"].get(pass_id, 0) + 1
    loops["total"] = loops.get("total", 0) + 1
    pass_count = loops["perPass"][pass_id]
    if pass_count >= loops["maxPerPass"]:
        state["status"] = "stopped"
        state["currentStep"] = "stopped"
        state["stopReason"] = (
            f"max-correction-loops-reached:{pass_id}:{pass_count}/"
            f"{loops['maxPerPass']}"
        )
    elif loops["total"] >= loops["maxTotal"]:
        state["status"] = "stopped"
        state["currentStep"] = "stopped"
        state["stopReason"] = (
            f"max-total-correction-loops-reached:"
            f"{loops['total']}/{loops['maxTotal']}"
        )
    else:
        _recompute(state)


def pipeline_status(state: dict[str, Any]) -> dict[str, Any]:
    """当前状态载荷: 状态/当前步骤/pass/循环计数/下一步命令/待办/停止原因。"""
    entry = next_entry(state)
    current_pass = str(state.get("currentPass") or "")
    loops = state["loops"]
    visible_scopes = {"setup", "final"}
    if current_pass != "complete":
        visible_scopes.add("pass")
    return {
        "status": state["status"],
        "currentStep": state["currentStep"],
        "currentPass": current_pass,
        "loop": {
            "passCount": loops.get("perPass", {}).get(current_pass, 0),
            "maxPerPass": loops["maxPerPass"],
            "totalCount": loops.get("total", 0),
            "maxTotal": loops["maxTotal"],
        },
        "nextCommand": (
            None
            if state["status"] != "active" or entry is None
            else _format_command(state, entry)
        ),
        "stopReason": state.get("stopReason") or None,
        "pending": [
            entry["id"]
            for entry in state["checklist"]
            if entry["scope"] in visible_scopes and entry["status"] == "pending"
        ],
    }
