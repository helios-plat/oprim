"""oprim.replay_step_record — DeerFlow/LLM-Space step capture & replay.

Records every harness step (LLM call, tool execution, sub-agent dispatch) as
a serializable event that can be replayed in the LLM Space desktop tool for
single-step debugging, failure replay, and benchmarking.

3O element: ``oprim.replay_step_record``.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def replay_step_record(
    step: dict[str, Any],
    thread_id: str | None = None,
    output_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record one agent-harness step to a replay log.

    Step dict keys:
      * ``step_no`` — monotonic step number
      * ``kind`` — "llm_call" | "tool_call" | "tool_result" | "subagent_dispatch" | "thinking" | "error"
      * ``messages`` — optional list of messages at this step
      * ``tool_calls`` — optional list of tool calls
      * ``tool_results`` — optional list of tool results
      * ``cost`` — optional cost in USD
      * ``elapsed_ms`` — optional wall-clock duration
    """
    ctx = context or {}
    base = Path(output_dir) if output_dir else Path.home() / ".veya" / "replays"
    base.mkdir(parents=True, exist_ok=True)
    tid = thread_id or ctx.get("thread_id") or uuid.uuid4().hex[:8]

    record = {
        "thread_id": tid,
        "step_no": step.get("step_no", 0),
        "kind": step.get("kind", "step"),
        "ts": time.time(),
        "data": _truncate(step, ctx.get("max_chars", 8000)),
    }
    path = base / f"{tid}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return {"status": "recorded", "path": str(path), "step_no": record["step_no"]}


def load_replay(
    thread_id: str,
    replay_dir: str | Path | None = None,
    max_steps: int = 1000,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load all recorded steps for a replay session.

    Returns {status, thread_id, steps: [record, ...], total_steps, errors: [...]}
    """
    ctx = context or {}
    base = Path(replay_dir) if replay_dir else Path.home() / ".veya" / "replays"
    path = base / f"{thread_id}.jsonl"
    if not path.exists():
        return {"status": "not_found", "thread_id": thread_id, "steps": [], "total_steps": 0}

    steps: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(json.loads(line))
        except Exception:
            errors.append(line[:120])
        if len(steps) >= max_steps:
            break
    return {"status": "loaded", "thread_id": thread_id, "steps": steps, "total_steps": len(steps), "errors": errors}


def replay_analysis(
    steps: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Analyze replay steps for failures and bottlenecks.

    Returns {status, failures, bottlenecks, cost_total, elapsed_total_ms, summary}
    """
    failures: list[int] = []
    costs = 0.0
    elapsed = 0
    kinds: dict[str, int] = {}
    for s in steps:
        k = s.get("kind", "step")
        kinds[k] = kinds.get(k, 0) + 1
        data = s.get("data", {})
        if data.get("error") or k == "error":
            failures.append(s.get("step_no", 0))
        costs += float(data.get("cost", 0) or 0)
        elapsed += int(data.get("elapsed_ms", 0) or 0)

    return {
        "status": "analyzed",
        "total_steps": len(steps),
        "failures": failures,
        "failure_count": len(failures),
        "cost_total_usd": round(costs, 6),
        "elapsed_total_ms": elapsed,
        "kind_distribution": kinds,
        "summary": f"{len(steps)} steps, {len(failures)} failures, ${costs:.4f}, {elapsed}ms",
    }


def _truncate(obj: Any, max_chars: int) -> Any:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return obj
    if isinstance(obj, dict):
        return {k: _truncate(v, max_chars // max(1, len(obj))) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, max_chars // max(1, len(obj))) for v in obj]
    return s[:max_chars] + "...[truncated]"
