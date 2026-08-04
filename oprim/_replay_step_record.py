"""oprim.replay_step_record — 回放步骤记录.

把一次回放步骤（run_id + 步骤 + 载荷）追加到 JSONL 回放日志
（recorder Protocol 注入，缺省本地 JSONL 追加）。

Example:
    >>> r = replay_step_record(3, run_id="run-1", payload={"action": "edit"})
    >>> r["recorded"]
    True
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class ReplayRecordError(OprimError):
    """回放记录失败。"""


@runtime_checkable
class ReplayRecorder(Protocol):
    """回放记录器协议（注入面）。"""

    def append(self, entry: dict[str, Any]) -> bool: ...


def replay_step_record(
    step: int,
    *,
    run_id: str,
    payload: dict[str, Any],
    recorder: ReplayRecorder | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """记录一个回放步骤。

    Args:
        step: 步骤号。
        run_id: 回放运行 ID。
        payload: 步骤载荷（动作/输入/输出）。
        recorder: 记录器（注入）；None 且给 log_path 时用 JSONL 追加。
        log_path: JSONL 日志路径（recorder 缺省时使用）。

    Returns:
        {"status": "ok", "recorded": bool, "entry": dict, "path": str|None}

    Raises:
        ReplayRecordError: 无 recorder 也无 log_path / 写入失败。
        OprimValidationError: step < 0 或 run_id 为空。
    """
    if step < 0:
        raise OprimValidationError("replay_step_record: step must be >= 0")
    if not run_id:
        raise OprimValidationError("replay_step_record: run_id must not be empty")

    entry = {
        "run_id": run_id,
        "step": step,
        "ts": time.time(),
        "payload": payload,
    }

    if recorder is not None:
        try:
            recorded = recorder.append(entry)
        except Exception as exc:
            raise ReplayRecordError(
                f"replay_step_record: recorder failed: {exc}", cause=exc
            ) from exc
        return {"status": "ok", "recorded": bool(recorded), "entry": entry, "path": None}

    if log_path is None:
        raise ReplayRecordError(
            "replay_step_record: need recorder or log_path to persist"
        )

    path = Path(log_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        raise ReplayRecordError(
            f"replay_step_record: cannot write {path}: {exc}", cause=exc
        ) from exc

    return {"status": "ok", "recorded": True, "entry": entry, "path": str(path)}
