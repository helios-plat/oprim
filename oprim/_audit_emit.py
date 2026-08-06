"""oprim._audit_emit — 决策审计统一写出口 (AuditEmitter)。

原料散落 (ActionLog 只记执行 / notes 只给人看 / 版本号在 Store 里),
没有一个地方按统一 Schema 把一次完整决策链路写出来。本模块就是那个地方:

在 diagnose → plan → decide → execute → learn 每个关键节点, 自动写出一条
结构化审计记录 —— 事后可回答: 为什么选这个动作? 用的哪版因果图/CPD?
谁授权执行的? 蜜罐触发后有没有正确隔离?

原则:
  1. **它不做决策**, 只负责把已经发生的决策按规范记下来 (生产落地的薄封装);
  2. 纯机制: sink 注入 (JSONL 文件 / 内存 / 复合), 不绑定任何持久化实现;
  3. trace_id 贯穿一次故障处理链路, audit_id 唯一标识单条记录。

统一 Schema (to_dict):
    {
      "audit_id", "trace_id", "ts", "event_type",
      "inputs":    {"graph_version", "cpd_version", "threat_level", ...},
      "decision":  {"chosen_strategy", "utilities", ...} | None,
      "execution": {"primitive", "status", "capability_nonce", ...} | None,
      "learning":  {"cpd_version_after", ...} | None,
      "context":   {"notes", "failure_context", ...}
    }
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# 决策链路的关键节点 (event_type 白名单)
EVENT_TYPES = ("diagnose", "plan", "decide", "execute", "learn")

_REQUIRED_KEYS = ("audit_id", "trace_id", "ts", "event_type", "inputs")


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class AuditEvent:
    """单条审计记录 (统一 Schema)。"""

    event_type: str
    trace_id: str
    audit_id: str = field(default_factory=new_id)
    ts: float = field(default_factory=time.time)
    inputs: Dict[str, Any] = field(default_factory=dict)      # graph_version/cpd_version/...
    decision: Optional[Dict[str, Any]] = None                  # chosen_strategy/utilities/...
    execution: Optional[Dict[str, Any]] = None                  # primitive/status/capability_nonce
    learning: Optional[Dict[str, Any]] = None                   # cpd_version_after/...
    context: Dict[str, Any] = field(default_factory=dict)       # notes/failure_context/...

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"event_type 必须是 {EVENT_TYPES} 之一, 收到 {self.event_type!r}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in _REQUIRED_KEYS} | {
            "decision": self.decision, "execution": self.execution,
            "learning": self.learning, "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        return cls(
            event_type=data["event_type"],
            trace_id=data["trace_id"],
            audit_id=data.get("audit_id", new_id()),
            ts=float(data.get("ts", time.time())),
            inputs=data.get("inputs", {}),
            decision=data.get("decision"),
            execution=data.get("execution"),
            learning=data.get("learning"),
            context=data.get("context", {}),
        )


# ---------------------------------------------------------------------------
# Sink (写入目的地注入)
# ---------------------------------------------------------------------------

class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...
    def read_trace(self, trace_id: str) -> List[Dict[str, Any]]: ...


class MemorySink:
    """内存 sink (测试/进程内回放)。"""

    def __init__(self) -> None:
        self.events: List[AuditEvent] = []

    def write(self, event: AuditEvent) -> None:
        self.events.append(event)

    def read_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.events if e.trace_id == trace_id]


class JsonlSink:
    """JSONL 文件 sink (生产默认): 追加写 + 按 trace_id 回放。"""

    def __init__(self, path: str, *, append: bool = True):
        self.path = path
        if not append:
            open(path, "w").close()
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, event: AuditEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def read_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        out = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("trace_id") == trace_id:
                        out.append(data)
        except FileNotFoundError:
            return []
        return out

    def read_all(self, limit: int = 0) -> List[Dict[str, Any]]:
        out = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                    if limit and len(out) >= limit:
                        break
        except FileNotFoundError:
            return []
        return out


class CompositeSink:
    """同时写入多个 sink (文件 + 内存 + 遥测...)。"""

    def __init__(self, sinks: List[AuditSink]):
        self.sinks = sinks

    def write(self, event: AuditEvent) -> None:
        for s in self.sinks:
            s.write(event)

    def read_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        for s in self.sinks:
            try:
                return s.read_trace(trace_id)
            except (AttributeError, NotImplementedError):
                continue
        return []


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class AuditEmitter:
    """决策审计统一写出口 —— 不做决策, 只按规范记录。

    Args:
        sink: 写入目的地; 缺省 MemorySink (进程内)。
        trace_id: 显式指定链路 ID; 缺省每次 emit 自动新开 (单发场景)。
    """

    def __init__(self, sink: Optional[AuditSink] = None, trace_id: Optional[str] = None):
        self.sink = sink or MemorySink()
        self.trace_id = trace_id or new_id()

    def emit(
        self,
        event_type: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        execution: Optional[Dict[str, Any]] = None,
        learning: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """写一条审计记录, 返回 audit_id。"""
        event = AuditEvent(
            event_type=event_type,
            trace_id=self.trace_id,
            inputs=inputs or {},
            decision=decision,
            execution=execution,
            learning=learning,
            context=context or {},
        )
        self.sink.write(event)
        return event.audit_id

    # ── 链路节点快捷方法 ────────────────────────────────────────────
    def diagnose(self, **kw: Any) -> str:
        return self.emit("diagnose", **kw)

    def plan(self, **kw: Any) -> str:
        return self.emit("plan", **kw)

    def decide(self, **kw: Any) -> str:
        return self.emit("decide", **kw)

    def execute(self, **kw: Any) -> str:
        return self.emit("execute", **kw)

    def learn(self, **kw: Any) -> str:
        return self.emit("learn", **kw)

    def replay(self) -> List[Dict[str, Any]]:
        """回放本 trace 的完整决策链路 (按写入顺序)。"""
        return self.sink.read_trace(self.trace_id)


__all__ = ["AuditEmitter", "AuditEvent", "AuditSink", "CompositeSink", "EVENT_TYPES",
           "JsonlSink", "MemorySink", "new_id"]
