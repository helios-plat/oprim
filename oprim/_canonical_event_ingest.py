"""oprim._canonical_event_ingest — 规范化事件管道 (3O 原语)。

把任意 vendor 的原始输出字节 (claude/codex/pi/agentscope/raw) 归一为统一
AuditEvent schema + 指纹防篡改 + 持久化回放。Vigla canonical events 复刻。

分层: oprim (原语) — 只做归一/指纹/落盘, 不做业务判定。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from oprim._audit_emit import AuditEvent, JsonlSink

# vendor 原始事件类型 → 统一 EVENT_TYPES 白名单映射 (其余进 context.raw_type)
_VENDOR_EVENT_MAP: dict[str, str] = {
    # claude code stream-json
    "content_block_delta": "execute",
    "message_start": "diagnose",
    "message_stop": "learn",
    "tool_use": "execute",
    "tool_result": "execute",
    # codex / pi
    "text_delta": "execute",
    "tool_call": "execute",
    "session_done": "learn",
    # agentscope
    "message": "execute",
    "end": "learn",
    "error": "execute",
    # raw fallback
    "text": "execute",
    "output": "execute",
}

# 已知 vendor 标识 (用于解析分派; 未知 → raw 逐行)
_KNOWN_VENDORS = ("claude", "codex", "pi", "agentscope", "raw")


def compute_event_fingerprint(event: dict[str, Any]) -> str:
    """规范事件指纹 (sha256, 防篡改)。"""
    canonical = json.dumps(event, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_event_type(raw_type: str) -> tuple[str, str]:
    """vendor 事件类型 → (白名单 event_type, 原始类型)。"""
    return _VENDOR_EVENT_MAP.get(raw_type.lower(), "execute"), raw_type


def deserialize_vendor(raw_bytes: bytes, vendor: str) -> list[dict[str, Any]]:
    """vendor 原始字节 → 事件 dict 列表。

    claude stream-json / 其余 JSONL 逐行解析; 非 JSON 行 → {"type": "text", "content": line}。
    """
    text = raw_bytes.decode(errors="replace")
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                # stream-json: {"type": ..., "delta": {...}} / {"type": ..., "content": ...}
                events.append({
                    "type": parsed.get("type", "text"),
                    "content": (parsed.get("content") or parsed.get("delta")
                                or parsed.get("text") or ""),
                })
            else:
                events.append({"type": "text",
                               "content": str(parsed)[:2000]})
        except json.JSONDecodeError:
            events.append({"type": "text", "content": line[:2000]})
    return events


def canonical_event_ingest(
    raw_bytes: bytes,
    vendor: str,
    *,
    source: str = "",
    trace_id: str | None = None,
    sink: JsonlSink | None = None,
) -> dict[str, Any]:
    """主入口: 原始字节 → 规范事件 (归一 + 指纹 + 持久化)。

    Args:
        raw_bytes: vendor 原始输出字节
        vendor: claude | codex | pi | agentscope | raw
        source: 来源标识 (如 "mission/m1/worker/w1")
        trace_id: 链路 ID (缺省自动生成)
        sink: 落盘 sink (缺省 JsonlSink ~/.veya/audit/canonical-events.jsonl)

    Returns:
        {"canonical_event": dict, "fingerprint": str, "persisted": bool, "count": int}
    """
    vendor = vendor.lower()
    if vendor not in _KNOWN_VENDORS:
        vendor = "raw"
    tid = trace_id or f"ev_{uuid.uuid4().hex[:12]}"
    raw_events = deserialize_vendor(raw_bytes, vendor)

    persisted = False
    last_event: dict[str, Any] = {}
    if sink is None:
        from pathlib import Path

        sink = JsonlSink(str(Path.home() / ".veya" / "audit" / "canonical-events.jsonl"))

    for raw_evt in raw_events:
        event_type, raw_type = normalize_event_type(str(raw_evt.get("type", "text")))
        event = AuditEvent(
            event_type=event_type,
            trace_id=tid,
            inputs={"vendor": vendor, "source": source, "raw_type": raw_type},
            execution={"content": str(raw_evt.get("content", ""))[:4000]},
        )
        d = event.to_dict()
        d["inputs"]["fingerprint"] = compute_event_fingerprint(d)
        sink.write(AuditEvent.from_dict(d))
        persisted = True
        last_event = d

    return {
        "canonical_event": last_event,
        "fingerprint": last_event.get("inputs", {}).get("fingerprint", ""),
        "persisted": persisted,
        "count": len(raw_events),
        "trace_id": tid,
        "vendor": vendor,
    }


__all__ = ["canonical_event_ingest", "compute_event_fingerprint",
           "deserialize_vendor", "normalize_event_type"]
