"""oprim._spec_parse — 可执行 Spec 解析原语 (spec-kit 3O 内化)。

结构化 spec (markdown 章节或 JSON) → 字段提取 + 完整性校验。
Spec-Driven Development: spec 直接生成实现 (非仅指导)。

章节约定 (兼容 docs/prd 风格):
  ## 目标 / ## 验收标准 / ## 约束 / ## 测试门 / ## 上下文
"""

from __future__ import annotations

import json
import re
from typing import Any

# spec 标准章节 (键 → 章节标题别名)
SECTION_ALIASES: dict[str, list[str]] = {
    "goal": ["目标", "目标与背景", "goal"],
    "acceptance": ["验收标准", "验收", "acceptance", "AC"],
    "constraints": ["约束", "非目标", "constraints", "non_goals"],
    "test_gate": ["测试门", "测试策略", "test_gate", "test_strategy"],
    "context": ["上下文", "背景", "context"],
}

REQUIRED_SECTIONS = ("goal", "acceptance", "test_gate")

_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


def parse_spec(text: str) -> dict[str, Any]:
    """markdown spec → 结构化 {goal, acceptance[], constraints[], test_gate, context}。"""
    if text.lstrip().startswith("{"):
        try:
            return _normalize_dict(json.loads(text))
        except json.JSONDecodeError:
            pass  # 非 JSON, 按 markdown 解析

    headings = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(headings):
        title = m.group(1).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        # 标题别名匹配
        for key, aliases in SECTION_ALIASES.items():
            if title in aliases or title.lower() in aliases:
                sections[key] = sections.get(key, "") + "\n" + body
                break
        else:
            sections.setdefault("_extra_" + title, body)

    return _normalize_dict(sections)


def _normalize_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """字段归一: acceptance/constraints → list (按行/列表项拆)。"""
    spec: dict[str, Any] = {
        "goal": str(raw.get("goal", "")).strip(),
        "acceptance": _to_list(raw.get("acceptance", "")),
        "constraints": _to_list(raw.get("constraints", "")),
        "test_gate": str(raw.get("test_gate", "")).strip(),
        "context": str(raw.get("context", "")).strip(),
    }
    # JSON 形态直接字段
    for k in ("goal", "test_gate", "context"):
        if isinstance(raw.get(k), str):
            spec[k] = raw[k]
    if isinstance(raw.get("acceptance"), list):
        spec["acceptance"] = [str(a).strip() for a in raw["acceptance"] if str(a).strip()]
    if isinstance(raw.get("constraints"), list):
        spec["constraints"] = [str(c).strip() for c in raw["constraints"] if str(c).strip()]
    return spec


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str):
        return []
    out = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•\d\.\)\s\[\]]+", "", line).strip()
        line = re.sub(r"^(\[\s?x?\]|\[ \]|\[x\])\s*", "", line, flags=re.I).strip()
        if line:
            out.append(line)
    return out


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """完整性校验: 必填章节 (goal/acceptance/test_gate)。"""
    missing = [k for k in REQUIRED_SECTIONS if not spec.get(k)]
    return {
        "ok": not missing,
        "missing": missing,
        "issues": [f"缺少章节: {k}" for k in missing],
        "spec": spec,
    }


__all__ = ["parse_spec", "validate_spec", "SECTION_ALIASES", "REQUIRED_SECTIONS"]
