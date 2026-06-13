"""Auto-split from hicode whl."""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from ._exceptions import OprimError
from ._protocols import PersistenceHandle
from .text import count_tokens

class PromptOprimError(OprimError):
    """prompt 构建 / 消息处理失败。"""

class SnapshotOprimError(OprimError):
    """会话快照失败。"""

@dataclass
class ThinkingResult:
    """扩展思考提取结果。"""
    thinking: str
    text: str
    has_thinking: bool
    thinking_blocks: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)

@dataclass
class ConversationSnapshot:
    """会话快照结构。"""
    snapshot_id: str
    session_id: str
    message_count: int
    created_at: float
    store_key: str
    revision: str

def build_system_prompt(
    *,
    mode: str = "build",
    agents_md: str = "",
    tools_summary: str = "",
    skills_context: str = "",
    custom_sections: dict[str, str] | None = None,
    max_length: int | None = None,
) -> str:
    """构建完整的系统 prompt 字符串（纯计算）。

    按固定结构拼接各部分，保证 prompt 格式一致性：
      1. 角色声明（含模式）
      2. AGENTS.md / 项目记忆
      3. 可用工具摘要
      4. Skills 上下文
      5. 自定义节（custom_sections）

    Args:
        mode: 运行模式，"build"（默认）或 "plan"。
        agents_md: AGENTS.md 或项目记忆内容（空字符串则省略此节）。
        tools_summary: 可用工具的文本摘要（空字符串则省略此节）。
        skills_context: 已加载的 skill body 内容（空字符串则省略此节）。
        custom_sections: 额外节，dict[标题, 内容]。
        max_length: 输出字符数上限；超出时截断（优先保留前面的节）。

    Returns:
        完整系统 prompt 字符串。

    Raises:
        PromptOprimError: mode 不合法。

    Example:
        >>> prompt = build_system_prompt(
        ...     mode="build",
        ...     agents_md="# Project\\nThis is a Python web service.",
        ... )
        >>> "BUILD" in prompt
        True
    """
    if mode not in ("build", "plan"):
        raise PromptOprimError(f"invalid mode '{mode}': must be 'build' or 'plan'")

    parts: list[str] = []

    # 1. 角色声明
    mode_upper = mode.upper()
    role = (
        f"You are hicode, an expert AI coding agent operating in {mode_upper} mode.\n"
    )
    if mode == "plan":
        role += (
            "In PLAN mode: analyze the codebase and propose changes only. "
            "Do NOT write files, execute commands, or make any modifications.\n"
        )
    else:
        role += (
            "In BUILD mode: you may read files, write files, execute commands, "
            "and make all necessary changes to complete the task.\n"
        )
    parts.append(role)

    # 2. AGENTS.md / 项目记忆
    if agents_md and agents_md.strip():
        parts.append(f"## Project Memory\n{agents_md.strip()}")

    # 3. 工具摘要
    if tools_summary and tools_summary.strip():
        parts.append(f"## Available Tools\n{tools_summary.strip()}")

    # 4. Skills
    if skills_context and skills_context.strip():
        parts.append(f"## Skills\n{skills_context.strip()}")

    # 5. 自定义节
    for title, content in (custom_sections or {}).items():
        if content and content.strip():
            parts.append(f"## {title}\n{content.strip()}")

    result = "\n\n".join(parts)

    # 长度截断（优先保留前面节）
    if max_length is not None and len(result) > max_length:
        result = result[:max_length]

    return result
