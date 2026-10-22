"""3O 范式内化 — Crux 切片器 (oprim 确定性原语, 无 LLM).

剥离注释/空白/物理行号, 规范化核心代码段 (Crux)。
目的: 上下游行增减不影响 Crux 的 Hash 稳定性。
"""

from __future__ import annotations

import ast
import hashlib
import re

_COMMENT_PATTERN = re.compile(r"#[^\n]*")


def extract_crux(code: str) -> str:
    """单次原子变换: 提取规范化核心代码段。

    Python 代码走 ast.unparse 规范化 (语法级稳定)

    其他代码走文本级归一化 (去注释/空白/空行)。
    """
    stripped = code.strip()
    if not stripped:
        return ""
    try:
        tree = ast.parse(stripped)
        return ast.unparse(tree)
    except SyntaxError:
        return _normalize_text(stripped)


def crux_hash(code: str) -> str:
    """Crux 的稳定 SHA256 (前 16 位)。"""
    return hashlib.sha256(extract_crux(code).encode("utf-8")).hexdigest()[:16]


def _normalize_text(code: str) -> str:
    # 1. 去掉字符串外的注释 (# 与 // 行注释, /* */ 块注释)
    text = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        no_comment = _COMMENT_PATTERN.sub("", line)
        no_comment = re.sub(r"//[^\n]*", "", no_comment)
        lines.append(no_comment.rstrip())
    # 2. 去除空行与行首尾空白
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    # 3. 折叠连续空行 (无), 归一化内部空白
    normalized = " ".join(non_empty)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
