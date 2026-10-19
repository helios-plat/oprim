"""oprim.file_read_slice — 大文件按 Token/行数截断读取.

按 max_lines / max_tokens 从 offset_lines 起截断读取大文件，防止一次性
读入超长内容打爆上下文。token 估算采用 chars/4 启发式（文档化近似）。

Example:
    >>> r = await file_read_slice("/tmp/big.log", max_lines=10)
    >>> r["truncated"]
    True
    >>> len(r["content"].splitlines())
    10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._exceptions import FileOprimError, OprimValidationError

_CHARS_PER_TOKEN = 4.0


async def file_read_slice(
    path: str | Path,
    *,
    max_tokens: int | None = None,
    max_lines: int | None = None,
    offset_lines: int = 0,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """按行/Token 截断读取文件。

    Args:
        path: 目标文件路径。
        max_tokens: 内容 token 估算上限（chars/4 近似）；None 不限制。
        max_lines: 读取行数上限；None 不限制。
        offset_lines: 起始行偏移（0 = 从头）。
        encoding: 文本编码。

    Returns:
        {"status": "ok", "path": str, "content": str, "total_lines": int,
         "total_chars": int, "truncated": bool, "reason": str | None}

    Raises:
        FileOprimError: 文件不存在 / 读取失败。
        OprimValidationError: 参数非法（负值 / 双上限全空）。
    """
    if max_tokens is not None and max_tokens <= 0:
        raise OprimValidationError("file_read_slice: max_tokens must be > 0")
    if max_lines is not None and max_lines <= 0:
        raise OprimValidationError("file_read_slice: max_lines must be > 0")
    if offset_lines < 0:
        raise OprimValidationError("file_read_slice: offset_lines must be >= 0")
    if max_tokens is None and max_lines is None:
        raise OprimValidationError(
            "file_read_slice: at least one of max_tokens / max_lines required"
        )

    src = Path(path).expanduser()
    if not src.is_file():
        raise FileOprimError(f"file_read_slice: file not found: {src}")

    try:
        with src.open("r", encoding=encoding, errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise FileOprimError(
            f"file_read_slice: cannot read {src}: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    total_lines = len(lines)
    selected = lines[offset_lines:]

    truncated = False
    reason: str | None = None
    if max_lines is not None and len(selected) > max_lines:
        selected = selected[:max_lines]
        truncated = True
        reason = "max_lines"

    content = "".join(selected)
    if max_tokens is not None:
        est = len(content) / _CHARS_PER_TOKEN
        if est > max_tokens:
            # 按比例截断到预算的 90%，留余量
            keep_chars = int(max_tokens * _CHARS_PER_TOKEN * 0.9)
            content = content[:keep_chars]
            truncated = True
            reason = "max_tokens"

    return {
        "status": "ok",
        "path": str(src),
        "content": content,
        "total_lines": total_lines,
        "total_chars": len(content),
        "truncated": truncated,
        "reason": reason,
    }
