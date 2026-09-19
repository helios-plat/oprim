"""3O 范式内化 — 语法树解析与符号提取 (oprim 确定性原语, 无 LLM).

Python 走零依赖 ast
其他语言委托 obase TreesitterIndexer (tree-sitter
可选依赖, 未安装时回退空集)。绝不调用 LLM。
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any


def parse_symbols(code: str, *, language: str = "python") -> list[dict[str, Any]]:
    """单次原子解析: 提取 Functions / Classes / Imports 符号网络。

    Args:
        code: 源码字符串。
        language: 语言标识 (python/typescript/go/rust/c/cpp/...)。

    Returns:
        符号记录列表: {name, kind, line, column, signature}。
    """
    if language == "python":
        return _parse_python(code)
    return _parse_via_obase(code, language)


def _parse_python(code: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "name": node.name,
                    "kind": "function",
                    "line": node.lineno,
                    "column": node.col_offset,
                    "signature": f"{node.name}({', '.join(a.arg for a in node.args.args)})",
                }
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "line": node.lineno,
                    "column": node.col_offset,
                    "signature": node.name,
                }
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append(
                    {
                        "name": alias.name,
                        "kind": "import",
                        "line": node.lineno,
                        "column": node.col_offset,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                symbols.append(
                    {
                        "name": f"{node.module}.{alias.name}" if node.module else alias.name,
                        "kind": "export" if node.module == "__init__" else "import",
                        "line": node.lineno,
                        "column": node.col_offset,
                    }
                )
    return symbols


def _parse_via_obase(code: str, language: str) -> list[dict[str, Any]]:
    """非 Python 语言: 临时文件 + obase TreesitterIndexer 委托。"""
    try:
        from obase.treesitter_indexer import TreesitterIndexer
    except ImportError:  # pragma: no cover
        return []

    suffix_map = {
        "typescript": ".ts",
        "javascript": ".js",
        "go": ".go",
        "rust": ".rs",
        "c": ".c",
        "cpp": ".cpp",
    }
    suffix = suffix_map.get(language, ".txt")
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        indexer = TreesitterIndexer()
        records = __import__("asyncio").run(indexer.index_file(tmp))
    except Exception:
        return []
    finally:
        Path(tmp).unlink(missing_ok=True)

    return [
        {
            "name": r.name,
            "kind": r.kind,
            "line": r.line,
            "column": r.column,
            "signature": r.signature,
        }
        for r in records
    ]
