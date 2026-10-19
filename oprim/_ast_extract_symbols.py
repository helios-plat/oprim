"""oprim.ast_extract_symbols — 符号提取（AST 单文件扁平符号表）.

单次解析 .py 文件并返回扁平符号列表（函数/类/方法/导入），
供 oskill.repomap_gen 做全局索引聚合。与 ast_parse_code 的区别：
本原子输出扁平符号表（同构于 obase.treesitter_indexer.SymbolRecord），
便于注入 indexer 协议的替换。

Example:
    >>> r = await ast_extract_symbols("/repo/app.py")
    >>> r["symbols"][0]["kind"]
    'function'
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

from oprim._exceptions import ParseOprimError

SymbolKind = Literal["function", "class", "method", "import", "constant"]


async def ast_extract_symbols(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    include_imports: bool = True,
    include_constants: bool = False,
) -> dict[str, Any]:
    """提取单文件符号表。

    Args:
        path: 目标 .py 文件路径。
        encoding: 文件编码。
        include_imports: 是否提取 import 符号。
        include_constants: 是否提取模块级常量（全大写名）。

    Returns:
        {
            "status": "ok", "path": str, "language": "python",
            "symbols": [{"name", "kind", "line", "column", "signature"}],
        }

    Raises:
        ParseOprimError: 文件缺失 / 语法错误。
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise ParseOprimError(f"ast_extract_symbols: file not found: {src}")
    try:
        source = src.read_text(encoding=encoding)
    except OSError as exc:
        raise ParseOprimError(f"ast_extract_symbols: cannot read {src}: {exc}", cause=exc) from exc

    try:
        tree = ast.parse(source, filename=str(src))
    except SyntaxError as exc:
        raise ParseOprimError(
            f"ast_extract_symbols: syntax error in {src} at line {exc.lineno}: {exc.msg}",
            cause=exc,
        ) from exc

    symbols: list[dict[str, Any]] = []

    def add(name: str, kind: SymbolKind, node: ast.AST, signature: str = "") -> None:
        symbols.append(
            {
                "name": name,
                "kind": kind,
                "line": node.lineno,
                "column": node.col_offset,
                "signature": signature,
            }
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(
                node.name,
                "function",
                node,
                signature=_fn_signature(node),
            )
        elif isinstance(node, ast.ClassDef):
            add(node.name, "class", node)
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(m.name, "method", m, signature=_fn_signature(m))
        elif isinstance(node, ast.Import) and include_imports:
            for a in node.names:
                add(a.name.split(".")[0], "import", node)
        elif isinstance(node, ast.ImportFrom) and include_imports:
            for a in node.names:
                add(a.name, "import", node)
        elif isinstance(node, ast.Assign) and include_constants:
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    add(t.id, "constant", node)

    return {
        "status": "ok",
        "path": str(src),
        "language": "python",
        "symbols": symbols,
    }


def _fn_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = ", ".join(a.arg for a in node.args.args)
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args})"
