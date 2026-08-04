"""oprim.ast_parse_code — 单次代码文件 AST 解析与符号提取.

纯计算原子（文件读取 + ast 解析），提取函数/类/导入/常量符号及行号，
供 oskill.code_symbol_investigate 构建依赖图。

Example:
    >>> r = ast_parse_code("/repo/app.py")
    >>> r["symbols"] >= 1
    True
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from oprim._exceptions import ParseOprimError


class AstParseError(ParseOprimError):
    """AST 解析失败。"""


async def ast_parse_code(
    path: str | Path,
    *,
    include_imports: bool = True,
    include_constants: bool = False,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """解析单文件 AST 并提取符号表。

    Args:
        path: 目标 .py 文件。
        include_imports: 是否提取 import 语句。
        include_constants: 是否提取模块级常量。
        encoding: 文件编码。

    Returns:
        {
            "status": "ok", "path": str, "language": "python",
            "functions": [{"name", "lineno", "args", "is_async", "decorators"}],
            "classes":   [{"name", "lineno", "bases", "methods"}],
            "imports":   [{"module", "names", "lineno"}],
            "constants": [{"name", "lineno"}],
            "symbols": int,
        }

    Raises:
        AstParseError: 文件缺失 / 语法错误。
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise AstParseError(f"ast_parse_code: file not found: {src}")

    try:
        source = src.read_text(encoding=encoding)
    except OSError as exc:
        raise AstParseError(f"ast_parse_code: cannot read {src}: {exc}", cause=exc) from exc

    try:
        tree = ast.parse(source, filename=str(src))
    except SyntaxError as exc:
        raise AstParseError(
            f"ast_parse_code: syntax error in {src} at line {exc.lineno}: {exc.msg}",
            cause=exc,
        ) from exc

    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    constants: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(_fn_info(node))
        elif isinstance(node, ast.ClassDef):
            methods = [
                _fn_info(n)
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append(
                {
                    "name": node.name,
                    "lineno": node.lineno,
                    "bases": [_unparse(n) for n in node.bases],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.Import) and include_imports:
            imports.append(
                {
                    "module": None,
                    "names": [a.name for a in node.names],
                    "lineno": node.lineno,
                }
            )
        elif isinstance(node, ast.ImportFrom) and include_imports:
            imports.append(
                {
                    "module": node.module,
                    "names": [a.name for a in node.names],
                    "lineno": node.lineno,
                }
            )
        elif isinstance(node, ast.Assign) and include_constants:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append({"name": target.id, "lineno": node.lineno})

    return {
        "status": "ok",
        "path": str(src),
        "language": "python",
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "constants": constants,
        "symbols": len(functions) + len(classes) + len(imports) + len(constants),
    }


def _fn_info(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = [a.arg for a in node.args.args]
    return {
        "name": node.name,
        "lineno": node.lineno,
        "args": args,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [_unparse(d) for d in node.decorator_list],
    }


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - 防御
        return type(node).__name__
