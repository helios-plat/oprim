"""Stateless Python syntax-tree primitives.

These functions expose syntax only.  They do not resolve symbols through a
language server and do not own parser/provider lifecycle.
"""

from __future__ import annotations

import ast as py_ast
from typing import Any


def ast_parse(*, source: str, language: str = "python") -> py_ast.AST:
    """Parse source into a syntax tree using the standard-library parser."""
    if language.lower() not in {"python", "py"}:
        raise ValueError(f"unsupported AST language: {language!r}")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    return py_ast.parse(source)


def _tree(value: py_ast.AST | str, *, language: str) -> py_ast.AST:
    if isinstance(value, str):
        return ast_parse(source=value, language=language)
    if isinstance(value, py_ast.AST):
        return value
    raise TypeError("tree must be an AST or source string")


def ast_symbols(
    *, tree: py_ast.AST | str, language: str = "python"
) -> list[dict[str, Any]]:
    """Return syntactic declarations/imports without semantic resolution."""
    root = _tree(tree, language=language)
    symbols: list[dict[str, Any]] = []
    for node in py_ast.walk(root):
        name: str | None = None
        kind: str | None = None
        if isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
            name, kind = node.name, "function"
        elif isinstance(node, py_ast.ClassDef):
            name, kind = node.name, "class"
        elif isinstance(node, (py_ast.Import, py_ast.ImportFrom)):
            for alias in node.names:
                symbols.append({
                    "name": alias.asname or alias.name.split(".")[0],
                    "kind": "import",
                    "line": node.lineno,
                    "column": node.col_offset,
                })
        elif isinstance(node, py_ast.Assign):
            for target in node.targets:
                if isinstance(target, py_ast.Name):
                    symbols.append({
                        "name": target.id,
                        "kind": "variable",
                        "line": target.lineno,
                        "column": target.col_offset,
                    })
        if name is not None and kind is not None:
            symbols.append({
                "name": name,
                "kind": kind,
                "line": node.lineno,
                "column": node.col_offset,
            })
    return symbols


def ast_references(
    *, tree: py_ast.AST | str, symbol: str, language: str = "python"
) -> list[dict[str, Any]]:
    """Return syntactic name-load references; no LSP/provider lookup."""
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string")
    root = _tree(tree, language=language)
    return [
        {
            "name": node.id,
            "line": node.lineno,
            "column": node.col_offset,
            "context": "load",
        }
        for node in py_ast.walk(root)
        if isinstance(node, py_ast.Name)
        and node.id == symbol
        and isinstance(node.ctx, py_ast.Load)
    ]


__all__ = ["ast_parse", "ast_references", "ast_symbols"]
