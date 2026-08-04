"""oprim.invariant_extract — 从源代码中原子抽取安全不变性断言.

单次使用 Python AST 抽离 assert 语句与条件守卫（raise 前置的 if）。

Example:
    >>> r = invariant_extract("def f(x):\n    assert x > 0\n    return x")
    >>> r["invariants"]
    ['x > 0']
"""

from __future__ import annotations

import ast
from typing import Any

from oprim._exceptions import OprimValidationError


def invariant_extract(
    code_content: str,
    *,
    target_function: str = "",
) -> dict[str, Any]:
    """抽取安全不变性断言。

    Args:
        code_content: Python 源代码。
        target_function: 目标函数名（可选；给出时只抽取该函数内的断言）。

    Returns:
        {"status": "success", "invariants": [str]}
        语法错误时 status="failed" + error（不 raise）。

    Raises:
        OprimValidationError: code_content 为空。
    """
    if not code_content or not code_content.strip():
        raise OprimValidationError("invariant_extract: code_content must not be empty")

    invariants: list[str] = []
    try:
        tree = ast.parse(code_content)
    except SyntaxError as exc:
        return {
            "status": "failed",
            "invariants": [],
            "error": f"line {exc.lineno}: {exc.msg}",
        }

    def in_target(node: ast.AST) -> bool:
        if not target_function:
            return True
        for parent_scope in _scopes_of(tree, node):
            if (
                isinstance(parent_scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent_scope.name == target_function
            ):
                return True
        return False

    for node in ast.walk(tree):
        if not in_target(node):
            continue
        if isinstance(node, ast.Assert):
            invariants.append(ast.unparse(node.test))
        elif (
            isinstance(node, ast.If)
            and node.body
            and isinstance(node.body[0], ast.Raise)
        ):
            invariants.append(f"Not({ast.unparse(node.test)})")

    return {"status": "success", "invariants": invariants}


def _scopes_of(tree: ast.AST, node: ast.AST) -> list[ast.AST]:
    """返回 node 的祖先作用域链（含自身）。"""
    scopes: list[ast.AST] = []

    def walk(current: ast.AST) -> bool:
        if current is node:
            scopes.append(current)
            return True
        for child in ast.iter_child_nodes(current):
            if walk(child):
                scopes.append(current)
                return True
        return False

    walk(tree)
    return scopes
