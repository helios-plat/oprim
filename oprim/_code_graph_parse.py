"""oprim.code_graph_parse — 单次解析代码 AST，导出类、函数与调用的图节点与边.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: ast (stdlib).

例:
    >>> result = code_graph_parse("def foo(): pass\\nclass Bar: pass", file_path="mod.py")
    >>> len(result["nodes"]) >= 2
    True
"""

from __future__ import annotations

import ast
from typing import Any


def code_graph_parse(
    file_content: str,
    *,
    file_path: str = "sample.py",
) -> dict[str, Any]:
    """单次解析代码 AST，导出类、函数与调用的图节点与边 (graphify 机制).

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    提取:
    - 函数定义 (FunctionDef, AsyncFunctionDef)
    - 类定义 (ClassDef)
    - 函数调用关系 (Call → 被调函数名)
    - 导入关系 (Import, ImportFrom)

    Args:
        file_content: 源代码文本
        file_path: 文件路径（用于节点 ID 前缀）

    Returns:
        {
            "status": "success" | "failed",
            "nodes": [{"id": str, "type": str, "name": str, ...}],
            "edges": [{"source": str, "target": str, "relation": str}],
        }
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        return {
            "status": "failed",
            "nodes": [],
            "edges": [],
            "error": f"Syntax error: {e}",
        }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    current_scope = file_path

    # 根节点
    nodes.append({"id": current_scope, "type": "file", "name": file_path})

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_id = f"{file_path}::{node.name}"
            nodes.append(
                {
                    "id": func_id,
                    "type": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "is_async": False,
                }
            )
            edges.append({"source": current_scope, "target": func_id, "relation": "contains"})

        elif isinstance(node, ast.AsyncFunctionDef):
            func_id = f"{file_path}::{node.name}"
            nodes.append(
                {
                    "id": func_id,
                    "type": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "is_async": True,
                }
            )
            edges.append({"source": current_scope, "target": func_id, "relation": "contains"})

        elif isinstance(node, ast.ClassDef):
            class_id = f"{file_path}::{node.name}"
            nodes.append(
                {
                    "id": class_id,
                    "type": "class",
                    "name": node.name,
                    "lineno": node.lineno,
                }
            )
            edges.append({"source": current_scope, "target": class_id, "relation": "contains"})

            # 类内方法
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_id = f"{class_id}.{item.name}"
                    nodes.append(
                        {
                            "id": method_id,
                            "type": "method",
                            "name": item.name,
                            "lineno": item.lineno,
                        }
                    )
                    edges.append({"source": class_id, "target": method_id, "relation": "contains"})

        elif isinstance(node, ast.Call):
            # 提取函数调用关系
            if isinstance(node.func, ast.Name):
                caller = _find_enclosing_function(tree, node)
                if caller:
                    target_id = f"{file_path}::{node.func.id}"
                    edges.append(
                        {
                            "source": caller,
                            "target": target_id,
                            "relation": "calls",
                        }
                    )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(
                    {
                        "source": current_scope,
                        "target": f"import:{alias.name}",
                        "relation": "imports",
                    }
                )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    edges.append(
                        {
                            "source": current_scope,
                            "target": f"import:{node.module}.{alias.name}",
                            "relation": "imports",
                        }
                    )

    return {"status": "success", "nodes": nodes, "edges": edges}


def _find_enclosing_function(tree: ast.Module, target: ast.AST) -> str | None:
    """查找包含 target 节点的最内层函数/方法."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return node.name
    return None
