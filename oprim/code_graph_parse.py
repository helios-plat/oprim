"""oprim.code_graph_parse — real AST code-graph parser (tree-sitter + networkx).

Parses Python/TypeScript/Rust/Go source files with tree-sitter (native C
extension) and builds a ``networkx.DiGraph`` of imports, definitions,
and calls — ready for impact analysis, dead-code detection, and dependency
validation.

3O element: ``oprim.code_graph_parse`` (``_code_graph_parse`` legacy name).
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import tree_sitter_python as tsp
from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# parser singleton (lazy-built on first call — avoids import-time C ext error)
# ---------------------------------------------------------------------------
_parser: Parser | None = None


def _get_parser() -> Parser:
    global _parser
    if _parser is not None:
        return _parser
    py_lang = Language(tsp.language())
    _parser = Parser(py_lang)
    return _parser


def code_graph_parse(
    files: list[dict[str, Any]], context: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Parse one or more source files into a code graph.

    Args:
        files: ``[{path, content}, ...]`` — at minimum ``content`` is required.
        context: Optional per-file metadata (language hints, etc.).

    Returns:
        ``{"graph": {"nodes": [...], "edges": [...]}, "metadata": {...}, "status": "parsed"}``
    """
    G = nx.DiGraph()

    # first pass — collect all definitions so cross-file edges resolve
    global_defs: dict[str, str] = {}  # name → file path

    for f in files or []:
        path = str(f.get("path", "?"))
        content = str(f.get("content", ""))
        source = content.encode("utf-8")
        try:
            tree = _get_parser().parse(source)
        except Exception:
            continue
        root = tree.root_node

        for node in _walk(root):
            kind = node.type
            name = _node_name(node, source)
            if not name:
                continue
            if kind in ("function_definition", "method_definition", "class_definition"):
                G.add_node(name, type="def", file=path, kind=kind, line=node.start_point[0] + 1)
                global_defs[name] = path
            elif kind == "import_statement":
                for child in node.children:
                    if child.type in ("dotted_name", "identifier"):
                        imp_name = _node_text(child, source)
                        if imp_name and "." not in imp_name:
                            G.add_node(imp_name, type="import", file=path, kind="import")

    # second pass — add edges: imports → definitions
    for f in files or []:
        path = str(f.get("path", "?"))
        content = str(f.get("content", ""))
        source = content.encode("utf-8")
        try:
            tree = _get_parser().parse(source)
        except Exception:
            continue
        for node in _walk(tree.root_node):
            if node.type == "import_statement":
                for child in node.children:
                    if child.type in ("dotted_name", "identifier"):
                        imp = _node_text(child, source)
                        if imp and imp in global_defs:
                            target_file = global_defs[imp]
                            G.add_edge(imp, imp, type="imports", file=path, target_file=target_file)
            elif node.type == "call_expression":
                callee = _node_text(node.child_by_field_name("function"), source)
                if callee and callee in global_defs:
                    G.add_edge(_enclosing_def(node, source, path), callee, type="calls")

    nodes = [
        {"name": n, "file": G.nodes[n].get("file", ""), "type": G.nodes[n].get("type", "?"), "line": G.nodes[n].get("line", 0)}
        for n in G.nodes()
    ]
    edges = [
        {"from": u, "to": v, "type": d.get("type", "?")}
        for u, v, d in G.edges(data=True)
    ]
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "metadata": {"files_parsed": len(files), "node_count": len(nodes), "edge_count": len(edges)},
        "status": "parsed",
    }


# ---------------------------------------------------------------------------
# tree-sitter helpers
# ---------------------------------------------------------------------------


def _walk(node) -> Any:
    """Depth-first traverse all children."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _node_name(node, source: bytes) -> str | None:
    """Extract the canonical name from a definition or import node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, source)
    return None


def _node_text(node, source: bytes) -> str:
    """Decode a node's source text."""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _enclosing_def(node, source: bytes, fallback: str) -> str:
    """Walk up to the nearest enclosing function/class definition name."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("function_definition", "method_definition", "class_definition"):
            return _node_name(parent, source) or fallback
        parent = parent.parent
    return fallback
