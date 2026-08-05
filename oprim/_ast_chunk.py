"""oprim.ast_chunk — single Python source AST chunking call.

3O layer: oprim (single atomic parse, pure logic, no LLM).
Splits Python source into function/method/class-level chunks — signature +
docstring + short body preview — for context-compressed semantic indexing
(consumed by oskill.workspace_rag).

Companion: ``code_tokens`` splits code identifiers (camelCase / snake_case)
for code-aware tokenization.
"""

from __future__ import annotations

import ast
import re
from typing import Any

_MAX_BODY_LINES = 15
_MAX_CHUNKS_PER_FILE = 200

# Code-search stopwords (python keywords + common noise)
_STOPWORDS = frozenset(
    {
        "def", "class", "return", "import", "from", "self", "if", "else", "elif",
        "for", "while", "try", "except", "with", "as", "pass", "lambda", "yield",
        "and", "or", "not", "in", "is", "none", "true", "false", "the", "a", "an",
        "of", "to", "this", "that", "it", "be", "are", "on",
    }
)
_IDENTIFIER_SPLIT_RE = re.compile(r"_+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def code_tokens(text: str) -> list[str]:
    """Code-aware tokenization: split camelCase/snake_case identifiers.

    Returns alphanumeric word stems (len >= 2, stopwords removed).
    """
    tokens = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()):
        for part in _IDENTIFIER_SPLIT_RE.split(raw):
            part = part.strip("_")
            if len(part) >= 2 and part not in _STOPWORDS:
                tokens.append(part)
    return tokens


def _func_signature(node: ast.AST) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    ret = ""
    if getattr(node, "returns", None) is not None:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            ret = ""
    return f"def {node.name}({args}){ret}"


def _body_preview(body: list[ast.stmt]) -> str:
    lines = []
    for stmt in body[: _MAX_BODY_LINES]:
        try:
            lines.append(ast.unparse(stmt))
        except Exception:
            continue
    return "\n".join(lines)[:800]


def ast_chunk_python(
    *,
    source: str,
    filepath: str,
    max_body_lines: int = _MAX_BODY_LINES,
    max_chunks: int = _MAX_CHUNKS_PER_FILE,
) -> list[dict[str, Any]]:
    """Split Python source into function/method/class-level chunks.

    Args:
        source: Full python source text.
        filepath: Logical path used in chunk ids/metadata (not read from disk).

    Returns:
        List of {chunk_id, content, metadata} where metadata carries
        file/type/name/start_line/end_line. Empty list on parse error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks: list[dict[str, Any]] = []

    def _doc(node: ast.AST) -> str:
        return ast.get_docstring(node) or ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = _doc(node)
            content = f"{_func_signature(node)}\n"
            if doc:
                content += f'"""{doc[:300]}"""\n'
            content += _body_preview(node.body)
            chunks.append(
                {
                    "chunk_id": f"{filepath}:{node.name}:{node.lineno}",
                    "content": content,
                    "metadata": {
                        "file": filepath,
                        "type": "function",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    },
                }
            )
        elif isinstance(node, ast.ClassDef):
            doc = _doc(node)
            base = ""
            if node.bases:
                try:
                    base = f"({', '.join(ast.unparse(b) for b in node.bases)})"
                except Exception:
                    base = ""
            content = f"class {node.name}{base}\n"
            if doc:
                content += f'"""{doc[:300]}"""\n'
            method_sigs = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_sigs.append(f"    {_func_signature(item)}")
            content += "\n".join(method_sigs[:20])
            chunks.append(
                {
                    "chunk_id": f"{filepath}:{node.name}:{node.lineno}",
                    "content": content,
                    "metadata": {
                        "file": filepath,
                        "type": "class",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    },
                }
            )
        if len(chunks) >= max_chunks:
            break
    return chunks
