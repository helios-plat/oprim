"""
oprim: 批次 B — LSP 单次 JSON-RPC 请求集
=========================================
包含（10个）：
    lsp_diagnostics / lsp_hover / lsp_definition / lsp_references
    lsp_document_symbols / lsp_workspace_symbols / lsp_rename
    lsp_completion / lsp_format / lsp_code_action

M4 Owner 裁决实现
-----------------
server 参数类型是 LspServerHandle Protocol（在 _protocols.py 定义）。
oprim 内部只调用 server.request(method, params)，不 import obase.lsp。
依赖方向：调用方持有 handle 并注入，oprim 对 obase 无感知。V1 守住。

归属约束
--------
✅ 每个函数 = 对运行中语言服务器的单次 JSON-RPC 请求
✅ server 参数通过调用方注入（Protocol，不是模块导入）
✅ async 本性：等待 LSP 响应是 IO 等待
✅ 互不裸调，不写 trail，不做业务编排
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._exceptions import OprimError
from ._protocols import LspServerHandle


class LspOprimError(OprimError):
    """LSP 请求失败。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _file_uri(path: str | Path) -> str:
    """路径 → LSP file URI。"""
    p = Path(path).resolve()
    return f"file://{p}"


def _position(line: int, character: int) -> dict:
    return {"line": line, "character": character}


def _text_doc(path: str | Path) -> dict:
    return {"uri": _file_uri(path)}


def _text_doc_pos(path: str | Path, line: int, character: int) -> dict:
    return {
        "textDocument": _text_doc(path),
        "position": _position(line, character),
    }


# ---------------------------------------------------------------------------
# 数据类型
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    path: str
    line: int              # 0-based
    character: int
    end_line: int
    end_character: int
    severity: int          # 1=Error 2=Warning 3=Info 4=Hint
    message: str
    source: str
    code: str | int | None = None

    @property
    def severity_name(self) -> str:
        return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(self.severity, "unknown")


@dataclass
class Hover:
    contents: str          # Markdown 格式
    range_start_line: int | None = None
    range_start_char: int | None = None


@dataclass
class Location:
    path: str
    start_line: int
    start_character: int
    end_line: int
    end_character: int


@dataclass
class Symbol:
    name: str
    kind: int              # SymbolKind enum
    path: str
    start_line: int
    start_character: int
    end_line: int
    end_character: int
    container: str = ""

    @property
    def kind_name(self) -> str:
        _kinds = {
            1: "File", 2: "Module", 3: "Namespace", 4: "Package",
            5: "Class", 6: "Method", 7: "Property", 8: "Field",
            9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
            13: "Variable", 14: "Constant", 15: "String", 16: "Number",
        }
        return _kinds.get(self.kind, f"Kind({self.kind})")


@dataclass
class Completion:
    label: str
    kind: int | None = None
    detail: str = ""
    documentation: str = ""
    insert_text: str = ""


@dataclass
class TextEdit:
    start_line: int
    start_character: int
    end_line: int
    end_character: int
    new_text: str


@dataclass
class WorkspaceEdit:
    changes: dict[str, list[TextEdit]] = field(default_factory=dict)
    # path → list[TextEdit]


@dataclass
class CodeAction:
    title: str
    kind: str                      # "quickfix" | "refactor" | ...
    diagnostics: list[Diagnostic] = field(default_factory=list)
    edit: WorkspaceEdit | None = None
    command: dict | None = None


# ---------------------------------------------------------------------------
# lsp_diagnostics
# ---------------------------------------------------------------------------

async def lsp_diagnostics(
    path: str | Path,
    *,
    server: LspServerHandle,
) -> list[Diagnostic]:
    """单次获取文件的诊断信息（错误/警告）。

    Args:
        path: 文件路径。
        server: LSP server handle（由 obase.lsp 注入）。

    Returns:
        Diagnostic 列表，按行号排序。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> diags = await lsp_diagnostics("src/main.py", server=server)
        >>> [d.message for d in diags if d.severity == 1]
        ['undefined name foo']
    """
    try:
        result = await server.request(
            "textDocument/diagnostic",
            {"textDocument": _text_doc(path)},
        )
    except Exception as e:
        raise LspOprimError(f"lsp_diagnostics failed for '{path}'", cause=e)

    items = result.get("items", []) if isinstance(result, dict) else (result or [])
    diags = []
    for d in items:
        r = d.get("range", {})
        start = r.get("start", {})
        end = r.get("end", {})
        diags.append(Diagnostic(
            path=str(Path(path).resolve()),
            line=start.get("line", 0),
            character=start.get("character", 0),
            end_line=end.get("line", 0),
            end_character=end.get("character", 0),
            severity=d.get("severity", 1),
            message=d.get("message", ""),
            source=d.get("source", ""),
            code=d.get("code"),
        ))
    return sorted(diags, key=lambda d: (d.line, d.character))


# ---------------------------------------------------------------------------
# lsp_hover
# ---------------------------------------------------------------------------

async def lsp_hover(
    path: str | Path,
    *,
    line: int,
    character: int,
    server: LspServerHandle,
) -> Hover | None:
    """单次获取光标位置的 hover 信息（类型/文档）。

    Args:
        path: 文件路径。
        line: 行号（0-based）。
        character: 列号（0-based）。
        server: LSP server handle。

    Returns:
        Hover（含 Markdown 格式文档），或 None（无 hover 信息）。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> h = await lsp_hover("src/main.py", line=10, character=5, server=server)
        >>> h.contents
        '```python\ndef foo() -> int\n```'
    """
    try:
        result = await server.request(
            "textDocument/hover",
            _text_doc_pos(path, line, character),
        )
    except Exception as e:
        raise LspOprimError(f"lsp_hover failed for '{path}':{line}", cause=e)

    if not result:
        return None

    contents = result.get("contents", "")
    if isinstance(contents, dict):
        contents = contents.get("value", "")
    elif isinstance(contents, list):
        contents = "\n".join(
            c.get("value", c) if isinstance(c, dict) else str(c)
            for c in contents
        )

    r = result.get("range", {})
    start = r.get("start", {}) if r else {}
    return Hover(
        contents=str(contents),
        range_start_line=start.get("line"),
        range_start_char=start.get("character"),
    )


# ---------------------------------------------------------------------------
# lsp_definition
# ---------------------------------------------------------------------------

async def lsp_definition(
    path: str | Path,
    *,
    line: int,
    character: int,
    server: LspServerHandle,
) -> list[Location]:
    """单次跳转到定义位置。

    Args:
        path: 文件路径。
        line: 行号（0-based）。
        character: 列号（0-based）。
        server: LSP server handle。

    Returns:
        Location 列表（通常 0-1 个）。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> locs = await lsp_definition("src/main.py", line=5, character=10, server=server)
        >>> locs[0].path
        '/project/src/utils.py'
    """
    try:
        result = await server.request(
            "textDocument/definition",
            _text_doc_pos(path, line, character),
        )
    except Exception as e:
        raise LspOprimError("lsp_definition failed", cause=e)

    return _parse_locations(result)


# ---------------------------------------------------------------------------
# lsp_references
# ---------------------------------------------------------------------------

async def lsp_references(
    path: str | Path,
    *,
    line: int,
    character: int,
    server: LspServerHandle,
    include_declaration: bool = False,
) -> list[Location]:
    """单次查找所有引用位置。

    Args:
        path: 文件路径。
        line: 行号（0-based）。
        character: 列号（0-based）。
        server: LSP server handle。
        include_declaration: 是否包含声明处，默认 False。

    Returns:
        Location 列表。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> refs = await lsp_references("src/main.py", line=3, character=4, server=server)
        >>> len(refs)
        5
    """
    try:
        result = await server.request(
            "textDocument/references",
            {
                **_text_doc_pos(path, line, character),
                "context": {"includeDeclaration": include_declaration},
            },
        )
    except Exception as e:
        raise LspOprimError("lsp_references failed", cause=e)

    return _parse_locations(result)


# ---------------------------------------------------------------------------
# lsp_document_symbols
# ---------------------------------------------------------------------------

async def lsp_document_symbols(
    path: str | Path,
    *,
    server: LspServerHandle,
) -> list[Symbol]:
    """单次获取文件中所有符号（函数/类/变量等）。

    Args:
        path: 文件路径。
        server: LSP server handle。

    Returns:
        Symbol 列表，按行号排序。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> syms = await lsp_document_symbols("src/main.py", server=server)
        >>> [s.name for s in syms if s.kind == 12]   # 12=Function
        ['parse', 'render', 'main']
    """
    try:
        result = await server.request(
            "textDocument/documentSymbol",
            {"textDocument": _text_doc(path)},
        )
    except Exception as e:
        raise LspOprimError(f"lsp_document_symbols failed for '{path}'", cause=e)

    return _parse_symbols(result or [], str(Path(path).resolve()))


# ---------------------------------------------------------------------------
# lsp_workspace_symbols
# ---------------------------------------------------------------------------

async def lsp_workspace_symbols(
    query: str,
    *,
    server: LspServerHandle,
) -> list[Symbol]:
    """单次在整个 workspace 搜索符号。

    Args:
        query: 搜索关键词（空字符串返回全部）。
        server: LSP server handle。

    Returns:
        Symbol 列表。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> syms = await lsp_workspace_symbols("parse", server=server)
    """
    try:
        result = await server.request(
            "workspace/symbol",
            {"query": query},
        )
    except Exception as e:
        raise LspOprimError("lsp_workspace_symbols failed", cause=e)

    return _parse_symbols(result or [], "")


# ---------------------------------------------------------------------------
# lsp_rename
# ---------------------------------------------------------------------------

async def lsp_rename(
    path: str | Path,
    *,
    line: int,
    character: int,
    new_name: str,
    server: LspServerHandle,
) -> WorkspaceEdit:
    """单次重命名符号，返回需要应用的 WorkspaceEdit。

    Args:
        path: 文件路径。
        line: 行号（0-based）。
        character: 列号（0-based）。
        new_name: 新名称。
        server: LSP server handle。

    Returns:
        WorkspaceEdit（含所有需要修改的文件和位置）。

    Raises:
        LspOprimError: LSP 请求失败或不支持重命名。

    Example:
        >>> edit = await lsp_rename("src/main.py", line=5, character=4,
        ...     new_name="process", server=server)
        >>> len(edit.changes)   # 受影响文件数
        3
    """
    try:
        result = await server.request(
            "textDocument/rename",
            {**_text_doc_pos(path, line, character), "newName": new_name},
        )
    except Exception as e:
        raise LspOprimError("lsp_rename failed", cause=e)

    if not result:
        return WorkspaceEdit()

    return _parse_workspace_edit(result)


# ---------------------------------------------------------------------------
# lsp_completion
# ---------------------------------------------------------------------------

async def lsp_completion(
    path: str | Path,
    *,
    line: int,
    character: int,
    server: LspServerHandle,
    trigger_character: str | None = None,
) -> list[Completion]:
    """单次获取补全候选列表。

    Args:
        path: 文件路径。
        line: 行号（0-based）。
        character: 列号（0-based）。
        server: LSP server handle。
        trigger_character: 触发补全的字符（如 "." "("），None 为手动触发。

    Returns:
        Completion 列表（最多 50 个）。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> completions = await lsp_completion("src/main.py",
        ...     line=10, character=8, server=server)
        >>> completions[0].label
        'parse_args'
    """
    context: dict[str, Any] = {"triggerKind": 2 if trigger_character else 1}
    if trigger_character:
        context["triggerCharacter"] = trigger_character

    try:
        result = await server.request(
            "textDocument/completion",
            {**_text_doc_pos(path, line, character), "context": context},
        )
    except Exception as e:
        raise LspOprimError("lsp_completion failed", cause=e)

    items = []
    if isinstance(result, dict):
        items = result.get("items", [])
    elif isinstance(result, list):
        items = result

    completions = []
    for item in items[:50]:  # 限制最多50个
        doc = item.get("documentation", "")
        if isinstance(doc, dict):
            doc = doc.get("value", "")
        completions.append(Completion(
            label=item.get("label", ""),
            kind=item.get("kind"),
            detail=item.get("detail", ""),
            documentation=str(doc),
            insert_text=item.get("insertText", item.get("label", "")),
        ))
    return completions


# ---------------------------------------------------------------------------
# lsp_format
# ---------------------------------------------------------------------------

async def lsp_format(
    path: str | Path,
    *,
    server: LspServerHandle,
    tab_size: int = 4,
    insert_spaces: bool = True,
) -> list[TextEdit]:
    """单次格式化整个文件，返回 TextEdit 列表。

    Args:
        path: 文件路径。
        server: LSP server handle。
        tab_size: Tab 大小，默认 4。
        insert_spaces: True 使用空格，False 使用 Tab。

    Returns:
        TextEdit 列表（空列表表示文件已格式化，无需修改）。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> edits = await lsp_format("src/main.py", server=server)
        >>> len(edits)
        3
    """
    try:
        result = await server.request(
            "textDocument/formatting",
            {
                "textDocument": _text_doc(path),
                "options": {
                    "tabSize": tab_size,
                    "insertSpaces": insert_spaces,
                },
            },
        )
    except Exception as e:
        raise LspOprimError(f"lsp_format failed for '{path}'", cause=e)

    return _parse_text_edits(result or [])


# ---------------------------------------------------------------------------
# lsp_code_action
# ---------------------------------------------------------------------------

async def lsp_code_action(
    path: str | Path,
    *,
    start_line: int,
    start_character: int,
    end_line: int,
    end_character: int,
    server: LspServerHandle,
    only_kinds: list[str] | None = None,
) -> list[CodeAction]:
    """单次获取指定范围的代码操作（quick fix / refactor 等）。

    Args:
        path: 文件路径。
        start_line: 范围起始行（0-based）。
        start_character: 范围起始列。
        end_line: 范围结束行。
        end_character: 范围结束列。
        server: LSP server handle。
        only_kinds: 过滤的 CodeAction kind 列表，None 表示全部返回。

    Returns:
        CodeAction 列表。

    Raises:
        LspOprimError: LSP 请求失败。

    Example:
        >>> actions = await lsp_code_action("src/main.py",
        ...     start_line=5, start_character=0,
        ...     end_line=5, end_character=10, server=server)
        >>> [a.title for a in actions]
        ['Import os', 'Add type annotation']
    """
    params: dict[str, Any] = {
        "textDocument": _text_doc(path),
        "range": {
            "start": _position(start_line, start_character),
            "end": _position(end_line, end_character),
        },
        "context": {"diagnostics": []},
    }
    if only_kinds:
        params["context"]["only"] = only_kinds

    try:
        result = await server.request("textDocument/codeAction", params)
    except Exception as e:
        raise LspOprimError("lsp_code_action failed", cause=e)

    actions = []
    for item in (result or []):
        if isinstance(item, dict):
            actions.append(CodeAction(
                title=item.get("title", ""),
                kind=item.get("kind", ""),
                edit=_parse_workspace_edit(item["edit"]) if "edit" in item else None,
                command=item.get("command"),
            ))
    return actions


# ---------------------------------------------------------------------------
# 内部解析工具
# ---------------------------------------------------------------------------

def _parse_locations(result: Any) -> list[Location]:
    """解析 LSP definition/references 响应为 Location 列表。"""
    if not result:
        return []
    if isinstance(result, dict):
        result = [result]
    locs = []
    for item in result:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri", "")
        path = uri.replace("file://", "") if uri.startswith("file://") else uri
        r = item.get("range", {})
        start = r.get("start", {})
        end = r.get("end", {})
        locs.append(Location(
            path=path,
            start_line=start.get("line", 0),
            start_character=start.get("character", 0),
            end_line=end.get("line", 0),
            end_character=end.get("character", 0),
        ))
    return locs


def _parse_symbols(result: list, default_path: str) -> list[Symbol]:
    """解析 LSP documentSymbol / workspaceSymbol 响应。"""
    syms = []
    for item in result:
        if not isinstance(item, dict):
            continue
        loc = item.get("location", {})
        uri = loc.get("uri", "") if loc else ""
        path = uri.replace("file://", "") if uri.startswith("file://") else (default_path or uri)
        r = (loc.get("range", {}) if loc else {}) or item.get("range", {}) or item.get("selectionRange", {})
        start = r.get("start", {})
        end = r.get("end", {})
        syms.append(Symbol(
            name=item.get("name", ""),
            kind=item.get("kind", 0),
            path=path,
            start_line=start.get("line", 0),
            start_character=start.get("character", 0),
            end_line=end.get("line", 0),
            end_character=end.get("character", 0),
            container=item.get("containerName", ""),
        ))
    return sorted(syms, key=lambda s: (s.path, s.start_line))


def _parse_text_edits(result: list) -> list[TextEdit]:
    """解析 TextEdit 列表。"""
    edits = []
    for item in result:
        if not isinstance(item, dict):
            continue  # pragma: no cover
        r = item.get("range", {})
        start = r.get("start", {})
        end = r.get("end", {})
        edits.append(TextEdit(
            start_line=start.get("line", 0),
            start_character=start.get("character", 0),
            end_line=end.get("line", 0),
            end_character=end.get("character", 0),
            new_text=item.get("newText", ""),
        ))
    return edits


def _parse_workspace_edit(result: dict) -> WorkspaceEdit:
    """解析 WorkspaceEdit。"""
    we = WorkspaceEdit()
    raw_changes = result.get("changes", {})
    for uri, edits in raw_changes.items():
        path = uri.replace("file://", "") if uri.startswith("file://") else uri
        we.changes[path] = _parse_text_edits(edits)

    # documentChanges 格式
    for dc in result.get("documentChanges", []):
        if not isinstance(dc, dict):
            continue
        uri = dc.get("textDocument", {}).get("uri", "")
        path = uri.replace("file://", "") if uri.startswith("file://") else uri
        if path and "edits" in dc:
            we.changes[path] = _parse_text_edits(dc["edits"])

    return we