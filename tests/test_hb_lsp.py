"""Tests — H-B D组: LSP IO 扩展 (12 functions)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from oprim.lsp import (
    CallItem,
    Diagnostic,
    Hover,
    Location,
    Pos,
    Symbol,
    diagnostics_to_summary,
    location_to_snippet,
    lsp_diagnostics,
    lsp_document_symbol,
    lsp_find_references,
    lsp_goto_definition,
    lsp_goto_implementation,
    lsp_hover,
    lsp_incoming_calls,
    lsp_outgoing_calls,
    lsp_prepare_call_hierarchy,
    lsp_workspace_symbol,
)


# ---------------------------------------------------------------------------
# Mock LSP server handle
# ---------------------------------------------------------------------------

def _mock_server(responses: dict[str, Any]) -> Any:
    server = AsyncMock()
    async def _request(method: str, params: dict) -> Any:
        return responses.get(method)
    server.request = _request
    server.root_uri = "file:///project"
    return server


def _loc(path: str = "/p/f.py", sl: int = 1, sc: int = 0, el: int = 1, ec: int = 5) -> dict:
    return {"uri": f"file://{path}", "range": {"start": {"line": sl, "character": sc}, "end": {"line": el, "character": ec}}}


def _call_item_raw(name: str = "fn") -> dict:
    return {
        "name": name, "kind": 12, "uri": "file:///p/f.py",
        "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}},
        "selectionRange": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 10}},
    }


# ---------------------------------------------------------------------------
# lsp_goto_definition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_goto_definition_single() -> None:
    server = _mock_server({"textDocument/definition": [_loc()]})
    locs = await lsp_goto_definition("/p/f.py", pos=(5, 10), lsp=server)
    assert len(locs) == 1
    assert locs[0].path == "/p/f.py"


@pytest.mark.asyncio
async def test_lsp_goto_definition_empty() -> None:
    server = _mock_server({"textDocument/definition": []})
    locs = await lsp_goto_definition("/p/f.py", pos=(0, 0), lsp=server)
    assert locs == []


@pytest.mark.asyncio
async def test_lsp_goto_definition_multi() -> None:
    server = _mock_server({"textDocument/definition": [_loc(), _loc("/p/g.py")]})
    locs = await lsp_goto_definition("/p/f.py", pos=(1, 1), lsp=server)
    assert len(locs) == 2


# ---------------------------------------------------------------------------
# lsp_find_references
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_find_references_multiple() -> None:
    server = _mock_server({"textDocument/references": [_loc(), _loc(), _loc()]})
    refs = await lsp_find_references("/p/f.py", pos=(3, 4), lsp=server)
    assert len(refs) == 3


@pytest.mark.asyncio
async def test_lsp_find_references_empty() -> None:
    server = _mock_server({"textDocument/references": None})
    refs = await lsp_find_references("/p/f.py", pos=(0, 0), lsp=server)
    assert refs == []


# ---------------------------------------------------------------------------
# lsp_hover (updated signature)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_hover_with_pos() -> None:
    server = _mock_server({"textDocument/hover": {"contents": {"kind": "markdown", "value": "**int**"}}})
    h = await lsp_hover("/p/f.py", pos=(10, 5), lsp=server)
    assert h is not None
    assert "int" in h.contents


@pytest.mark.asyncio
async def test_lsp_hover_none_result() -> None:
    server = _mock_server({"textDocument/hover": None})
    h = await lsp_hover("/p/f.py", pos=(0, 0), lsp=server)
    assert h is None


@pytest.mark.asyncio
async def test_lsp_hover_legacy_line_character() -> None:
    server = _mock_server({"textDocument/hover": {"contents": "doc"}})
    h = await lsp_hover("/p/f.py", line=5, character=3, server=server)
    assert h is not None


@pytest.mark.asyncio
async def test_lsp_hover_no_server_raises() -> None:
    with pytest.raises(ValueError, match="server or lsp"):
        await lsp_hover("/p/f.py", pos=(0, 0))


# ---------------------------------------------------------------------------
# lsp_document_symbol
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_document_symbol_basic() -> None:
    server = _mock_server({"textDocument/documentSymbol": [
        {"name": "MyClass", "kind": 5, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 10, "character": 0}}},
    ]})
    syms = await lsp_document_symbol("/p/f.py", lsp=server)
    assert len(syms) == 1
    assert syms[0].name == "MyClass"


@pytest.mark.asyncio
async def test_lsp_document_symbol_empty() -> None:
    server = _mock_server({"textDocument/documentSymbol": []})
    syms = await lsp_document_symbol("/p/f.py", lsp=server)
    assert syms == []


# ---------------------------------------------------------------------------
# lsp_workspace_symbol
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_workspace_symbol_query() -> None:
    server = _mock_server({"workspace/symbol": [
        {"name": "parse", "kind": 12, "location": _loc()},
    ]})
    syms = await lsp_workspace_symbol(query="parse", lsp=server)
    assert syms[0].name == "parse"


@pytest.mark.asyncio
async def test_lsp_workspace_symbol_empty() -> None:
    server = _mock_server({"workspace/symbol": None})
    syms = await lsp_workspace_symbol(query="nonexistent", lsp=server)
    assert syms == []


# ---------------------------------------------------------------------------
# lsp_goto_implementation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_goto_implementation_single() -> None:
    server = _mock_server({"textDocument/implementation": [_loc()]})
    locs = await lsp_goto_implementation("/p/f.py", pos=(2, 5), lsp=server)
    assert len(locs) == 1


@pytest.mark.asyncio
async def test_lsp_goto_implementation_none() -> None:
    server = _mock_server({"textDocument/implementation": None})
    locs = await lsp_goto_implementation("/p/f.py", pos=(0, 0), lsp=server)
    assert locs == []


# ---------------------------------------------------------------------------
# lsp_prepare_call_hierarchy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_prepare_call_hierarchy_ok() -> None:
    server = _mock_server({"textDocument/prepareCallHierarchy": [_call_item_raw("myFunc")]})
    item = await lsp_prepare_call_hierarchy("/p/f.py", pos=(10, 5), lsp=server)
    assert item is not None
    assert item.name == "myFunc"


@pytest.mark.asyncio
async def test_lsp_prepare_call_hierarchy_empty() -> None:
    server = _mock_server({"textDocument/prepareCallHierarchy": None})
    item = await lsp_prepare_call_hierarchy("/p/f.py", pos=(0, 0), lsp=server)
    assert item is None


# ---------------------------------------------------------------------------
# lsp_incoming_calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_incoming_calls_multiple() -> None:
    server = _mock_server({"callHierarchy/incomingCalls": [
        {"from": _call_item_raw("caller1")},
        {"from": _call_item_raw("caller2")},
    ]})
    item = CallItem(name="fn", kind=12, uri="file:///p/f.py",
                    range_start_line=0, range_start_char=0, range_end_line=0, range_end_char=0)
    callers = await lsp_incoming_calls(item, lsp=server)
    assert len(callers) == 2
    assert callers[0].name == "caller1"


@pytest.mark.asyncio
async def test_lsp_incoming_calls_none() -> None:
    server = _mock_server({"callHierarchy/incomingCalls": None})
    item = CallItem(name="fn", kind=12, uri="file:///p/f.py",
                    range_start_line=0, range_start_char=0, range_end_line=0, range_end_char=0)
    callers = await lsp_incoming_calls(item, lsp=server)
    assert callers == []


# ---------------------------------------------------------------------------
# lsp_outgoing_calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_outgoing_calls_multiple() -> None:
    server = _mock_server({"callHierarchy/outgoingCalls": [
        {"to": _call_item_raw("callee1")},
    ]})
    item = CallItem(name="fn", kind=12, uri="file:///p/f.py",
                    range_start_line=0, range_start_char=0, range_end_line=0, range_end_char=0)
    callees = await lsp_outgoing_calls(item, lsp=server)
    assert callees[0].name == "callee1"


# ---------------------------------------------------------------------------
# lsp_diagnostics (updated with lsp: param)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lsp_diagnostics_lsp_param() -> None:
    server = _mock_server({"textDocument/diagnostic": {
        "items": [{"severity": 1, "message": "err", "source": "pyright", "range": {
            "start": {"line": 3, "character": 0}, "end": {"line": 3, "character": 5}
        }}]
    }})
    diags = await lsp_diagnostics("/p/f.py", lsp=server)
    assert len(diags) == 1
    assert diags[0].severity == 1
    assert diags[0].message == "err"


@pytest.mark.asyncio
async def test_lsp_diagnostics_empty() -> None:
    server = _mock_server({"textDocument/diagnostic": {"items": []}})
    diags = await lsp_diagnostics("/p/f.py", lsp=server)
    assert diags == []


@pytest.mark.asyncio
async def test_lsp_diagnostics_no_server_raises() -> None:
    with pytest.raises(ValueError):
        await lsp_diagnostics("/p/f.py")


# ---------------------------------------------------------------------------
# diagnostics_to_summary  [s] 纯计算
# ---------------------------------------------------------------------------

def test_diagnostics_to_summary_empty() -> None:
    assert diagnostics_to_summary([]) == "No diagnostics."


def test_diagnostics_to_summary_errors_warnings() -> None:
    diags = [
        Diagnostic(path="/p/f.py", line=0, character=0, end_line=0, end_character=5,
                   severity=1, message="undefined foo", source="pyright"),
        Diagnostic(path="/p/f.py", line=5, character=2, end_line=5, end_character=8,
                   severity=2, message="unused import", source="ruff"),
    ]
    summary = diagnostics_to_summary(diags)
    assert "Errors (1)" in summary
    assert "Warnings (1)" in summary
    assert "undefined foo" in summary
    assert "unused import" in summary


def test_diagnostics_to_summary_line_numbers() -> None:
    diags = [
        Diagnostic(path="/p/f.py", line=9, character=3, end_line=9, end_character=6,
                   severity=1, message="bad", source="x"),
    ]
    summary = diagnostics_to_summary(diags)
    assert ":10:" in summary  # 0-based line 9 → display as 10


def test_diagnostics_to_summary_grouping() -> None:
    diags = [
        Diagnostic(path="/a.py", line=0, character=0, end_line=0, end_character=1,
                   severity=3, message="info msg", source=""),
        Diagnostic(path="/b.py", line=0, character=0, end_line=0, end_character=1,
                   severity=4, message="hint msg", source=""),
    ]
    summary = diagnostics_to_summary(diags)
    assert "Info (1)" in summary
    assert "Hints (1)" in summary


# ---------------------------------------------------------------------------
# location_to_snippet
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_location_to_snippet_normal(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("line0\nline1\nline2\nline3\nline4\n")
    loc = Location(path=str(src), start_line=2, start_character=0, end_line=2, end_character=5)
    snippet = await location_to_snippet(loc, ctx=1)
    assert "line1" in snippet
    assert "line2" in snippet
    assert "line3" in snippet


@pytest.mark.asyncio
async def test_location_to_snippet_file_start(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("first\nsecond\nthird\n")
    loc = Location(path=str(src), start_line=0, start_character=0, end_line=0, end_character=5)
    snippet = await location_to_snippet(loc, ctx=2)
    assert "first" in snippet


@pytest.mark.asyncio
async def test_location_to_snippet_file_end(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("a\nb\nc\nd\ne\n")
    loc = Location(path=str(src), start_line=4, start_character=0, end_line=4, end_character=1)
    snippet = await location_to_snippet(loc, ctx=1)
    assert "e" in snippet


@pytest.mark.asyncio
async def test_location_to_snippet_not_found(tmp_path: Path) -> None:
    loc = Location(path=str(tmp_path / "missing.py"), start_line=0,
                   start_character=0, end_line=0, end_character=1)
    with pytest.raises(FileNotFoundError):
        await location_to_snippet(loc)


@pytest.mark.asyncio
async def test_location_to_snippet_ctx_zero(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("alpha\nbeta\ngamma\n")
    loc = Location(path=str(src), start_line=1, start_character=0, end_line=1, end_character=4)
    snippet = await location_to_snippet(loc, ctx=0)
    assert "beta" in snippet
