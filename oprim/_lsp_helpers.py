"""LSP helper functions used across LSP atomic operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _text_doc(path: str | Path) -> dict[str, str]:
    """Build a textDocument identifier for LSP requests."""
    return {"uri": Path(path).resolve().as_uri()}


def _text_doc_pos(path: str | Path, line: int, character: int) -> dict[str, Any]:
    """Build a textDocument + position payload for LSP requests."""
    return {
        "textDocument": _text_doc(path),
        "position": {"line": line, "character": character},
    }


def _position(line: int, character: int) -> dict[str, int]:
    """Build a position object."""
    return {"line": line, "character": character}


def _parse_locations(result: Any) -> list[Any]:
    """Parse LSP Location[] result into Location objects."""
    from oprim._lsp_definition import Location

    if not result:
        return []
    if isinstance(result, dict):
        result = [result]
    locations = []
    for item in result:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri", "")
        rng = item.get("range", {})
        start = rng.get("start", {})
        end = rng.get("end", {})
        locations.append(
            Location(
                path=uri,
                start_line=start.get("line", 0),
                start_character=start.get("character", 0),
                end_line=end.get("line", 0),
                end_character=end.get("character", 0),
            )
        )
    return locations


def _parse_symbols(result: Any, path: str) -> list[Any]:
    """Parse LSP SymbolInformation[] result into Symbol objects."""
    from oprim._lsp_document_symbols import Symbol

    if not result:
        return []
    if isinstance(result, dict):
        result = [result]
    symbols = []
    for item in result:
        if not isinstance(item, dict):
            continue
        loc = item.get("location", {})
        rng = loc.get("range", {})
        start = rng.get("start", {})
        end = rng.get("end", {})
        symbols.append(
            Symbol(
                name=item.get("name", ""),
                kind=item.get("kind", 0),
                path=path,
                start_line=start.get("line", 0),
                start_character=start.get("character", 0),
                end_line=end.get("line", 0),
                end_character=end.get("character", 0),
                container=item.get("containerName", ""),
            )
        )
    return symbols


def _parse_text_edits(result: Any) -> list[Any]:
    """Parse LSP TextEdit[] result into TextEdit objects."""
    from oprim._lsp_format import TextEdit

    if not result:
        return []
    if isinstance(result, dict):
        result = [result]
    edits = []
    for item in result:
        if not isinstance(item, dict):
            continue
        rng = item.get("range", {})
        start = rng.get("start", {})
        end = rng.get("end", {})
        edits.append(
            TextEdit(
                start_line=start.get("line", 0),
                start_character=start.get("character", 0),
                end_line=end.get("line", 0),
                end_character=end.get("character", 0),
                new_text=item.get("newText", ""),
            )
        )
    return edits


def _parse_workspace_edit(result: Any) -> Any:
    """Parse LSP WorkspaceEdit result into a WorkspaceEdit object."""
    from oprim._lsp_rename import TextEdit, WorkspaceEdit

    if not result:
        return WorkspaceEdit()
    if not isinstance(result, dict):
        return WorkspaceEdit()

    changes: dict[str, list[TextEdit]] = {}
    for uri, edits in result.get("changes", {}).items():
        parsed: list[TextEdit] = []
        for e in edits:
            if not isinstance(e, dict):
                continue
            rng = e.get("range", {})
            start = rng.get("start", {})
            end = rng.get("end", {})
            parsed.append(
                TextEdit(
                    start_line=start.get("line", 0),
                    start_character=start.get("character", 0),
                    end_line=end.get("line", 0),
                    end_character=end.get("character", 0),
                    new_text=e.get("newText", ""),
                )
            )
        changes[uri] = parsed
    return WorkspaceEdit(changes=changes)
