from __future__ import annotations

import pytest

from oprim import ast_parse, ast_references, ast_symbols


def test_ast_parse_symbols_and_references_are_syntactic():
    tree = ast_parse(source="import os\ndef f(x):\n    return x + x\n")
    symbols = ast_symbols(tree=tree)
    refs = ast_references(tree=tree, symbol="x")

    assert {item["name"] for item in symbols} >= {"os", "f"}
    assert len(refs) == 2
    assert all(item["context"] == "load" for item in refs)


def test_ast_primitives_accept_source_and_reject_unknown_language():
    assert ast_symbols(tree="value = 1")
    with pytest.raises(ValueError, match="unsupported AST language"):
        ast_parse(source="x", language="typescript")


def test_ast_references_validate_symbol():
    with pytest.raises(ValueError, match="non-empty"):
        ast_references(tree="x = 1", symbol="")
