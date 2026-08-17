"""jailed_file_read / sandboxed_exec / hash_tool_call."""

from __future__ import annotations

import sys

import pytest
from obase.sandbox import PathJail, ProcessJail

from oprim._hash_tool_call import hash_tool_call
from oprim._jailed_file_read import jailed_file_read
from oprim._sandboxed_exec import sandboxed_exec


def test_jailed_file_read(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    text = jailed_file_read(PathJail(tmp_path), file_path="note.txt")
    assert text == "hello\n"


def test_jailed_file_read_blocks_escape(tmp_path) -> None:
    with pytest.raises(PermissionError):
        jailed_file_read(PathJail(tmp_path), file_path="../x")


def test_sandboxed_exec(tmp_path) -> None:
    jail = ProcessJail(tmp_path)
    rec = sandboxed_exec(jail, argv=[sys.executable, "-c", "print(2+2)"])
    assert rec["ok"] is True
    assert rec["exit_code"] == 0
    assert "4" in rec["stdout"]


def test_hash_tool_call_is_stable() -> None:
    a = hash_tool_call("read", arguments={"path": "a.py", "n": 1})
    b = hash_tool_call("read", arguments={"n": 1, "path": "a.py"})
    c = hash_tool_call("write", arguments={"path": "a.py", "n": 1})
    assert a == b
    assert a != c
    assert len(a) == 64
