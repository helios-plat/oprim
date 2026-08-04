"""Tests for Veya 顶级 Agent 增强元素 — oprim 层 (3 个).

Covers: ast_extract_symbols, hitl_wait_approval, mcp_register_tool.
"""

from __future__ import annotations

import pathlib

import pytest

# ---------------------------------------------------------------------------
# ast_extract_symbols
# ---------------------------------------------------------------------------


class TestAstExtractSymbols:
    async def test_flat_symbol_table(self, tmp_path: pathlib.Path):
        from oprim import ast_extract_symbols

        p = tmp_path / "mod.py"
        p.write_text(
            "import os\nfrom pathlib import Path\n\n"
            "async def foo(a, b=1):\n    return a\n\n"
            "class Bar:\n    def baz(self): pass\n"
        )
        r = await ast_extract_symbols(p)
        assert r["status"] == "ok" and r["language"] == "python"
        by_name = {s["name"]: s for s in r["symbols"]}
        assert by_name["foo"]["kind"] == "function"
        assert by_name["foo"]["signature"] == "async def foo(a, b)"
        assert by_name["Bar"]["kind"] == "class"
        assert by_name["baz"]["kind"] == "method"
        assert by_name["os"]["kind"] == "import"
        assert by_name["Path"]["kind"] == "import"

    async def test_constants_optional(self, tmp_path: pathlib.Path):
        from oprim import ast_extract_symbols

        p = tmp_path / "c.py"
        p.write_text("MAX_N = 10\nMIN_N = 1\nx = 2\n")
        r = await ast_extract_symbols(p, include_constants=True)
        names = {s["name"] for s in r["symbols"]}
        assert {"MAX_N", "MIN_N"} <= names and "x" not in names

    async def test_missing_file(self, tmp_path: pathlib.Path):
        from oprim import ast_extract_symbols

        with pytest.raises(Exception, match="not found"):
            await ast_extract_symbols(tmp_path / "nope.py")

    async def test_syntax_error(self, tmp_path: pathlib.Path):
        from oprim import ast_extract_symbols

        p = tmp_path / "bad.py"
        p.write_text("def broken(:\n")
        with pytest.raises(Exception, match="syntax error"):
            await ast_extract_symbols(p)


# ---------------------------------------------------------------------------
# hitl_wait_approval
# ---------------------------------------------------------------------------


class FakeBus:
    def __init__(self):
        self.decisions = {}

    def set_decision(self, request_id, outcome, note=""):
        self.decisions[request_id] = {"outcome": outcome, "note": note}

    async def wait_for_decision(self, request_id, *, timeout=60.0):
        import asyncio

        d = self.decisions.get(request_id)
        if d is None:
            await asyncio.sleep(timeout)  # pragma: no cover - 不应走到
            return {"outcome": "timed_out"}
        return {"request_id": request_id, **d}


class TestHitlWaitApproval:
    async def test_approved(self):
        from oprim import hitl_wait_approval

        bus = FakeBus()
        bus.set_decision("r1", "approved", note="ok")
        d = await hitl_wait_approval("r1", bus=bus, timeout=1.0)
        assert d["status"] == "ok" and d["outcome"] == "approved"
        assert d["note"] == "ok"

    async def test_rejected(self):
        from oprim import hitl_wait_approval

        bus = FakeBus()
        bus.set_decision("r2", "rejected")
        d = await hitl_wait_approval("r2", bus=bus)
        assert d["outcome"] == "rejected"

    async def test_timed_out(self):
        from oprim import hitl_wait_approval

        class TimeoutBus:
            async def wait_for_decision(self, request_id, *, timeout=60.0):
                return {"request_id": request_id, "outcome": "timed_out", "note": ""}

        d = await hitl_wait_approval("r3", bus=TimeoutBus(), timeout=0.01)
        assert d["outcome"] == "timed_out"

    async def test_no_bus(self):
        from oprim import hitl_wait_approval

        with pytest.raises(Exception, match="bus"):
            await hitl_wait_approval("r4", bus=None)

    async def test_validation(self):
        from oprim import OprimValidationError, hitl_wait_approval

        with pytest.raises(OprimValidationError):
            await hitl_wait_approval("", bus=FakeBus())
        with pytest.raises(OprimValidationError):
            await hitl_wait_approval("r", bus=FakeBus(), timeout=0)


# ---------------------------------------------------------------------------
# mcp_register_tool
# ---------------------------------------------------------------------------


class TestMcpRegisterTool:
    async def test_register_via_obase_server(self):
        from obase.mcp_server import MCPServer

        from oprim import mcp_register_tool

        server = MCPServer(name="veya", version="1.0.0")

        async def handler(args):
            return {"echo": args.get("text")}

        r = mcp_register_tool(
            server,
            name="veya.echo",
            description="Echo text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=handler,
        )
        assert r["registered"] is True and r["tool_name"] == "veya.echo"
        assert server.has_tool("veya.echo")
        assert server.list_tools()[0]["description"] == "Echo text"

    async def test_duplicate_rejected(self):
        from obase.mcp_server import MCPServer

        from oprim import McpRegisterError, mcp_register_tool

        server = MCPServer(name="d", version="1")

        async def h(args):
            return {}

        mcp_register_tool(
            server, name="d.t", description="t",
            input_schema={"type": "object", "properties": {}}, handler=h,
        )
        with pytest.raises(McpRegisterError):
            mcp_register_tool(
                server, name="d.t", description="t",
                input_schema={"type": "object", "properties": {}}, handler=h,
            )

    async def test_validation(self):
        from oprim import OprimValidationError, mcp_register_tool

        with pytest.raises(OprimValidationError):
            mcp_register_tool(
                None, name="", description="d",
                input_schema={"type": "object", "properties": {}}, handler=lambda a: a,
            )
        with pytest.raises(OprimValidationError):
            mcp_register_tool(
                object(), name="x", description="d",
                input_schema={"not": "a schema"}, handler=lambda a: a,
            )

    async def test_incompatible_server(self):
        from oprim import McpRegisterError, mcp_register_tool

        class NoRegister:
            pass

        with pytest.raises(McpRegisterError):
            mcp_register_tool(
                NoRegister(), name="x", description="d",
                input_schema={"type": "object", "properties": {}}, handler=lambda a: a,
            )
