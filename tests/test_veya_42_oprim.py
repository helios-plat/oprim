"""Tests for the Veya/3O 42-element inventory — oprim layer (14 atoms).

Covers: sandbox_exec, git_worktree_create/remove, llm_chat_call, embedding_gen,
vector_search, browser_fetch_dom, browser_click_element, file_write_atomic,
file_read_slice, mcp_call_tool, image_analyze, ast_parse_code, docker_image_build.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest


# ---------------------------------------------------------------------------
# 公共 fakes
# ---------------------------------------------------------------------------


class FakeLLMCaller:
    def __init__(self, text="hi", usage=None):
        self.text = text
        self.usage = usage or {"input_tokens": 5, "output_tokens": 3}

    async def __call__(self, *, messages, tools=None, max_tokens=4096, system=None):
        return {
            "content": [{"type": "text", "text": self.text}],
            "usage": self.usage,
            "stop_reason": "end_turn",
        }


class FakeEmbedCaller:
    async def __call__(self, *, text, model="x"):
        return [0.5, 0.5]


class FakeSearchStore:
    def __init__(self, hits):
        self.hits = hits

    async def search(self, vector, *, top_k=10):
        return self.hits


class FakePage:
    async def goto(self, url, *, timeout_ms=30_000):
        pass

    async def content(self):
        return "<html><head><title>T</title></head><body>ok</body></html>"

    async def click(self, selector, *, timeout_ms=10_000):
        pass

    async def screenshot(self, *, path=None):
        return b"PNG"


class FakeMcpClient:
    async def list_tools(self):
        return [{"name": "ping", "description": "ping tool"}]

    async def call_tool(self, name, args):
        return {"content": [{"type": "text", "text": "pong"}], "isError": False}


# ---------------------------------------------------------------------------
# file_write_atomic / file_read_slice
# ---------------------------------------------------------------------------


class TestFileAtoms:
    async def test_atomic_write_and_slice(self, tmp_path: pathlib.Path):
        from oprim import file_read_slice, file_write_atomic

        p = tmp_path / "a.txt"
        w = await file_write_atomic(p, content="l1\nl2\nl3\n", sandbox_root=tmp_path)
        assert w["bytes_written"] == 9
        r = await file_read_slice(p, max_lines=2)
        assert r["truncated"] and r["content"] == "l1\nl2\n"
        r2 = await file_read_slice(p, max_tokens=1)
        assert r2["truncated"] and r2["reason"] == "max_tokens"
        assert r2["total_lines"] == 3

    async def test_path_escape_blocked(self, tmp_path: pathlib.Path):
        from oprim import PathSecurityError, file_write_atomic

        with pytest.raises(PathSecurityError):
            await file_write_atomic("/etc/evil.txt", content="x", sandbox_root=tmp_path)

    async def test_overwrite_false(self, tmp_path: pathlib.Path):
        from oprim import FileOprimError, file_write_atomic

        p = tmp_path / "b.txt"
        await file_write_atomic(p, content="1")
        with pytest.raises(FileOprimError):
            await file_write_atomic(p, content="2", overwrite=False)

    async def test_read_missing_file(self, tmp_path: pathlib.Path):
        from oprim import FileOprimError, file_read_slice

        with pytest.raises(FileOprimError):
            await file_read_slice(tmp_path / "nope.txt", max_lines=10)

    async def test_no_limit_rejected(self, tmp_path: pathlib.Path):
        from oprim import OprimValidationError, file_read_slice

        (tmp_path / "c.txt").write_text("x")
        with pytest.raises(OprimValidationError):
            await file_read_slice(tmp_path / "c.txt")


# ---------------------------------------------------------------------------
# ast_parse_code
# ---------------------------------------------------------------------------


class TestAstParseCode:
    async def test_symbols_and_classes(self, tmp_path: pathlib.Path):
        from oprim import ast_parse_code

        p = tmp_path / "mod.py"
        p.write_text(
            "import os\nfrom pathlib import Path\n\n"
            "async def foo(a, b=1):\n    return a\n\n"
            "class Bar:\n    def baz(self): pass\n"
            "MAX_N = 10\n"
        )
        r = await ast_parse_code(p, include_constants=True)
        assert r["language"] == "python"
        assert r["functions"][0]["name"] == "foo"
        assert r["functions"][0]["is_async"] is True
        assert r["classes"][0]["name"] == "Bar"
        assert r["classes"][0]["methods"][0]["name"] == "baz"
        assert len(r["imports"]) == 2
        assert any(c["name"] == "MAX_N" for c in r["constants"])
        assert r["symbols"] >= 5

    async def test_syntax_error(self, tmp_path: pathlib.Path):
        from oprim import ast_parse_code

        p = tmp_path / "bad.py"
        p.write_text("def broken(:\n")
        with pytest.raises(Exception, match="syntax error"):
            await ast_parse_code(p)


# ---------------------------------------------------------------------------
# llm_chat_call / embedding_gen / vector_search
# ---------------------------------------------------------------------------


class TestLlmEmbeddingVector:
    async def test_llm_chat_call_standardized(self):
        from oprim import llm_chat_call

        resp = await llm_chat_call(
            [{"role": "user", "content": "hi"}], caller=FakeLLMCaller()
        )
        assert resp["text"] == "hi"
        assert resp["usage"] == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        assert resp["stop_reason"] == "end_turn"
        assert resp["cost_usd"] >= 0

    async def test_llm_chat_call_empty_messages(self):
        from oprim import LLMOprimError, llm_chat_call

        with pytest.raises(LLMOprimError):
            await llm_chat_call([], caller=FakeLLMCaller())

    async def test_embedding_gen(self):
        from oprim import embedding_gen

        r = await embedding_gen("hello", caller=FakeEmbedCaller(), model="m")
        assert r["dim"] == 2 and r["vector"] == [0.5, 0.5]

    async def test_vector_search_positional_store(self):
        from oprim import vector_search

        class C:
            chunk_id = "c1"
            content = "x"
            path = None

        hits = await vector_search(
            [1.0, 0.0], store=FakeSearchStore([(C(), 0.1)])
        )
        assert hits[0]["chunk_id"] == "c1" and abs(hits[0]["score"] - 0.1) < 1e-9

    async def test_vector_search_filter_store(self):
        from oprim import vector_search

        class Store:
            async def search(self, *, vector, top_k=5, filter=None):
                assert filter == {"k": 1}
                return [{"chunk_id": "c2", "content": "y", "score": 0.7}]

        hits = await vector_search([1.0], store=Store(), filter={"k": 1})
        assert hits[0]["chunk_id"] == "c2"

    async def test_vector_search_validation(self):
        from oprim import OprimValidationError, vector_search

        with pytest.raises(OprimValidationError):
            await vector_search([], store=FakeSearchStore([]))


# ---------------------------------------------------------------------------
# browser_fetch_dom / browser_click_element
# ---------------------------------------------------------------------------


class TestBrowserAtoms:
    async def test_fetch_dom(self):
        from oprim import browser_fetch_dom

        r = await browser_fetch_dom("http://x", browser=FakePage())
        assert r["title"] == "T" and "ok" in r["html"]

    async def test_fetch_dom_screenshot(self):
        from oprim import browser_fetch_dom

        r = await browser_fetch_dom("http://x", browser=FakePage(), screenshot=True)
        assert r["screenshot_bytes"] == b"PNG"

    async def test_click(self):
        from oprim import browser_click_element

        r = await browser_click_element("#btn", browser=FakePage())
        assert r["clicked"] is True

    async def test_fetch_validation(self):
        from oprim import OprimValidationError, browser_fetch_dom

        with pytest.raises(OprimValidationError):
            await browser_fetch_dom("", browser=FakePage())


# ---------------------------------------------------------------------------
# mcp_call_tool（已有元素复验）
# ---------------------------------------------------------------------------


class TestMcpCallTool:
    async def test_call(self):
        from oprim import mcp_call_tool

        r = await mcp_call_tool("ping", arguments={}, client=FakeMcpClient())
        assert r["content"][0]["text"] == "pong" and r["isError"] is False


# ---------------------------------------------------------------------------
# sandbox_exec / docker_image_build — 依赖真实 docker，验证优雅错误
# ---------------------------------------------------------------------------


class TestDockerDependentAtoms:
    async def test_sandbox_exec_missing_docker(self):
        import importlib.util

        if importlib.util.find_spec("docker") is not None:
            pytest.skip("docker SDK installed — 该路径在无 SDK 环境验证")
        from oprim import OprimError, sandbox_exec

        with pytest.raises(OprimError):
            await sandbox_exec("echo hi", container_id="nope")

    async def test_docker_image_build_missing_docker(self):
        import importlib.util

        if importlib.util.find_spec("docker") is not None:
            pytest.skip("docker SDK installed — 该路径在无 SDK 环境验证")
        from oprim import DockerImageBuildError, docker_image_build

        with pytest.raises(DockerImageBuildError):
            await docker_image_build("/tmp", tag="x:1")

    async def test_docker_image_build_bad_context(self):
        from oprim import DockerImageBuildError, docker_image_build

        with pytest.raises(DockerImageBuildError):
            await docker_image_build("/nonexistent-dir-xyz", tag="x:1")


# ---------------------------------------------------------------------------
# image_analyze — 校验路径（VLM 依赖 provider，验证参数守卫）
# ---------------------------------------------------------------------------


class TestImageAnalyze:
    async def test_validation(self, tmp_path: pathlib.Path):
        from oprim import OprimValidationError, image_analyze

        img = tmp_path / "a.png"
        img.write_bytes(b"png")
        with pytest.raises(OprimValidationError):
            await image_analyze(str(img), prompt="", provider="qwen_vl")
        with pytest.raises(OprimValidationError):
            await image_analyze(str(img), prompt="x", provider="")


# ---------------------------------------------------------------------------
# git_worktree_create / remove — 真实 git 集成
# ---------------------------------------------------------------------------


class TestGitWorktreeAtoms:
    async def test_create_remove_roundtrip(self, tmp_path: pathlib.Path):
        from oprim import git_worktree_create, git_worktree_remove
        from oprim.git import _git

        repo = tmp_path / "repo"
        repo.mkdir()
        _git("init", "-b", "main", repo=repo)
        (repo / "f.txt").write_text("x")
        _git("add", ".", repo=repo)
        _git("commit", "-m", "init", repo=repo)

        wt = await git_worktree_create("feat/x", repo=repo)
        assert wt.exists()
        # git_worktree_remove 为同步原子（oprim.worktree 既有实现，本性 sync）
        git_worktree_remove(str(wt), repo=repo)
        assert not wt.exists()

    async def test_create_empty_branch_rejected(self, tmp_path: pathlib.Path):
        from oprim import GitOprimError, git_worktree_create

        with pytest.raises(GitOprimError):
            await git_worktree_create("", repo=tmp_path)
