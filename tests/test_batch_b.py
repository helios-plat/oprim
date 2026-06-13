"""
oprim 批次 B 测试套件
======================
17 个 oprim，每个 ≥5 个测试。
LSP/MCP 使用 mock handle（Protocol 注入，不需要真实服务器）。
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from oprim import (
    CodeAction, Completion, Diagnostic, Hover,
    HookResult, ImageBlock, Location, LspOprimError,
    McpOprimError, ShellOprimError, SkillMeta, Symbol,
    TextEdit, WorkspaceEdit,
    lsp_code_action, lsp_completion, lsp_definition,
    lsp_diagnostics, lsp_document_symbols, lsp_format,
    lsp_hover, lsp_references, lsp_rename, lsp_workspace_symbols,
    load_image, mcp_call_tool, mcp_list_tools,
    read_skill_frontmatter, run_hook,
)
from oprim.worktree import git_worktree_add, git_worktree_list, git_worktree_remove
from oprim._exceptions import FileOprimError, ParseOprimError, GitOprimError


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.com"], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], capture_output=True)
    f = repo / "main.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], capture_output=True)
    return repo


def make_lsp_server(response: Any = None, raises: Exception | None = None):
    """构造 mock LspServerHandle。"""
    server = AsyncMock()
    server.root_uri = "file:///project"
    if raises:
        server.request = AsyncMock(side_effect=raises)
    else:
        server.request = AsyncMock(return_value=response)
    return server


def make_mcp_client(list_response=None, call_response=None, raises=None):
    """构造 mock McpClientHandle。"""
    client = AsyncMock()
    if raises:
        client.list_tools = AsyncMock(side_effect=raises)
        client.call_tool = AsyncMock(side_effect=raises)
    else:
        client.list_tools = AsyncMock(return_value=list_response or [])
        client.call_tool = AsyncMock(return_value=call_response or {})
    return client


# ===========================================================================
# run_hook 测试
# ===========================================================================

class TestRunHook:
    def test_allow_decision_json(self, tmp_path):
        """hook 输出 JSON allow → HookResult.decision='allow'。"""
        script = tmp_path / "hook.sh"
        script.write_text('#!/bin/sh\necho \'{"decision":"allow","output":"ok"}\'')
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={"event": "PreToolUse"}))
        assert result.decision == "allow"
        assert result.exit_code == 0

    def test_block_decision_json(self, tmp_path):
        """hook 输出 JSON block → decision='block'。"""
        script = tmp_path / "block.sh"
        script.write_text('#!/bin/sh\necho \'{"decision":"block","output":"denied"}\'')
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={"event": "PreToolUse"}))
        assert result.decision == "block"

    def test_nonzero_exit_becomes_block(self, tmp_path):
        """非零退出码 → decision='block'（保守策略）。"""
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/sh\nexit 1")
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={}))
        assert result.decision == "block"
        assert result.exit_code == 1

    def test_non_json_output_allows(self, tmp_path):
        """非 JSON 输出 → allow，原始输出透传。"""
        script = tmp_path / "text.sh"
        script.write_text("#!/bin/sh\necho 'plain text output'")
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={}))
        assert result.decision == "allow"
        assert "plain text" in result.output

    def test_timeout_returns_allow(self, tmp_path):
        """超时 → allow（不阻塞主流程）。"""
        script = tmp_path / "slow.sh"
        script.write_text("#!/bin/sh\nsleep 10")
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={}, timeout=1))
        assert result.decision == "allow"
        assert result.exit_code == -1

    def test_receives_event_json_on_stdin(self, tmp_path):
        """hook 收到事件 JSON 并可读取。"""
        script = tmp_path / "echo_stdin.sh"
        script.write_text('#!/bin/sh\nread line\necho \'{"decision":"allow","output":"\'$line\'"}\' ')
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={"event": "test", "tool": "bash"}))
        assert result.decision == "allow"

    def test_returns_hook_result_type(self, tmp_path):
        """返回类型是 HookResult。"""
        script = tmp_path / "ok.sh"
        script.write_text('#!/bin/sh\necho \'{"decision":"allow","output":""}\'')
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={}))
        assert isinstance(result, HookResult)


# ===========================================================================
# load_image 测试
# ===========================================================================

class TestLoadImage:
    def test_loads_png(self, tmp_path):
        """PNG 文件正确编码为 base64。"""
        # 最小有效 PNG (1x1 透明)
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x62, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82,
        ])
        p = tmp_path / "test.png"
        p.write_bytes(png_bytes)
        block = load_image(p)
        assert block.media_type == "image/png"
        assert block.type == "image"
        assert block.source_type == "base64"
        assert base64.standard_b64decode(block.data) == png_bytes

    def test_jpeg_media_type(self, tmp_path):
        """JPEG 文件 media_type 正确。"""
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg_data")
        block = load_image(p)
        assert block.media_type == "image/jpeg"

    def test_jpeg_extension_alias(self, tmp_path):
        """.jpeg 扩展名也支持。"""
        p = tmp_path / "photo.jpeg"
        p.write_bytes(b"\xff\xd8fake")
        block = load_image(p)
        assert block.media_type == "image/jpeg"

    def test_webp_supported(self, tmp_path):
        p = tmp_path / "img.webp"
        p.write_bytes(b"RIFF....WEBP")
        block = load_image(p)
        assert block.media_type == "image/webp"

    def test_size_bytes_correct(self, tmp_path):
        p = tmp_path / "s.png"
        data = b"fakedata" * 100
        p.write_bytes(data)
        block = load_image(p)
        assert block.size_bytes == len(data)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileOprimError, match="not found"):
            load_image(tmp_path / "no.png")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_bytes(b"data")
        with pytest.raises(ParseOprimError, match="unsupported"):
            load_image(p)

    def test_returns_image_block(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"fakedata")
        assert isinstance(load_image(p), ImageBlock)

    def test_path_field_set(self, tmp_path):
        p = tmp_path / "y.gif"
        p.write_bytes(b"GIF89a")
        block = load_image(p)
        assert str(p.resolve()) == block.path


# ===========================================================================
# read_skill_frontmatter 测试
# ===========================================================================

class TestReadSkillFrontmatter:
    def _make_skill(self, tmp_path, frontmatter: str, body: str = "# body") -> Path:
        d = tmp_path / "skill"
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}")
        return d

    def test_reads_name(self, tmp_path):
        d = self._make_skill(tmp_path, "name: refactor_python\ndescription: refactors python code\nversion: 1.0.0")
        meta = read_skill_frontmatter(d)
        assert meta.name == "refactor_python"

    def test_reads_description(self, tmp_path):
        d = self._make_skill(tmp_path, "name: my_skill\ndescription: does stuff")
        meta = read_skill_frontmatter(d)
        assert meta.description == "does stuff"

    def test_reads_tools_list(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s\ntools: [bash_exec, file_read]")
        meta = read_skill_frontmatter(d)
        assert "bash_exec" in meta.tools
        assert "file_read" in meta.tools

    def test_reads_tags(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s\ntags: [python, refactor]")
        meta = read_skill_frontmatter(d)
        assert "python" in meta.tags

    def test_reads_version(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s\nversion: 2.1.0")
        meta = read_skill_frontmatter(d)
        assert meta.version == "2.1.0"

    def test_missing_skill_md_raises(self, tmp_path):
        d = tmp_path / "empty_skill"
        d.mkdir()
        with pytest.raises(FileOprimError, match="SKILL.md not found"):
            read_skill_frontmatter(d)

    def test_no_frontmatter_raises(self, tmp_path):
        d = tmp_path / "bad_skill"
        d.mkdir()
        (d / "SKILL.md").write_text("# No frontmatter here\njust body text")
        with pytest.raises(ParseOprimError, match="frontmatter"):
            read_skill_frontmatter(d)

    def test_missing_name_raises(self, tmp_path):
        d = self._make_skill(tmp_path, "description: no name field")
        with pytest.raises(ParseOprimError, match="name"):
            read_skill_frontmatter(d)

    def test_returns_skill_meta(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s")
        assert isinstance(read_skill_frontmatter(d), SkillMeta)

    def test_skill_dir_field(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s")
        meta = read_skill_frontmatter(d)
        assert str(d) == meta.skill_dir

    def test_raw_dict_preserved(self, tmp_path):
        d = self._make_skill(tmp_path, "name: s\ncustom_field: hello")
        meta = read_skill_frontmatter(d)
        assert "name" in meta.raw


# ===========================================================================
# git_worktree 测试
# ===========================================================================

class TestGitWorktree:
    def test_add_creates_directory(self, git_repo):
        wt = git_worktree_add("feat/wt-test", repo=git_repo)
        assert wt.exists()
        assert wt.is_dir()

    def test_add_returns_path(self, git_repo):
        wt = git_worktree_add("wt-branch", repo=git_repo)
        assert isinstance(wt, Path)

    def test_add_auto_path_naming(self, git_repo):
        wt = git_worktree_add("feat/my-feature", repo=git_repo)
        # 路径包含 branch safe name (/ → -)
        assert "feat-my-feature" in str(wt)

    def test_add_custom_path(self, git_repo, tmp_path):
        custom = tmp_path / "custom_wt"
        wt = git_worktree_add("custom-branch", repo=git_repo, path=custom)
        assert wt == custom.resolve()
        assert wt.exists()

    def test_list_shows_worktrees(self, git_repo):
        git_worktree_add("list-branch", repo=git_repo)
        wts = git_worktree_list(repo=git_repo)
        assert len(wts) >= 2  # 主 + 新 worktree
        assert all("path" in w for w in wts)

    def test_remove_deletes_directory(self, git_repo):
        wt = git_worktree_add("remove-branch", repo=git_repo)
        assert wt.exists()
        git_worktree_remove(wt, repo=git_repo)
        assert not wt.exists()

    def test_remove_force(self, git_repo):
        wt = git_worktree_add("force-branch", repo=git_repo)
        # 在 worktree 里写一个文件（未 commit）
        (wt / "dirty.py").write_text("x=1")
        # force remove
        git_worktree_remove(wt, repo=git_repo, force=True)
        assert not wt.exists()

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_worktree_add("branch", repo=tmp_path)


# ===========================================================================
# lsp_* 测试（全部使用 mock handle）
# ===========================================================================

class TestLspDiagnostics:
    def test_returns_diagnostics(self):
        server = make_lsp_server({
            "items": [
                {"range": {"start": {"line": 0, "character": 0},
                           "end": {"line": 0, "character": 5}},
                 "severity": 1, "message": "undefined name", "source": "pylsp"}
            ]
        })
        result = asyncio.run(lsp_diagnostics("main.py", server=server))
        assert len(result) == 1
        assert result[0].message == "undefined name"
        assert result[0].severity == 1

    def test_empty_result(self):
        server = make_lsp_server({"items": []})
        result = asyncio.run(lsp_diagnostics("main.py", server=server))
        assert result == []

    def test_sorted_by_line(self):
        server = make_lsp_server({"items": [
            {"range": {"start": {"line": 5, "character": 0},
                       "end": {"line": 5, "character": 1}},
             "severity": 2, "message": "warn", "source": ""},
            {"range": {"start": {"line": 1, "character": 0},
                       "end": {"line": 1, "character": 1}},
             "severity": 1, "message": "err", "source": ""},
        ]})
        result = asyncio.run(lsp_diagnostics("main.py", server=server))
        assert result[0].line == 1
        assert result[1].line == 5

    def test_severity_name(self):
        server = make_lsp_server({"items": [
            {"range": {"start": {"line": 0, "character": 0},
                       "end": {"line": 0, "character": 1}},
             "severity": 2, "message": "warn", "source": ""}
        ]})
        result = asyncio.run(lsp_diagnostics("main.py", server=server))
        assert result[0].severity_name == "warning"

    def test_raises_on_server_error(self):
        server = make_lsp_server(raises=RuntimeError("lsp error"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_diagnostics("main.py", server=server))

    def test_none_response_empty(self):
        server = make_lsp_server(None)
        result = asyncio.run(lsp_diagnostics("main.py", server=server))
        assert result == []


class TestLspHover:
    def test_returns_hover(self):
        server = make_lsp_server({
            "contents": {"kind": "markdown", "value": "```python\ndef foo()\n```"},
            "range": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 7}}
        })
        result = asyncio.run(lsp_hover("main.py", line=5, character=4, server=server))
        assert result is not None
        assert "foo" in result.contents
        assert result.range_start_line == 5

    def test_returns_none_when_no_hover(self):
        server = make_lsp_server(None)
        result = asyncio.run(lsp_hover("main.py", line=0, character=0, server=server))
        assert result is None

    def test_string_contents(self):
        server = make_lsp_server({"contents": "simple string"})
        result = asyncio.run(lsp_hover("f.py", line=0, character=0, server=server))
        assert result.contents == "simple string"

    def test_list_contents(self):
        server = make_lsp_server({"contents": [{"value": "a"}, {"value": "b"}]})
        result = asyncio.run(lsp_hover("f.py", line=0, character=0, server=server))
        assert "a" in result.contents and "b" in result.contents

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("boom"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_hover("f.py", line=0, character=0, server=server))


class TestLspDefinition:
    def test_returns_locations(self):
        server = make_lsp_server([{
            "uri": "file:///project/utils.py",
            "range": {"start": {"line": 10, "character": 0},
                      "end": {"line": 10, "character": 5}}
        }])
        result = asyncio.run(lsp_definition("main.py", line=5, character=4, server=server))
        assert len(result) == 1
        assert result[0].path == "/project/utils.py"
        assert result[0].start_line == 10

    def test_empty_result(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_definition("f.py", line=0, character=0, server=server))
        assert result == []

    def test_single_dict_result(self):
        server = make_lsp_server({
            "uri": "file:///a.py",
            "range": {"start": {"line": 3, "character": 0},
                      "end": {"line": 3, "character": 5}}
        })
        result = asyncio.run(lsp_definition("f.py", line=0, character=0, server=server))
        assert len(result) == 1

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_definition("f.py", line=0, character=0, server=server))

    def test_returns_location_objects(self):
        server = make_lsp_server([{"uri": "file:///x.py",
                                   "range": {"start": {"line": 0, "character": 0},
                                             "end": {"line": 0, "character": 1}}}])
        result = asyncio.run(lsp_definition("f.py", line=0, character=0, server=server))
        assert all(isinstance(l, Location) for l in result)


class TestLspReferences:
    def test_returns_multiple(self):
        server = make_lsp_server([
            {"uri": "file:///a.py", "range": {"start": {"line": 1, "character": 0},
                                              "end": {"line": 1, "character": 3}}},
            {"uri": "file:///b.py", "range": {"start": {"line": 2, "character": 5},
                                              "end": {"line": 2, "character": 8}}},
        ])
        result = asyncio.run(lsp_references("f.py", line=0, character=0, server=server))
        assert len(result) == 2

    def test_empty(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_references("f.py", line=0, character=0, server=server))
        assert result == []

    def test_include_declaration_param(self):
        server = make_lsp_server([])
        asyncio.run(lsp_references("f.py", line=0, character=0,
                                   server=server, include_declaration=True))
        call_params = server.request.call_args[0][1]
        assert call_params["context"]["includeDeclaration"] is True

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_references("f.py", line=0, character=0, server=server))

    def test_returns_location_objects(self):
        server = make_lsp_server([{"uri": "file:///x.py",
                                   "range": {"start": {"line": 0, "character": 0},
                                             "end": {"line": 0, "character": 1}}}])
        result = asyncio.run(lsp_references("f.py", line=0, character=0, server=server))
        assert all(isinstance(l, Location) for l in result)


class TestLspDocumentSymbols:
    def test_returns_symbols(self):
        server = make_lsp_server([
            {"name": "MyClass", "kind": 5,
             "range": {"start": {"line": 0, "character": 0},
                       "end": {"line": 10, "character": 0}},
             "selectionRange": {"start": {"line": 0, "character": 6},
                                "end": {"line": 0, "character": 13}}},
        ])
        result = asyncio.run(lsp_document_symbols("main.py", server=server))
        assert result[0].name == "MyClass"
        assert result[0].kind == 5
        assert result[0].kind_name == "Class"

    def test_empty(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_document_symbols("f.py", server=server))
        assert result == []

    def test_sorted_by_line(self):
        server = make_lsp_server([
            {"name": "b", "kind": 12,
             "range": {"start": {"line": 10, "character": 0},
                       "end": {"line": 12, "character": 0}}},
            {"name": "a", "kind": 12,
             "range": {"start": {"line": 2, "character": 0},
                       "end": {"line": 4, "character": 0}}},
        ])
        result = asyncio.run(lsp_document_symbols("f.py", server=server))
        assert result[0].name == "a"

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_document_symbols("f.py", server=server))

    def test_returns_symbol_objects(self):
        server = make_lsp_server([{"name": "x", "kind": 6,
                                   "range": {"start": {"line": 0, "character": 0},
                                             "end": {"line": 0, "character": 5}}}])
        result = asyncio.run(lsp_document_symbols("f.py", server=server))
        assert all(isinstance(s, Symbol) for s in result)


class TestLspWorkspaceSymbols:
    def test_returns_symbols(self):
        server = make_lsp_server([{
            "name": "parse_args", "kind": 12,
            "location": {"uri": "file:///main.py",
                         "range": {"start": {"line": 5, "character": 0},
                                   "end": {"line": 10, "character": 0}}}
        }])
        result = asyncio.run(lsp_workspace_symbols("parse", server=server))
        assert result[0].name == "parse_args"
        assert result[0].path == "/main.py"

    def test_empty_query(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_workspace_symbols("", server=server))
        assert result == []

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_workspace_symbols("foo", server=server))

    def test_query_passed_to_server(self):
        server = make_lsp_server([])
        asyncio.run(lsp_workspace_symbols("myQuery", server=server))
        assert server.request.call_args[0][1]["query"] == "myQuery"

    def test_returns_symbol_objects(self):
        server = make_lsp_server([{"name": "x", "kind": 12,
                                   "location": {"uri": "file:///f.py",
                                                "range": {"start": {"line": 0, "character": 0},
                                                          "end": {"line": 1, "character": 0}}}}])
        result = asyncio.run(lsp_workspace_symbols("x", server=server))
        assert all(isinstance(s, Symbol) for s in result)


class TestLspRename:
    def test_returns_workspace_edit(self):
        server = make_lsp_server({
            "changes": {
                "file:///a.py": [{"range": {"start": {"line": 1, "character": 0},
                                            "end": {"line": 1, "character": 3}},
                                  "newText": "new_name"}]
            }
        })
        result = asyncio.run(lsp_rename("a.py", line=1, character=0,
                                        new_name="new_name", server=server))
        assert isinstance(result, WorkspaceEdit)
        assert "/a.py" in result.changes

    def test_empty_result_returns_empty_edit(self):
        server = make_lsp_server(None)
        result = asyncio.run(lsp_rename("f.py", line=0, character=0,
                                        new_name="x", server=server))
        assert isinstance(result, WorkspaceEdit)
        assert result.changes == {}

    def test_new_name_passed(self):
        server = make_lsp_server({})
        asyncio.run(lsp_rename("f.py", line=0, character=0, new_name="renamed", server=server))
        assert server.request.call_args[0][1]["newName"] == "renamed"

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_rename("f.py", line=0, character=0, new_name="x", server=server))

    def test_text_edits_parsed(self):
        server = make_lsp_server({
            "changes": {
                "file:///b.py": [{"range": {"start": {"line": 2, "character": 0},
                                            "end": {"line": 2, "character": 3}},
                                  "newText": "renamed"}]
            }
        })
        result = asyncio.run(lsp_rename("f.py", line=0, character=0,
                                        new_name="renamed", server=server))
        edits = result.changes.get("/b.py", [])
        assert edits[0].new_text == "renamed"


class TestLspCompletion:
    def test_returns_completions(self):
        server = make_lsp_server({
            "items": [
                {"label": "parse_args", "kind": 3, "detail": "function",
                 "documentation": "parses args", "insertText": "parse_args"},
            ]
        })
        result = asyncio.run(lsp_completion("f.py", line=5, character=3, server=server))
        assert result[0].label == "parse_args"

    def test_list_response(self):
        server = make_lsp_server([{"label": "foo", "kind": 6}])
        result = asyncio.run(lsp_completion("f.py", line=0, character=0, server=server))
        assert result[0].label == "foo"

    def test_max_50_items(self):
        server = make_lsp_server({"items": [{"label": f"item{i}"} for i in range(100)]})
        result = asyncio.run(lsp_completion("f.py", line=0, character=0, server=server))
        assert len(result) == 50

    def test_trigger_character_passed(self):
        server = make_lsp_server({"items": []})
        asyncio.run(lsp_completion("f.py", line=0, character=5,
                                   server=server, trigger_character="."))
        params = server.request.call_args[0][1]
        assert params["context"]["triggerCharacter"] == "."
        assert params["context"]["triggerKind"] == 2

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_completion("f.py", line=0, character=0, server=server))

    def test_returns_completion_objects(self):
        server = make_lsp_server({"items": [{"label": "x"}]})
        result = asyncio.run(lsp_completion("f.py", line=0, character=0, server=server))
        assert all(isinstance(c, Completion) for c in result)


class TestLspFormat:
    def test_returns_text_edits(self):
        server = make_lsp_server([
            {"range": {"start": {"line": 1, "character": 0},
                       "end": {"line": 1, "character": 3}},
             "newText": "    x = 1"}
        ])
        result = asyncio.run(lsp_format("f.py", server=server))
        assert result[0].new_text == "    x = 1"

    def test_empty_means_already_formatted(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_format("f.py", server=server))
        assert result == []

    def test_tab_size_passed(self):
        server = make_lsp_server([])
        asyncio.run(lsp_format("f.py", server=server, tab_size=2))
        params = server.request.call_args[0][1]
        assert params["options"]["tabSize"] == 2

    def test_insert_spaces_passed(self):
        server = make_lsp_server([])
        asyncio.run(lsp_format("f.py", server=server, insert_spaces=False))
        params = server.request.call_args[0][1]
        assert params["options"]["insertSpaces"] is False

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_format("f.py", server=server))

    def test_returns_text_edit_objects(self):
        server = make_lsp_server([{"range": {"start": {"line": 0, "character": 0},
                                             "end": {"line": 0, "character": 1}},
                                   "newText": "x"}])
        result = asyncio.run(lsp_format("f.py", server=server))
        assert all(isinstance(e, TextEdit) for e in result)


class TestLspCodeAction:
    def test_returns_actions(self):
        server = make_lsp_server([
            {"title": "Import os", "kind": "quickfix"},
            {"title": "Extract method", "kind": "refactor"},
        ])
        result = asyncio.run(lsp_code_action("f.py",
                                             start_line=0, start_character=0,
                                             end_line=0, end_character=5,
                                             server=server))
        assert len(result) == 2
        assert result[0].title == "Import os"

    def test_empty(self):
        server = make_lsp_server([])
        result = asyncio.run(lsp_code_action("f.py", start_line=0, start_character=0,
                                             end_line=0, end_character=1, server=server))
        assert result == []

    def test_only_kinds_passed(self):
        server = make_lsp_server([])
        asyncio.run(lsp_code_action("f.py", start_line=0, start_character=0,
                                    end_line=0, end_character=1, server=server,
                                    only_kinds=["quickfix"]))
        params = server.request.call_args[0][1]
        assert params["context"]["only"] == ["quickfix"]

    def test_raises_on_error(self):
        server = make_lsp_server(raises=RuntimeError("err"))
        with pytest.raises(LspOprimError):
            asyncio.run(lsp_code_action("f.py", start_line=0, start_character=0,
                                        end_line=0, end_character=1, server=server))

    def test_returns_code_action_objects(self):
        server = make_lsp_server([{"title": "Fix", "kind": "quickfix"}])
        result = asyncio.run(lsp_code_action("f.py", start_line=0, start_character=0,
                                             end_line=0, end_character=1, server=server))
        assert all(isinstance(a, CodeAction) for a in result)


# ===========================================================================
# mcp_list_tools / mcp_call_tool 测试
# ===========================================================================

class TestMcpListTools:
    def test_returns_tools(self):
        client = make_mcp_client(list_response=[
            {"name": "search_web", "description": "searches web",
             "inputSchema": {"type": "object"}}
        ])
        result = asyncio.run(mcp_list_tools(client=client))
        assert result[0]["name"] == "search_web"

    def test_normalizes_input_schema(self):
        client = make_mcp_client(list_response=[
            {"name": "tool", "input_schema": {"type": "object"}}
        ])
        result = asyncio.run(mcp_list_tools(client=client))
        assert "inputSchema" in result[0]

    def test_empty_returns_empty(self):
        client = make_mcp_client(list_response=[])
        result = asyncio.run(mcp_list_tools(client=client))
        assert result == []

    def test_raises_on_error(self):
        client = make_mcp_client(raises=RuntimeError("conn error"))
        with pytest.raises(McpOprimError):
            asyncio.run(mcp_list_tools(client=client))

    def test_filters_non_dict(self):
        client = make_mcp_client(list_response=[
            {"name": "valid", "description": "ok"},
            "not_a_dict",
            42,
        ])
        result = asyncio.run(mcp_list_tools(client=client))
        assert len(result) == 1


class TestMcpCallTool:
    def test_returns_content(self):
        client = make_mcp_client(call_response={
            "content": [{"type": "text", "text": "result text"}],
            "isError": False
        })
        result = asyncio.run(mcp_call_tool("search_web",
                                           arguments={"query": "python"},
                                           client=client))
        assert result["content"][0]["text"] == "result text"
        assert result["isError"] is False

    def test_name_and_args_passed(self):
        client = make_mcp_client(call_response={"content": [], "isError": False})
        asyncio.run(mcp_call_tool("my_tool", arguments={"key": "val"}, client=client))
        client.call_tool.assert_called_once_with("my_tool", {"key": "val"})

    def test_non_dict_response_wrapped(self):
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value="raw string")
        result = asyncio.run(mcp_call_tool("t", arguments={}, client=client))
        assert "content" in result

    def test_missing_content_field_added(self):
        client = make_mcp_client(call_response={"data": "something"})
        result = asyncio.run(mcp_call_tool("t", arguments={}, client=client))
        assert "content" in result

    def test_raises_on_error(self):
        client = make_mcp_client(raises=RuntimeError("timeout"))
        with pytest.raises(McpOprimError):
            asyncio.run(mcp_call_tool("t", arguments={}, client=client))

    def test_is_error_preserved(self):
        client = make_mcp_client(call_response={
            "content": [{"type": "text", "text": "error msg"}],
            "isError": True
        })
        result = asyncio.run(mcp_call_tool("t", arguments={}, client=client))
        assert result["isError"] is True


# ===========================================================================
# 覆盖率补足
# ===========================================================================

class TestBatchBCoverageGaps:
    """补足 B 批次剩余 miss 行。"""

    # --- hooks_image_skill.py: decision invalid → allow
    def test_run_hook_invalid_decision_defaults_allow(self, tmp_path):
        """decision 值不在 allow/block/modify → 默认 allow。"""
        script = tmp_path / "h.sh"
        script.write_text('#!/bin/sh\necho \'{"decision":"UNKNOWN","output":"x"}\'')
        script.chmod(0o755)
        result = asyncio.run(run_hook(str(script), event_json={}))
        assert result.decision == "allow"

    # --- hooks_image_skill.py: _to_str_list string input
    def test_skill_tools_string_becomes_list(self, tmp_path):
        """tools 字段为单个字符串时 → 变成单元素列表。"""
        d = tmp_path / "sk"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: s\ntools: bash_exec\n---\n# body")
        meta = read_skill_frontmatter(d)
        assert isinstance(meta.tools, list)

    # --- hooks_image_skill.py: _parse_simple_yaml ImportError path
    # 在有 pyyaml 的环境里无法走到，直接 pragma
    # lsp.py: completion doc as dict
    def test_lsp_completion_doc_dict(self):
        """completion documentation 为 dict 时取 value 字段。"""
        server = make_lsp_server({"items": [
            {"label": "foo", "documentation": {"kind": "markdown", "value": "doc text"}}
        ]})
        result = asyncio.run(lsp_completion("f.py", line=0, character=0, server=server))
        assert "doc text" in result[0].documentation

    # --- lsp.py: _parse_symbols non-dict item skip (line 684)
    def test_lsp_document_symbols_filters_non_dict(self):
        """symbols 列表中非 dict 项被跳过。"""
        server = make_lsp_server([
            {"name": "good", "kind": 12,
             "range": {"start": {"line": 0, "character": 0},
                       "end": {"line": 1, "character": 0}}},
            "not_a_dict",
            42,
        ])
        result = asyncio.run(lsp_document_symbols("f.py", server=server))
        assert len(result) == 1 and result[0].name == "good"

    # --- lsp.py: _parse_locations non-dict skip (line 709)
    def test_lsp_references_filters_non_dict(self):
        server = make_lsp_server([
            {"uri": "file:///a.py", "range": {"start": {"line": 0, "character": 0},
                                              "end": {"line": 0, "character": 1}}},
            "bad_item",
        ])
        result = asyncio.run(lsp_references("f.py", line=0, character=0, server=server))
        assert len(result) == 1

    # --- lsp.py: _parse_workspace_edit documentChanges path (lines 733-738)
    def test_lsp_rename_document_changes_format(self):
        """documentChanges 格式的 WorkspaceEdit 也能正确解析。"""
        server = make_lsp_server({
            "documentChanges": [
                {
                    "textDocument": {"uri": "file:///c.py", "version": 1},
                    "edits": [{"range": {"start": {"line": 3, "character": 0},
                                         "end": {"line": 3, "character": 5}},
                               "newText": "renamed"}]
                }
            ]
        })
        result = asyncio.run(lsp_rename("f.py", line=0, character=0,
                                        new_name="renamed", server=server))
        assert "/c.py" in result.changes
        assert result.changes["/c.py"][0].new_text == "renamed"

    # --- lsp.py: non-dict item in documentChanges (line 733)
    def test_lsp_rename_document_changes_non_dict_skipped(self):
        server = make_lsp_server({
            "documentChanges": ["not_a_dict", None]
        })
        result = asyncio.run(lsp_rename("f.py", line=0, character=0,
                                        new_name="x", server=server))
        assert result.changes == {}

    # --- worktree.py: list branch field (line 68)
    def test_worktree_list_branch_field(self, git_repo):
        """worktree list 的 branch 字段去掉 refs/heads/ 前缀。"""
        wts = git_worktree_list(repo=git_repo)
        main_wt = wts[0]
        # 应该是 "main" 或 "master"，不含 refs/heads/
        assert "refs/heads/" not in main_wt["branch"]

    # --- _protocols.py: Protocol 方法体不执行（pragma）
    # 这些 ... 行不会被测试到，Protocol 只做类型检查用途
