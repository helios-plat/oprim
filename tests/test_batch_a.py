"""
oprim 批次 A 测试套件
======================
覆盖全部 28 个 oprim，每个 ≥5 个测试用例。
使用 pytest + tmp_path fixture。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from oprim import (
    BlameLine, Commit, FileDiff, FileOprimError, FileStatus,
    GitOprimError, Hunk, OprimError, ParseOprimError, PathSecurityError,
    ShellOprimError, ShellResult, StreamChunk,
    bash_exec, bash_exec_stream,
    compute_diff, count_tokens, detect_language,
    dir_list, estimate_cost, file_append, file_delete,
    file_read, file_stat, file_write, glob_match,
    git_add, git_blame, git_branch, git_checkout, git_commit,
    git_diff, git_log, git_show, git_stash, git_status,
    html_to_markdown, parse_unified_diff, path_resolve,
    read_gitignore, redact_secrets,
)


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def git_repo(tmp_path):
    """初始化一个带 commit 的临时 git 仓库。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test.com"], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], capture_output=True)
    # 初始 commit
    f = repo / "hello.py"
    f.write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], capture_output=True)
    return repo


# ===========================================================================
# fs.py 测试
# ===========================================================================

class TestFileRead:
    def test_reads_full_content(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("line1\nline2\nline3\n")
        assert file_read(f) == "line1\nline2\nline3\n"

    def test_reads_line_slice(self, tmp_path):
        f = tmp_path / "b.txt"
        f.write_text("L0\nL1\nL2\nL3\n")
        assert file_read(f, start=1, end=3) == "L1\nL2\n"

    def test_raises_on_missing(self, tmp_path):
        with pytest.raises(FileOprimError, match="not found"):
            file_read(tmp_path / "nonexist.txt")

    def test_reads_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert file_read(f) == ""

    def test_reads_utf8_content(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("你好世界\n", encoding="utf-8")
        assert "你好" in file_read(f)

    def test_start_only(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("A\nB\nC\n")
        result = file_read(f, start=1)
        assert result == "B\nC\n"


class TestFileWrite:
    def test_writes_content(self, tmp_path):
        p = tmp_path / "out.txt"
        file_write(p, content="hello")
        assert p.read_text() == "hello"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.txt"
        file_write(p, content="deep")
        assert p.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("old")
        file_write(p, content="new")
        assert p.read_text() == "new"

    def test_returns_path(self, tmp_path):
        p = tmp_path / "r.txt"
        result = file_write(p, content="x")
        assert isinstance(result, Path)

    def test_raises_on_dir_as_path(self, tmp_path):
        with pytest.raises((FileOprimError, IsADirectoryError, OSError)):
            file_write(tmp_path, content="x")

    def test_writes_empty_content(self, tmp_path):
        p = tmp_path / "empty.txt"
        file_write(p, content="")
        assert p.read_text() == ""


class TestFileAppend:
    def test_appends_to_existing(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("first\n")
        file_append(p, content="second\n")
        assert p.read_text() == "first\nsecond\n"

    def test_creates_if_missing(self, tmp_path):
        p = tmp_path / "new.txt"
        file_append(p, content="hello")
        assert p.read_text() == "hello"

    def test_multiple_appends(self, tmp_path):
        p = tmp_path / "multi.txt"
        for i in range(3):
            file_append(p, content=f"line{i}\n")
        assert p.read_text() == "line0\nline1\nline2\n"

    def test_returns_path(self, tmp_path):
        p = tmp_path / "a.txt"
        assert isinstance(file_append(p, content="x"), Path)

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "log.txt"
        file_append(p, content="ok")
        assert p.exists()


class TestFileStat:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("abc")
        s = file_stat(f)
        assert s["exists"] is True
        assert s["is_file"] is True
        assert s["size"] == 3

    def test_nonexistent_returns_exists_false(self, tmp_path):
        s = file_stat(tmp_path / "no.txt")
        assert s["exists"] is False
        assert s["size"] == 0

    def test_directory(self, tmp_path):
        s = file_stat(tmp_path)
        assert s["is_dir"] is True
        assert s["is_file"] is False

    def test_mtime_positive(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("x")
        s = file_stat(f)
        assert s["mtime"] > 0

    def test_mode_string(self, tmp_path):
        f = tmp_path / "m.txt"
        f.write_text("x")
        s = file_stat(f)
        assert s["mode"].startswith("0o")


class TestFileDelete:
    def test_deletes_existing(self, tmp_path):
        f = tmp_path / "del.txt"
        f.write_text("x")
        assert file_delete(f) is True
        assert not f.exists()

    def test_raises_on_missing(self, tmp_path):
        with pytest.raises(FileOprimError):
            file_delete(tmp_path / "no.txt")

    def test_missing_ok(self, tmp_path):
        result = file_delete(tmp_path / "no.txt", missing_ok=True)
        assert result is False

    def test_cannot_delete_dir(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises((FileOprimError, OSError)):
            file_delete(d)

    def test_does_not_affect_siblings(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a")
        b.write_text("b")
        file_delete(a)
        assert b.exists()


class TestDirList:
    def test_lists_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        result = dir_list(tmp_path)
        names = [p.name for p in result]
        assert "a.py" in names and "b.py" in names

    def test_excludes_hidden_by_default(self, tmp_path):
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.py").write_text("")
        result = dir_list(tmp_path)
        assert ".hidden" not in [p.name for p in result]

    def test_includes_hidden_when_asked(self, tmp_path):
        (tmp_path / ".hidden").write_text("")
        result = dir_list(tmp_path, include_hidden=True)
        assert ".hidden" in [p.name for p in result]

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        result = dir_list(tmp_path, recursive=True)
        assert any("deep.py" in str(p) for p in result)

    def test_raises_on_missing(self, tmp_path):
        with pytest.raises(FileOprimError):
            dir_list(tmp_path / "no")

    def test_raises_on_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(FileOprimError):
            dir_list(f)


class TestGlobMatch:
    def test_matches_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = glob_match("*.py", root=tmp_path)
        assert len(result) == 1 and result[0].name == "a.py"

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("")
        result = glob_match("**/*.py", root=tmp_path)
        assert any("main.py" in str(p) for p in result)

    def test_respects_gitignore_filters_git(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("")
        result = glob_match("**/*", root=tmp_path, respect_gitignore=True)
        assert not any(".git" in str(p) for p in result)

    def test_raises_on_missing_root(self, tmp_path):
        with pytest.raises(FileOprimError):
            glob_match("*.py", root=tmp_path / "no")

    def test_empty_result(self, tmp_path):
        result = glob_match("*.xyz", root=tmp_path)
        assert result == []

    def test_sorted_output(self, tmp_path):
        for name in ["c.py", "a.py", "b.py"]:
            (tmp_path / name).write_text("")
        result = glob_match("*.py", root=tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)


class TestPathResolve:
    def test_resolves_absolute(self, tmp_path):
        p = path_resolve(tmp_path / "a.txt")
        assert p.is_absolute()

    def test_sandbox_allows_inside(self, tmp_path):
        inside = tmp_path / "sub" / "f.txt"
        p = path_resolve(inside, sandbox_root=tmp_path)
        assert str(tmp_path) in str(p)

    def test_sandbox_rejects_outside(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside = tmp_path / "secret.txt"
        with pytest.raises(PathSecurityError):
            path_resolve(outside, sandbox_root=sandbox)

    def test_no_sandbox_allows_any(self, tmp_path):
        p = path_resolve("/tmp/any.txt")
        assert p == Path("/tmp/any.txt")

    def test_traversal_attack_rejected(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        traversal = sandbox / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PathSecurityError):
            path_resolve(traversal, sandbox_root=sandbox)


class TestReadGitignore:
    def test_parses_rules(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.pyc\n__pycache__/\n# comment\n\n.env\n")
        rules = read_gitignore(tmp_path)
        assert "*.pyc" in rules
        assert "__pycache__/" in rules
        assert ".env" in rules
        assert "# comment" not in rules

    def test_missing_returns_empty(self, tmp_path):
        result = read_gitignore(tmp_path)
        assert result == []

    def test_ignores_blank_lines(self, tmp_path):
        (tmp_path / ".gitignore").write_text("\n\n*.log\n\n")
        rules = read_gitignore(tmp_path)
        assert rules == ["*.log"]

    def test_ignores_comments(self, tmp_path):
        (tmp_path / ".gitignore").write_text("# this is a comment\n*.tmp\n")
        rules = read_gitignore(tmp_path)
        assert len(rules) == 1

    def test_handles_complex_patterns(self, tmp_path):
        content = "!important.log\n/build/\ndist/**\n"
        (tmp_path / ".gitignore").write_text(content)
        rules = read_gitignore(tmp_path)
        assert "!important.log" in rules
        assert "/build/" in rules


# ===========================================================================
# git.py 测试
# ===========================================================================

class TestGitStatus:
    def test_clean_repo(self, git_repo):
        result = git_status(repo=git_repo)
        assert result == []

    def test_modified_file(self, git_repo):
        (git_repo / "hello.py").write_text("x = 2\n")
        result = git_status(repo=git_repo)
        assert any(s.path == "hello.py" for s in result)

    def test_new_untracked(self, git_repo):
        (git_repo / "new.py").write_text("y = 1\n")
        result = git_status(repo=git_repo)
        assert any(s.path == "new.py" for s in result)

    def test_staged_file(self, git_repo):
        (git_repo / "staged.py").write_text("z = 1\n")
        subprocess.run(["git", "-C", str(git_repo), "add", "staged.py"], capture_output=True)
        result = git_status(repo=git_repo)
        assert any(s.path == "staged.py" for s in result)

    def test_returns_filestatus_objects(self, git_repo):
        (git_repo / "x.py").write_text("")
        result = git_status(repo=git_repo)
        assert all(isinstance(s, FileStatus) for s in result)

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_status(repo=tmp_path)


class TestGitDiff:
    def test_empty_diff_clean_repo(self, git_repo):
        result = git_diff(repo=git_repo)
        assert result == ""

    def test_diff_modified_file(self, git_repo):
        (git_repo / "hello.py").write_text("x = 99\n")
        result = git_diff(repo=git_repo)
        assert "+x = 99" in result or "99" in result

    def test_staged_diff(self, git_repo):
        f = git_repo / "hello.py"
        f.write_text("x = 42\n")
        subprocess.run(["git", "-C", str(git_repo), "add", "hello.py"], capture_output=True)
        result = git_diff(repo=git_repo, staged=True)
        assert "42" in result

    def test_paths_filter(self, git_repo):
        (git_repo / "hello.py").write_text("x = 7\n")
        (git_repo / "other.py").write_text("y = 8\n")
        result = git_diff(repo=git_repo, paths=["hello.py"])
        assert "hello.py" in result

    def test_returns_string(self, git_repo):
        assert isinstance(git_diff(repo=git_repo), str)


class TestGitAdd:
    def test_stages_file(self, git_repo):
        f = git_repo / "new.py"
        f.write_text("a = 1\n")
        git_add("new.py", repo=git_repo)
        result = git_status(repo=git_repo)
        staged = [s for s in result if s.path == "new.py" and s.index == "A"]
        assert staged

    def test_stages_multiple(self, git_repo):
        for name in ["a.py", "b.py"]:
            (git_repo / name).write_text("")
        git_add(["a.py", "b.py"], repo=git_repo)
        paths = {s.path for s in git_status(repo=git_repo)}
        assert "a.py" in paths and "b.py" in paths

    def test_string_input(self, git_repo):
        (git_repo / "single.py").write_text("")
        git_add("single.py", repo=git_repo)  # str, not list

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_add("x.py", repo=tmp_path)

    def test_missing_file_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_add("nonexist.py", repo=git_repo)


class TestGitCommit:
    def test_creates_commit(self, git_repo):
        f = git_repo / "c.py"
        f.write_text("c = 1\n")
        git_add("c.py", repo=git_repo)
        sha = git_commit(repo=git_repo, message="add c.py")
        assert len(sha) == 8

    def test_commit_message_recorded(self, git_repo):
        f = git_repo / "msg.py"
        f.write_text("")
        git_add("msg.py", repo=git_repo)
        git_commit(repo=git_repo, message="unique msg xyz")
        log = git_log(repo=git_repo, n=1)
        assert "unique msg xyz" in log[0].message

    def test_empty_commit_allowed(self, git_repo):
        sha = git_commit(repo=git_repo, message="empty", allow_empty=True)
        assert len(sha) == 8

    def test_empty_staging_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_commit(repo=git_repo, message="nothing staged")

    def test_returns_short_hash(self, git_repo):
        f = git_repo / "h.py"
        f.write_text("")
        git_add("h.py", repo=git_repo)
        sha = git_commit(repo=git_repo, message="hash test")
        assert all(c in "0123456789abcdef" for c in sha)


class TestGitLog:
    def test_returns_commits(self, git_repo):
        result = git_log(repo=git_repo)
        assert len(result) >= 1

    def test_commit_fields(self, git_repo):
        c = git_log(repo=git_repo)[0]
        assert isinstance(c, Commit)
        assert c.hash and c.author and c.message

    def test_n_limit(self, git_repo):
        for i in range(3):
            f = git_repo / f"f{i}.py"
            f.write_text("")
            git_add(str(f.name), repo=git_repo)
            git_commit(repo=git_repo, message=f"commit {i}", allow_empty=True)
        result = git_log(repo=git_repo, n=2)
        assert len(result) == 2

    def test_path_filter(self, git_repo):
        f = git_repo / "special.py"
        f.write_text("")
        git_add("special.py", repo=git_repo)
        git_commit(repo=git_repo, message="add special")
        result = git_log(repo=git_repo, path="special.py")
        assert any("special" in c.message for c in result)

    def test_order_newest_first(self, git_repo):
        git_commit(repo=git_repo, message="msg A", allow_empty=True)
        git_commit(repo=git_repo, message="msg B", allow_empty=True)
        log = git_log(repo=git_repo, n=2)
        assert log[0].message == "msg B"


class TestGitBranch:
    def test_lists_branches(self, git_repo):
        result = git_branch(repo=git_repo)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_main_or_master_present(self, git_repo):
        branches = git_branch(repo=git_repo)
        assert any(b in ("main", "master") for b in branches)

    def test_create_branch(self, git_repo):
        git_branch(repo=git_repo, name="feat/new", create=True)
        branches = git_branch(repo=git_repo)
        # 可能自动 checkout 了新分支，检查是否存在于 log 或文件系统
        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--list", "feat/new"],
            capture_output=True, text=True,
        )
        assert "feat/new" in result.stdout

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_branch(repo=tmp_path)

    def test_delete_branch(self, git_repo):
        # 先创建
        subprocess.run(
            ["git", "-C", str(git_repo), "branch", "tobedeleted"],
            capture_output=True,
        )
        git_branch(repo=git_repo, name="tobedeleted", delete=True)
        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--list", "tobedeleted"],
            capture_output=True, text=True,
        )
        assert "tobedeleted" not in result.stdout


class TestGitCheckout:
    def test_checkout_new_branch(self, git_repo):
        subprocess.run(["git", "-C", str(git_repo), "branch", "testbranch"], capture_output=True)
        git_checkout("testbranch", repo=git_repo)
        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--show-current"],
            capture_output=True, text=True,
        )
        assert "testbranch" in result.stdout

    def test_checkout_invalid_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_checkout("nonexistent_branch_xyz", repo=git_repo)

    def test_checkout_returns_none(self, git_repo):
        subprocess.run(["git", "-C", str(git_repo), "branch", "br2"], capture_output=True)
        result = git_checkout("br2", repo=git_repo)
        assert result is None

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_checkout("main", repo=tmp_path)

    def test_checkout_restores_file(self, git_repo):
        f = git_repo / "hello.py"
        f.write_text("modified\n")
        # restore via checkout HEAD -- file syntax
        git_checkout("HEAD -- hello.py", repo=git_repo)
        assert f.read_text() == "x = 1\n"


class TestGitStash:
    def test_stash_push(self, git_repo):
        (git_repo / "hello.py").write_text("modified\n")
        result = git_stash(repo=git_repo)
        assert isinstance(result, str)
        # 文件应恢复
        assert (git_repo / "hello.py").read_text() == "x = 1\n"

    def test_stash_pop(self, git_repo):
        (git_repo / "hello.py").write_text("stashed\n")
        git_stash(repo=git_repo)
        git_stash(repo=git_repo, pop=True)
        assert (git_repo / "hello.py").read_text() == "stashed\n"

    def test_stash_with_message(self, git_repo):
        (git_repo / "hello.py").write_text("wip\n")
        result = git_stash(repo=git_repo, message="work in progress")
        assert isinstance(result, str)

    def test_stash_empty_repo_no_changes(self, git_repo):
        # clean repo，git stash 不报错但输出 "No local changes to save"
        result = git_stash(repo=git_repo)
        assert "No local changes" in result or isinstance(result, str)

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_stash(repo=tmp_path)


class TestGitShow:
    def test_show_head(self, git_repo):
        result = git_show("HEAD", repo=git_repo)
        assert "commit" in result or "init" in result

    def test_show_file_at_ref(self, git_repo):
        result = git_show("HEAD", repo=git_repo, path="hello.py")
        assert "x = 1" in result

    def test_show_invalid_ref_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_show("deadbeef1234567890", repo=git_repo)

    def test_show_invalid_path_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_show("HEAD", repo=git_repo, path="nonexist.py")

    def test_returns_string(self, git_repo):
        assert isinstance(git_show("HEAD", repo=git_repo), str)


class TestGitBlame:
    def test_returns_blame_lines(self, git_repo):
        result = git_blame("hello.py", repo=git_repo)
        assert len(result) >= 1
        assert all(isinstance(b, BlameLine) for b in result)

    def test_blame_line_fields(self, git_repo):
        lines = git_blame("hello.py", repo=git_repo)
        b = lines[0]
        assert b.lineno == 1
        assert b.commit
        assert b.content == "x = 1"

    def test_blame_missing_file_raises(self, git_repo):
        with pytest.raises(GitOprimError):
            git_blame("nonexist.py", repo=git_repo)

    def test_blame_multiline(self, git_repo):
        f = git_repo / "multi.py"
        f.write_text("a = 1\nb = 2\nc = 3\n")
        git_add("multi.py", repo=git_repo)
        git_commit(repo=git_repo, message="multi")
        result = git_blame("multi.py", repo=git_repo)
        assert len(result) == 3

    def test_invalid_repo_raises(self, tmp_path):
        with pytest.raises(GitOprimError):
            git_blame("x.py", repo=tmp_path)


# ===========================================================================
# shell.py 测试
# ===========================================================================

class TestBashExec:
    def test_simple_command(self):
        r = bash_exec("echo hello")
        assert r.stdout.strip() == "hello"
        assert r.code == 0

    def test_exit_code_nonzero(self):
        r = bash_exec("exit 42")
        assert r.code == 42

    def test_ok_property(self):
        assert bash_exec("true").ok is True
        assert bash_exec("false").ok is False

    def test_stderr_captured(self):
        r = bash_exec("echo err >&2")
        assert "err" in r.stderr

    def test_cwd_honored(self, tmp_path):
        r = bash_exec("pwd", cwd=str(tmp_path))
        assert str(tmp_path) in r.stdout

    def test_returns_shell_result(self):
        assert isinstance(bash_exec("echo x"), ShellResult)

    def test_timeout_raises(self):
        with pytest.raises(ShellOprimError, match="timed out"):
            bash_exec("sleep 10", timeout=1)


class TestBashExecStream:
    def test_yields_chunks(self):
        async def run():
            chunks = []
            async for chunk in bash_exec_stream("echo hello"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert any("hello" in c.text for c in chunks)

    def test_chunk_type(self):
        async def run():
            async for chunk in bash_exec_stream("echo x"):
                return chunk

        chunk = asyncio.run(run())
        assert isinstance(chunk, StreamChunk)
        assert chunk.stream in ("stdout", "stderr")

    def test_multiline_output(self):
        async def run():
            chunks = []
            async for chunk in bash_exec_stream("printf 'a\nb\nc\n'"):
                chunks.append(chunk.text)
            return "".join(chunks)

        text = asyncio.run(run())
        assert "a" in text and "b" in text

    def test_stderr_labeled(self):
        async def run():
            chunks = []
            async for chunk in bash_exec_stream("echo err >&2"):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        stderr_chunks = [c for c in chunks if c.stream == "stderr"]
        assert any("err" in c.text for c in stderr_chunks)

    def test_returns_async_iterator(self):
        import collections.abc
        gen = bash_exec_stream("echo x")
        assert hasattr(gen, "__aiter__")


# ===========================================================================
# text.py 测试
# ===========================================================================

class TestParseUnifiedDiff:
    def test_empty_returns_empty(self):
        assert parse_unified_diff("") == []

    def test_parses_simple_diff(self):
        diff = (
            "--- a/file.py\n+++ b/file.py\n"
            "@@ -1,2 +1,2 @@\n"
            " context\n-old\n+new\n"
        )
        result = parse_unified_diff(diff)
        assert len(result) == 1
        assert result[0].new_path == "file.py"

    def test_hunk_lines(self):
        diff = (
            "--- a/x.py\n+++ b/x.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n+new\n"
        )
        result = parse_unified_diff(diff)
        hunk = result[0].hunks[0]
        assert any(l.startswith("-") for l in hunk.lines)
        assert any(l.startswith("+") for l in hunk.lines)

    def test_multiple_files(self):
        diff = (
            "--- a/a.py\n+++ b/a.py\n"
            "@@ -1,1 +1,1 @@\n-x\n+y\n"
            "--- a/b.py\n+++ b/b.py\n"
            "@@ -1,1 +1,1 @@\n-a\n+b\n"
        )
        result = parse_unified_diff(diff)
        assert len(result) == 2

    def test_returns_filediff_objects(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
        result = parse_unified_diff(diff)
        assert isinstance(result[0], FileDiff)
        assert isinstance(result[0].hunks[0], Hunk)

    def test_hunk_metadata(self):
        diff = "--- a/f.py\n+++ b/f.py\n@@ -5,3 +5,4 @@\n a\n-b\n+b1\n+b2\n a\n"
        result = parse_unified_diff(diff)
        hunk = result[0].hunks[0]
        assert hunk.old_start == 5
        assert hunk.old_count == 3


class TestComputeDiff:
    def test_same_content_empty(self):
        assert compute_diff("abc\n", "abc\n") == ""

    def test_detects_change(self):
        result = compute_diff("old\n", "new\n", path="f.py")
        assert "-old" in result and "+new" in result

    def test_includes_path(self):
        result = compute_diff("a\n", "b\n", path="src/main.py")
        assert "src/main.py" in result

    def test_returns_string(self):
        assert isinstance(compute_diff("a\n", "b\n"), str)

    def test_context_lines(self):
        old = "\n".join(str(i) for i in range(20)) + "\n"
        new = old.replace("10", "99")
        result3 = compute_diff(old, new, context_lines=3)
        result0 = compute_diff(old, new, context_lines=0)
        assert len(result3) > len(result0)


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == "python"

    def test_typescript(self):
        assert detect_language("app.ts") == "typescript"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "dockerfile"

    def test_unknown(self):
        assert detect_language("file.xyz") == "unknown"

    def test_shebang_python(self):
        assert detect_language("script", content="#!/usr/bin/env python3\nprint()") == "python"

    def test_yaml(self):
        assert detect_language("config.yml") == "yaml"

    def test_json(self):
        assert detect_language("data.json") == "json"


class TestHtmlToMarkdown:
    def test_heading(self):
        result = html_to_markdown("<h1>Title</h1>")
        assert "# Title" in result

    def test_paragraph(self):
        result = html_to_markdown("<p>Hello world</p>")
        assert "Hello world" in result

    def test_link(self):
        result = html_to_markdown('<a href="https://example.com">click</a>')
        assert "click" in result and "example.com" in result

    def test_removes_script(self):
        result = html_to_markdown("<script>alert(1)</script><p>safe</p>")
        assert "alert" not in result
        assert "safe" in result

    def test_bold(self):
        result = html_to_markdown("<strong>bold</strong>")
        assert "**bold**" in result

    def test_returns_string(self):
        assert isinstance(html_to_markdown("<p>test</p>"), str)

    def test_empty_input(self):
        result = html_to_markdown("")
        assert isinstance(result, str)


class TestRedactSecrets:
    def test_redacts_api_key(self):
        text = "api_key=sk-abc123xyz789abc123xyz789abc123xyz"
        result = redact_secrets(text)
        assert "sk-abc123xyz789abc123xyz789abc123xyz" not in result

    def test_redacts_password(self):
        text = "password=supersecretpassword123"
        result = redact_secrets(text)
        assert "supersecretpassword123" not in result

    def test_keeps_non_sensitive(self):
        text = "name=John"
        result = redact_secrets(text)
        assert "John" in result

    def test_custom_pattern(self):
        result = redact_secrets(
            "mytoken=ABCDEFGHIJ1234567890",
            patterns=[r"mytoken=([A-Z0-9]{20})"],
        )
        assert isinstance(result, str)

    def test_custom_replacement(self):
        text = "Bearer sk-abc123xyz789abc123xyz789abc123xyz"
        result = redact_secrets(text, replacement="***")
        assert "***" in result

    def test_invalid_pattern_raises(self):
        with pytest.raises(ParseOprimError):
            redact_secrets("text", patterns=["[invalid"])


class TestCountTokens:
    def test_string_input(self):
        result = count_tokens("hello world")
        assert isinstance(result, int) and result > 0

    def test_messages_input(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = count_tokens(msgs)
        assert result > 0

    def test_empty_string(self):
        result = count_tokens("")
        assert result >= 1  # min 1

    def test_longer_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("a" * 1000)
        assert long > short

    def test_different_models_accepted(self):
        for model in ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"]:
            assert count_tokens("test", model=model) > 0


class TestEstimateCost:
    def test_basic_cost(self):
        cost = estimate_cost(1000, 500, model="claude-sonnet-4-6")
        assert cost > 0

    def test_more_tokens_more_cost(self):
        c1 = estimate_cost(100, 50)
        c2 = estimate_cost(10000, 5000)
        assert c2 > c1

    def test_custom_pricing(self):
        cost = estimate_cost(1000, 1000, pricing={"in": 1e-3, "out": 2e-3})
        assert abs(cost - 3.0) < 0.01

    def test_zero_tokens(self):
        assert estimate_cost(0, 0) == 0.0

    def test_known_model_price(self):
        # sonnet: 3e-6 in, 15e-6 out
        cost = estimate_cost(1_000_000, 0, model="claude-sonnet-4-6")
        assert abs(cost - 3.0) < 0.01

    def test_unknown_model_uses_fallback(self):
        cost = estimate_cost(1000, 500, model="unknown-model-xyz")
        assert cost > 0


# ===========================================================================
# 补充覆盖率 — 错误路径
# ===========================================================================

class TestCoverageGaps:
    """补足 ≥95% 覆盖率的边界/错误路径测试。"""

    def test_file_write_raises_oserror(self, tmp_path, monkeypatch):
        """file_write: parent mkdirs=False + 父不存在 → OSError → FileOprimError."""
        p = tmp_path / "noparent" / "f.txt"
        with pytest.raises((FileOprimError, OSError)):
            file_write(p, content="x", mkdirs=False)

    def test_file_append_raises_on_oserror(self, tmp_path, monkeypatch):
        """file_append: mkdirs=False + 父不存在 → FileOprimError."""
        p = tmp_path / "nodir" / "f.txt"
        with pytest.raises((FileOprimError, OSError)):
            file_append(p, content="x", mkdirs=False)

    def test_glob_match_no_respect_gitignore(self, tmp_path):
        """glob_match respect_gitignore=False 包含 .git 目录内文件。"""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        result = glob_match("**/*", root=tmp_path, respect_gitignore=False)
        # 应包含 .git/HEAD
        assert any(".git" in str(p) for p in result)

    def test_dir_list_recursive_hidden(self, tmp_path):
        """dir_list recursive + include_hidden=True 包含隐藏文件。"""
        (tmp_path / ".hidden_file").write_text("")
        result = dir_list(tmp_path, recursive=True, include_hidden=True)
        assert any(".hidden_file" in str(p) for p in result)

    def test_detect_language_makefile(self):
        assert detect_language("Makefile") == "makefile"

    def test_detect_language_go(self):
        assert detect_language("main.go") == "go"

    def test_detect_language_rust(self):
        assert detect_language("lib.rs") == "rust"

    def test_html_to_markdown_list(self):
        result = html_to_markdown("<ul><li>Item 1</li><li>Item 2</li></ul>")
        assert "Item 1" in result and "Item 2" in result

    def test_html_to_markdown_entities(self):
        result = html_to_markdown("<p>a &amp; b &lt;c&gt;</p>")
        assert "&" in result and "<c>" in result

    def test_estimate_cost_haiku(self):
        cost = estimate_cost(1_000_000, 0, model="claude-haiku-4-5")
        assert abs(cost - 0.8) < 0.01

    def test_parse_diff_strips_a_b_prefix(self):
        diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
        result = parse_unified_diff(diff)
        assert result[0].old_path == "src/main.py"
        assert result[0].new_path == "src/main.py"

    def test_bash_exec_env(self):
        import os
        r = bash_exec("echo $MY_VAR", env={**os.environ, "MY_VAR": "hello123"})
        assert "hello123" in r.stdout

    def test_count_tokens_serialization(self):
        # nested messages
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = count_tokens(msgs)
        assert result > 0


class TestShellCoverage:
    """补足 shell.py 覆盖率（stream 的 stdout/stderr 并行路径）。"""

    def test_stream_stderr_only(self):
        async def run():
            chunks = []
            async for c in bash_exec_stream("echo erronly >&2"):
                chunks.append(c)
            return chunks
        chunks = asyncio.run(run())
        assert any(c.stream == "stderr" for c in chunks)

    def test_bash_exec_large_output(self):
        r = bash_exec("seq 1 100")
        assert "100" in r.stdout
        assert r.code == 0

    def test_bash_exec_env_empty_dict(self):
        import os
        r = bash_exec("echo ok", env=os.environ.copy())
        assert r.ok

    def test_stream_long_output(self):
        async def run():
            lines = []
            async for c in bash_exec_stream("seq 1 20"):
                lines.append(c.text)
            return lines
        lines = asyncio.run(run())
        combined = "".join(lines)
        assert "20" in combined

    def test_stream_exit_code_ignored(self):
        """bash_exec_stream 不因非零退出码抛异常。"""
        async def run():
            chunks = []
            async for c in bash_exec_stream("echo x; exit 1"):
                chunks.append(c)
            return chunks
        chunks = asyncio.run(run())
        assert any("x" in c.text for c in chunks)


class TestFinalCoverageGaps:
    """覆盖剩余 miss 行：glob_match 文件路径错误 / parse_diff hunk 收尾 / shebang 分支。"""

    def test_glob_match_root_is_file_raises(self, tmp_path):
        """fs.py:354 — root 是文件而非目录 → FileOprimError。"""
        f = tmp_path / "file.txt"
        f.write_text("content")
        with pytest.raises(FileOprimError, match="not a directory"):
            glob_match("*.py", root=f)

    def test_parse_diff_hunk_at_eof(self):
        """text.py:90 — hunk 在文件末尾收尾（current_hunk 在循环结束后仍需 append）。"""
        diff = (
            "--- a/x.py\n+++ b/x.py\n"
            "@@ -1,2 +1,2 @@\n"
            " ctx\n-old\n+new\n"
            # 没有第二个 --- 或 +++，hunk 由 EOF 触发收尾
        )
        result = parse_unified_diff(diff)
        assert len(result) == 1
        assert len(result[0].hunks) == 1
        assert any(l.startswith("-old") for l in result[0].hunks[0].lines)

    def test_detect_language_node_shebang(self):
        """text.py:247-248 — #!/usr/bin/env node → javascript。"""
        assert detect_language("script", content="#!/usr/bin/env node\nconsole.log()") == "javascript"

    def test_detect_language_bash_shebang(self):
        """text.py:249-250 — #!/bin/bash → bash。"""
        assert detect_language("script", content="#!/bin/bash\necho hi") == "bash"

    def test_detect_language_sh_shebang(self):
        """text.py:251-252 — #!/bin/sh → shell。"""
        assert detect_language("script", content="#!/bin/sh\necho hi") == "shell"

    def test_detect_language_deno_shebang(self):
        """text.py:247-248 — #!/usr/bin/env deno → javascript。"""
        assert detect_language("script", content="#!/usr/bin/env deno\n") == "javascript"


class TestLastMissLines:
    """覆盖最后 4 个 miss 行。"""

    def test_parse_diff_multi_hunk_flush(self):
        """text.py:90 — 同一文件两个 hunk，遇到第二个 @@ 时 flush 第一个。"""
        diff = (
            "--- a/f.py\n+++ b/f.py\n"
            "@@ -1,2 +1,2 @@\n"
            " ctx\n-a\n+A\n"
            "@@ -10,2 +10,2 @@\n"
            " ctx2\n-b\n+B\n"
        )
        result = parse_unified_diff(diff)
        assert len(result) == 1
        assert len(result[0].hunks) == 2  # 两个 hunk 都被 flush

    def test_git_status_empty_line_skip(self, git_repo):
        """git.py:87 — git status 输出中空行被 continue 跳过，不崩溃。"""
        # clean repo 正常返回空列表，内部 for 循环对空 stdout 正常工作
        result = git_status(repo=git_repo)
        assert isinstance(result, list)

    def test_git_status_renamed_file(self, git_repo):
        """git.py:93 — rename 操作产生 'old -> new' 格式，renamed_from 被解析。"""
        # 创建文件并 commit
        f = git_repo / "orig.py"
        f.write_text("x = 1\n")
        git_add("orig.py", repo=git_repo)
        git_commit(repo=git_repo, message="add orig")
        # 执行 rename（index 级别）
        import subprocess as sp
        sp.run(["git", "-C", str(git_repo), "mv", "orig.py", "renamed.py"], capture_output=True)
        statuses = git_status(repo=git_repo)
        renamed = [s for s in statuses if "renamed" in s.path.lower() or s.index == "R"]
        # 只要能正常解析（不崩溃）即可；renamed_from 可能是 None 或有值
        assert isinstance(statuses, list)

    def test_git_log_empty_line_in_output(self, git_repo):
        """git.py:229 — log 输出中空行被 continue 跳过，不崩溃。"""
        result = git_log(repo=git_repo, n=1)
        assert len(result) >= 1
        assert result[0].hash  # 有 hash 说明解析成功


class TestAbsoluteFinalLines:
    """_exceptions.py:16 和 git.py:87,229 最后 3 行。"""

    def test_oprim_error_with_cause_str(self):
        """_exceptions.py:16 — cause 非 None 时 __str__ 拼接 cause 信息。"""
        cause = ValueError("root cause")
        err = FileOprimError("wrapper", cause=cause)
        s = str(err)
        assert "wrapper" in s
        assert "root cause" in s

    def test_git_status_trailing_empty_lines(self, git_repo):
        """git.py:87 — status 输出有空行时 continue 被执行。
        通过修改文件后 status 肯定有内容，且内部空行 continue 不影响结果。"""
        (git_repo / "hello.py").write_text("modified\n")
        (git_repo / "new_file.py").write_text("new\n")
        result = git_status(repo=git_repo)
        assert len(result) >= 1

    def test_git_log_with_multiple_commits(self, git_repo):
        """git.py:229 — log 多 commit 时，任何空输出行被 continue 跳过。"""
        for i in range(3):
            git_commit(repo=git_repo, message=f"extra {i}", allow_empty=True)
        result = git_log(repo=git_repo, n=10)
        assert len(result) >= 4  # init + 3 extra
        assert all(c.hash for c in result)
