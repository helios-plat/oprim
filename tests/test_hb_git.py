"""Tests — H-B H组: Git 原子 + 纯计算 (7 functions)."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim._hb_git import (
    FileChange,
    GitIgnorePattern,
    GitStatus,
    ProjectType,
    SnapshotId,
    StatusEntry,
    detect_project_type,
    git_current_branch,
    git_restore_snapshot,
    git_snapshot,
    parse_git_diff,
    parse_git_status,
    parse_gitignore,
)


# ---------------------------------------------------------------------------
# parse_git_status  [s] pure
# ---------------------------------------------------------------------------

def test_parse_git_status_modified() -> None:
    raw = "M  src/main.py\n"
    gs = parse_git_status(raw)
    assert len(gs.files) == 1
    assert gs.files[0].path == "src/main.py"
    assert gs.files[0].index == "M"


def test_parse_git_status_untracked() -> None:
    raw = "?? new_file.txt\n"
    gs = parse_git_status(raw)
    assert gs.files[0].index == "?"
    assert len(gs.untracked) == 1


def test_parse_git_status_deleted() -> None:
    raw = "D  deleted.py\n"
    gs = parse_git_status(raw)
    assert gs.files[0].index == "D"


def test_parse_git_status_rename() -> None:
    raw = "R  old.py -> new.py\n"
    gs = parse_git_status(raw)
    f = gs.files[0]
    assert f.index == "R"
    assert f.old_path == "old.py"
    assert f.path == "new.py"


def test_parse_git_status_clean() -> None:
    gs = parse_git_status("")
    assert gs.clean


def test_parse_git_status_mixed() -> None:
    raw = "M  a.py\nA  b.py\n?? c.py\n"
    gs = parse_git_status(raw)
    assert len(gs.files) == 3
    assert len(gs.staged) == 2
    assert len(gs.untracked) == 1


# ---------------------------------------------------------------------------
# parse_git_diff  [s] pure
# ---------------------------------------------------------------------------

_SIMPLE_DIFF = """\
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 line1
+line2_new
 line3
-line4_del
"""

def test_parse_git_diff_single_file() -> None:
    changes = parse_git_diff(_SIMPLE_DIFF)
    assert len(changes) == 1
    assert "main.py" in changes[0].new_path
    assert changes[0].additions == 1
    assert changes[0].deletions == 1


def test_parse_git_diff_empty() -> None:
    changes = parse_git_diff("")
    assert changes == []


def test_parse_git_diff_new_file() -> None:
    diff = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "+content\n"
    )
    changes = parse_git_diff(diff)
    assert changes[0].status == "A"


def test_parse_git_diff_deleted_file() -> None:
    diff = (
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
    )
    changes = parse_git_diff(diff)
    assert changes[0].status == "D"


def test_parse_git_diff_binary() -> None:
    diff = (
        "diff --git a/img.png b/img.png\n"
        "Binary files a/img.png and b/img.png differ\n"
    )
    changes = parse_git_diff(diff)
    assert changes[0].is_binary is True
    assert changes[0].status == "B"


def test_parse_git_diff_multi_file() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n+line\n"
        "diff --git a/b.py b/b.py\n+other\n"
    )
    changes = parse_git_diff(diff)
    assert len(changes) == 2


# ---------------------------------------------------------------------------
# parse_gitignore  [s] pure
# ---------------------------------------------------------------------------

def test_parse_gitignore_simple() -> None:
    patterns = parse_gitignore("*.pyc\n__pycache__/\n")
    assert patterns[0].pattern == "*.pyc"
    assert patterns[1].pattern == "__pycache__"
    assert patterns[1].dir_only is True


def test_parse_gitignore_comment_and_blank() -> None:
    patterns = parse_gitignore("# comment\n\n*.log\n")
    assert len(patterns) == 1
    assert patterns[0].pattern == "*.log"


def test_parse_gitignore_negated() -> None:
    patterns = parse_gitignore("!important.pyc\n")
    assert patterns[0].negated is True
    assert patterns[0].pattern == "important.pyc"


def test_parse_gitignore_anchored() -> None:
    patterns = parse_gitignore("/root_only.txt\n")
    assert patterns[0].anchored is True
    assert patterns[0].pattern == "root_only.txt"


def test_parse_gitignore_empty() -> None:
    assert parse_gitignore("") == []


def test_parse_gitignore_glob() -> None:
    patterns = parse_gitignore("**/*.tmp\n")
    assert patterns[0].pattern == "**/*.tmp"
    assert patterns[0].anchored is True  # contains slash


# ---------------------------------------------------------------------------
# detect_project_type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_project_type_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    pt = await detect_project_type(tmp_path)
    assert "python" in pt.languages


@pytest.mark.asyncio
async def test_detect_project_type_go(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/m\ngo 1.21\n")
    pt = await detect_project_type(tmp_path)
    assert "go" in pt.languages


@pytest.mark.asyncio
async def test_detect_project_type_unknown(tmp_path: Path) -> None:
    pt = await detect_project_type(tmp_path)
    assert pt.primary == "unknown"
    assert pt.is_monorepo is False


@pytest.mark.asyncio
async def test_detect_project_type_monorepo(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module m\ngo 1.21\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    pt = await detect_project_type(tmp_path)
    assert pt.is_monorepo is True
    assert "go" in pt.languages
    assert "python" in pt.languages


@pytest.mark.asyncio
async def test_detect_project_type_rust(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'\n")
    pt = await detect_project_type(tmp_path)
    assert "rust" in pt.languages


@pytest.mark.asyncio
async def test_detect_project_type_node(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name": "app"}')
    pt = await detect_project_type(tmp_path)
    assert "node" in pt.languages


# ---------------------------------------------------------------------------
# git_current_branch (requires obase.git.run_git — mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_git_current_branch_normal(tmp_path: Path) -> None:
    mock_result = AsyncMock()
    mock_result.ok = True
    mock_result.stdout = "main\n"

    with patch("obase.git.run_git", return_value=mock_result):
        branch = await git_current_branch(cwd=tmp_path)
    assert branch == "main"


@pytest.mark.asyncio
async def test_git_current_branch_detached(tmp_path: Path) -> None:
    empty_result = AsyncMock()
    empty_result.ok = True
    empty_result.stdout = "\n"  # detached → empty branch name

    hash_result = AsyncMock()
    hash_result.ok = True
    hash_result.stdout = "abc12345\n"

    with patch("obase.git.run_git", side_effect=[empty_result, hash_result]):
        branch = await git_current_branch(cwd=tmp_path)
    assert branch == "abc12345"


@pytest.mark.asyncio
async def test_git_current_branch_not_repo(tmp_path: Path) -> None:
    from oprim._exceptions import GitOprimError

    fail_result = AsyncMock()
    fail_result.ok = False
    fail_result.stdout = ""
    fail_result.stderr = "not a git repository"
    fail_result.returncode = 128

    with patch("obase.git.run_git", return_value=fail_result):
        with pytest.raises(GitOprimError):
            await git_current_branch(cwd=tmp_path)


# ---------------------------------------------------------------------------
# git_snapshot / git_restore_snapshot (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_git_snapshot_success(tmp_path: Path) -> None:
    stash_result = AsyncMock()
    stash_result.ok = True
    stash_result.stdout = "Saved working directory"
    stash_result.stderr = ""

    with patch("obase.git.run_git", return_value=stash_result):
        snap_id = await git_snapshot(cwd=tmp_path)
    assert snap_id.startswith("oprim-snap-")


@pytest.mark.asyncio
async def test_git_snapshot_no_changes(tmp_path: Path) -> None:
    fail_result = AsyncMock()
    fail_result.ok = False
    fail_result.stdout = ""
    fail_result.stderr = "No local changes to save"

    with patch("obase.git.run_git", return_value=fail_result):
        snap_id = await git_snapshot(cwd=tmp_path)
    assert snap_id.startswith("empty:")


@pytest.mark.asyncio
async def test_git_snapshot_fail(tmp_path: Path) -> None:
    from oprim._exceptions import GitOprimError

    fail_result = AsyncMock()
    fail_result.ok = False
    fail_result.stdout = ""
    fail_result.stderr = "fatal: unexpected error"
    fail_result.returncode = 1

    with patch("obase.git.run_git", return_value=fail_result):
        with pytest.raises(GitOprimError):
            await git_snapshot(cwd=tmp_path)


@pytest.mark.asyncio
async def test_git_restore_snapshot_empty(tmp_path: Path) -> None:
    # empty: prefix → no-op, no git calls
    with patch("obase.git.run_git") as mock_git:
        await git_restore_snapshot("empty:oprim-snap-abc", cwd=tmp_path)
        mock_git.assert_not_called()


@pytest.mark.asyncio
async def test_git_restore_snapshot_not_found(tmp_path: Path) -> None:
    list_result = AsyncMock()
    list_result.ok = True
    list_result.stdout = "stash@{0}: oprim-snap-other\n"

    with patch("obase.git.run_git", return_value=list_result):
        with pytest.raises(ValueError, match="not found"):
            await git_restore_snapshot("oprim-snap-MISSING", cwd=tmp_path)


@pytest.mark.asyncio
async def test_git_restore_snapshot_success(tmp_path: Path) -> None:
    list_result = AsyncMock()
    list_result.ok = True
    list_result.stdout = "stash@{0}: On main: oprim-snap-abc123\n"

    pop_result = AsyncMock()
    pop_result.ok = True
    pop_result.stdout = "Applied stash"
    pop_result.stderr = ""

    with patch("obase.git.run_git", side_effect=[list_result, pop_result]):
        await git_restore_snapshot("oprim-snap-abc123", cwd=tmp_path)  # no raise
