"""Tests for engineering-gate oprims."""

from __future__ import annotations

from pathlib import Path

from oprim._archive_note import archive_note
from oprim._capture_gui_clip import capture_gui_clip
from oprim._diff_since import diff_since
from oprim._read_standards import BUILTIN_STANDARDS, read_standards
from oprim._run_targeted_checks import run_targeted_checks
from oprim._write_artifact import write_artifact, write_proposal


def test_read_standards_builtin_when_missing(tmp_path: Path) -> None:
    result = read_standards(project_root=tmp_path)
    assert result["ok"] is True
    assert result["standards_source"] == "builtin"
    assert "Security" in result["text"]
    assert result["check_map_source"] == "builtin"
    assert result["text"] == BUILTIN_STANDARDS


def test_read_standards_project_file(tmp_path: Path) -> None:
    dest = tmp_path / ".veya-project" / "engineering"
    dest.mkdir(parents=True)
    (dest / "STANDARDS.md").write_text("# Project rules\n- no eval\n", encoding="utf-8")
    result = read_standards(project_root=tmp_path)
    assert result["standards_source"] == "project"
    assert "no eval" in result["text"]


def test_write_artifact_jails_to_engineering(tmp_path: Path) -> None:
    ok = write_artifact(project_root=tmp_path, relpath="reviews/a.md", content="hi", kind="review")
    assert ok["ok"] is True
    assert (tmp_path / ".veya-project" / "engineering" / "reviews" / "a.md").read_text() == "hi"

    bad = write_artifact(project_root=tmp_path, relpath="../../secret.py", content="x")
    assert bad["ok"] is False
    assert "escapes" in bad["error"]
    assert not (tmp_path / "secret.py").exists()


def test_write_proposal_stays_under_proposals(tmp_path: Path) -> None:
    rec = write_proposal(
        project_root=tmp_path, title="Drop extra wrapper", body="one helper is enough"
    )
    assert rec["ok"] is True
    path = Path(rec["path"])
    assert path.is_file()
    assert ".veya-project/engineering/proposals/" in str(path).replace("\\", "/")
    assert "Drop extra wrapper" in path.read_text(encoding="utf-8")


def test_archive_note_promote_and_suppress(tmp_path: Path) -> None:
    inbox = tmp_path / ".veya-project" / "engineering" / "notes-inbox"
    inbox.mkdir(parents=True)
    src = inbox / "gotcha.md"
    src.write_text("never reboot the host", encoding="utf-8")
    promo = archive_note(
        project_root=tmp_path,
        title="host reboot",
        body="never reboot the host",
        decision="promote",
        reason="durable constraint",
        source_path=str(src),
    )
    assert promo["ok"] is True
    assert "archive/" in promo["path"].replace("\\", "/")
    assert "suppressed" not in promo["path"]
    assert not src.exists()

    quiet = archive_note(
        project_root=tmp_path,
        title="wip",
        body="today i poked around",
        decision="suppress",
        reason="noise",
    )
    assert quiet["ok"] is True
    assert "suppressed" in quiet["path"].replace("\\", "/")


def test_run_targeted_checks_skips_empty_and_md_only(tmp_path: Path) -> None:
    empty = run_targeted_checks(project_root=tmp_path, files=[])
    assert empty["ok"] is True
    assert empty["skipped"] is True
    assert empty["reason"] == "no changes"

    md = run_targeted_checks(project_root=tmp_path, files=["README.md", "docs/a.md"])
    assert md["ok"] is True
    assert md["skipped"] is True


def test_run_targeted_checks_pytest_is_path_scoped(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x = 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_foo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    seen: list[list[str]] = []

    def runner(cmd, cwd=None, timeout=None):
        seen.append(list(cmd))
        return {"code": 0, "stdout": "1 passed", "stderr": "", "ran": True, "cmd": list(cmd)}

    rec = run_targeted_checks(
        project_root=tmp_path,
        files=["foo.py"],
        runner=runner,
    )
    assert rec["ok"] is True
    pytest_cmds = [c for c in seen if "pytest" in c]
    assert pytest_cmds, rec
    cmd = pytest_cmds[0]
    assert "tests/test_foo.py" in cmd
    assert cmd[-1] != "pytest"


def test_run_targeted_checks_force_full_allows_bare_pytest(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def runner(cmd, cwd=None, timeout=None):
        seen.append(list(cmd))
        return {"code": 0, "stdout": "1 passed", "stderr": "", "ran": True, "cmd": list(cmd)}

    rec = run_targeted_checks(
        project_root=tmp_path, files=["foo.py"], force_full=True, runner=runner
    )
    assert rec["ok"] is True
    pytest_cmds = [c for c in seen if "pytest" in c]
    assert pytest_cmds
    assert pytest_cmds[0][-2:] == ["pytest", "-q"] or pytest_cmds[0][-1] == "-q"


def test_capture_gui_clip_does_not_fabricate_without_playwright(tmp_path: Path) -> None:
    dest = tmp_path / "clip.gif"
    rec = capture_gui_clip(output_path=dest, url="http://127.0.0.1:9/")
    assert rec["ok"] is False
    assert "will not fabricate" in rec["reason"]
    assert not dest.exists()


def test_diff_since_missing_repo(tmp_path: Path) -> None:
    rec = diff_since(repo=tmp_path / "nope", since_ref="HEAD")
    assert rec["ok"] is False
    assert rec["changed"] == []
