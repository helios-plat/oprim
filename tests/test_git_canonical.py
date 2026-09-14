from __future__ import annotations

from types import SimpleNamespace

import pytest

import oprim.git as git


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("git_status", ["status", "--porcelain=v1", "-u"]),
        ("git_diff", ["diff", "-U3"]),
        ("git_show", ["show", "HEAD"]),
    ],
)
def test_git_read_primitives_delegate_to_obase(monkeypatch, tmp_path, name, args):
    calls: list[tuple[list[str], str | None]] = []

    async def fake_run(actual, *, cwd, input_text=None, **_):
        calls.append((actual, input_text))
        output = "M  changed.py\n" if name == "git_status" else "atomic output"
        return SimpleNamespace(ok=True, stdout=output, stderr="", returncode=0)

    monkeypatch.setattr(git, "run_git", fake_run)
    kwargs = {"ref": "HEAD"} if name == "git_show" else {}
    result = getattr(git, name)(repo=tmp_path, **kwargs)

    assert calls == [(args, None)]
    if name == "git_status":
        assert result[0].path == "changed.py"
        assert result[0].index == "M"


def test_git_apply_passes_patch_as_stdin(monkeypatch, tmp_path):
    calls = []

    async def fake_run(actual, *, cwd, input_text=None, **_):
        calls.append((actual, input_text))
        return SimpleNamespace(ok=True, stdout="", stderr="", returncode=0)

    monkeypatch.setattr(git, "run_git", fake_run)
    git.git_apply("diff --git a/a b/a\n", repo=tmp_path, check=True)

    assert calls == [(["apply", "--check"], "diff --git a/a b/a\n")]


def test_git_failure_is_normalized(monkeypatch, tmp_path):
    async def fake_run(*_, **__):
        return SimpleNamespace(ok=False, stdout="", stderr="bad patch", returncode=128)

    monkeypatch.setattr(git, "run_git", fake_run)
    with pytest.raises(git.GitOprimError, match="git show failed"):
        git.git_show("HEAD", repo=tmp_path)
