"""Tests for Veya P1/P2 顶级 Agent 功能复现 — oprim 层 (12 个).

Covers: agent_prompt_synthesize, git_worktree_merge, tmux_pane_create,
kanban_task_update, stt_transcribe_stream, tts_synthesize_stream,
frontend_tool_forward, soul_config_rewrite, replay_step_record,
media_content_parse, media_publish_post, support_bundle_pack.
"""

from __future__ import annotations

import pathlib

import pytest

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    async def __call__(self, *, messages, tools=None, max_tokens=4096, system=None):
        return {
            "content": [{"type": "text", "text": "你是数据分析专家"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "stop_reason": "end_turn",
        }


class FakeTranscriber:
    async def transcribe(self, audio, *, language=None):
        return {"text": "你好世界", "segments": [{"start": 0, "end": 1}]}


class FakeSynthesizer:
    async def synthesize_stream(self, text, *, voice=None):
        for _ in range(3):
            yield b"\x00" * 320


class FakeGateway:
    async def forward(self, tool_name, payload):
        return {"ok": True}


class FakePublisher:
    async def publish(self, channel, *, content, media_paths):
        return {"post_id": "p-1"}


class FakeKanban:
    def __init__(self):
        self.tasks = {"t1": {"status": "todo"}}

    async def get(self, task_id):
        return self.tasks.get(task_id)

    async def update(self, task_id, fields):
        self.tasks[task_id].update(fields)
        return self.tasks[task_id]


# ---------------------------------------------------------------------------
# agent_prompt_synthesize
# ---------------------------------------------------------------------------


class TestAgentPromptSynthesize:
    async def test_synthesize(self):
        from oprim import agent_prompt_synthesize

        r = await agent_prompt_synthesize(
            "数据分析 agent", caller=FakeLLM(), capabilities=["chart"]
        )
        assert r["status"] == "ok"
        assert r["prompt"] == "你是数据分析专家"
        assert r["usage"]["input_tokens"] == 5

    async def test_validation(self):
        from oprim import OprimValidationError, agent_prompt_synthesize

        with pytest.raises(OprimValidationError):
            await agent_prompt_synthesize("", caller=FakeLLM())
        with pytest.raises(OprimValidationError):
            await agent_prompt_synthesize("x", caller=None)


# ---------------------------------------------------------------------------
# git_worktree_merge
# ---------------------------------------------------------------------------


class TestGitWorktreeMerge:
    async def test_fast_forward_merge(self, tmp_path: pathlib.Path):
        from oprim import git_worktree_merge
        from oprim.git import _git

        repo = tmp_path / "repo"
        repo.mkdir()
        _git("init", "-b", "main", repo=repo)
        (repo / "f.txt").write_text("base")
        _git("add", ".", repo=repo)
        _git("commit", "-m", "init", repo=repo)
        _git("checkout", "-b", "feat/x", repo=repo)
        (repo / "f.txt").write_text("feat")
        _git("commit", "-am", "feat", repo=repo)
        _git("checkout", "main", repo=repo)

        r = await git_worktree_merge("feat/x", repo=repo, target="main")
        assert r["status"] == "merged" and r["merged"] is True

    async def test_conflict_detected(self, tmp_path: pathlib.Path):
        from oprim import git_worktree_merge
        from oprim.git import _git

        repo = tmp_path / "repo2"
        repo.mkdir()
        _git("init", "-b", "main", repo=repo)
        (repo / "f.txt").write_text("base\n")
        _git("add", ".", repo=repo)
        _git("commit", "-m", "init", repo=repo)
        _git("checkout", "-b", "feat/y", repo=repo)
        (repo / "f.txt").write_text("theirs\n")
        _git("commit", "-am", "theirs", repo=repo)
        _git("checkout", "main", repo=repo)
        (repo / "f.txt").write_text("ours\n")
        _git("commit", "-am", "ours", repo=repo)

        r = await git_worktree_merge("feat/y", repo=repo, target="main")
        assert r["status"] == "conflict" and r["merged"] is False
        assert any("CONFLICT" in c for c in r["conflicts"])

    async def test_empty_branch(self, tmp_path: pathlib.Path):
        from oprim import OprimValidationError, git_worktree_merge

        with pytest.raises(OprimValidationError):
            await git_worktree_merge("", repo=tmp_path)


# ---------------------------------------------------------------------------
# tmux_pane_create
# ---------------------------------------------------------------------------


class TestTmuxPaneCreate:
    async def test_missing_binary(self):
        import shutil

        if shutil.which("tmux") is not None:
            pytest.skip("tmux installed")
        from oprim import TmuxPaneError, tmux_pane_create

        with pytest.raises(TmuxPaneError):
            tmux_pane_create("session-x")

    async def test_empty_session(self):
        from oprim import OprimValidationError, tmux_pane_create

        with pytest.raises(OprimValidationError):
            tmux_pane_create("")


# ---------------------------------------------------------------------------
# kanban_task_update
# ---------------------------------------------------------------------------


class TestKanbanTaskUpdate:
    async def test_update(self):
        from oprim import kanban_task_update

        r = await kanban_task_update(
            "t1", store=FakeKanban(), status="doing", assignee="alice"
        )
        assert r["status"] == "ok"
        assert r["updated"]["status"] == "doing"
        assert r["updated"]["assignee"] == "alice"

    async def test_invalid_status(self):
        from oprim import OprimValidationError, kanban_task_update

        with pytest.raises(OprimValidationError):
            await kanban_task_update("t1", store=FakeKanban(), status="bogus")

    async def test_missing_task(self):
        from oprim import KanbanUpdateError, kanban_task_update

        with pytest.raises(KanbanUpdateError):
            await kanban_task_update("nope", store=FakeKanban(), status="doing")

    async def test_no_fields(self):
        from oprim import OprimValidationError, kanban_task_update

        with pytest.raises(OprimValidationError):
            await kanban_task_update("t1", store=FakeKanban())


# ---------------------------------------------------------------------------
# stt_transcribe_stream / tts_synthesize_stream
# ---------------------------------------------------------------------------


class TestVoiceStreamAtoms:
    async def test_stt(self, tmp_path: pathlib.Path):
        from oprim import stt_transcribe_stream

        r = await stt_transcribe_stream(
            str(tmp_path / "a.wav"), transcriber=FakeTranscriber()
        )
        assert r["text"] == "你好世界" and len(r["segments"]) == 1

    async def test_stt_frames(self):
        from oprim import stt_transcribe_stream

        r = await stt_transcribe_stream(
            [b"\x00" * 320] * 4, transcriber=FakeTranscriber(), sample_rate=16000
        )
        assert r["frames"] == 4

    async def test_tts_stream(self):
        from oprim import tts_synthesize_stream

        r = await tts_synthesize_stream("你好", synthesizer=FakeSynthesizer())
        assert r["status"] == "ok"
        assert r["chunks"] == 3 and r["bytes"] == 960

    async def test_validation(self):
        from oprim import OprimValidationError, stt_transcribe_stream, tts_synthesize_stream

        with pytest.raises(OprimValidationError):
            await stt_transcribe_stream([], transcriber=FakeTranscriber())
        with pytest.raises(OprimValidationError):
            await tts_synthesize_stream("", synthesizer=FakeSynthesizer())


# ---------------------------------------------------------------------------
# frontend_tool_forward
# ---------------------------------------------------------------------------


class TestFrontendToolForward:
    async def test_forward(self):
        from oprim import frontend_tool_forward

        r = await frontend_tool_forward("notify", payload={"m": 1}, gateway=FakeGateway())
        assert r["delivered"] is True and r["response"]["ok"] is True

    async def test_timeout(self):
        import asyncio

        from oprim import FrontendForwardError, frontend_tool_forward

        class SlowGateway:
            async def forward(self, tool_name, payload):
                await asyncio.sleep(5)

        with pytest.raises(FrontendForwardError):
            await frontend_tool_forward("x", payload={}, gateway=SlowGateway(), timeout=0.05)


# ---------------------------------------------------------------------------
# soul_config_rewrite
# ---------------------------------------------------------------------------


class TestSoulConfigRewrite:
    async def test_rewrite_with_versioning(self, tmp_path: pathlib.Path):
        from obase import VersionStore

        from oprim import soul_config_rewrite

        vs = VersionStore(tmp_path / "vs")
        cfg = tmp_path / "soul.md"
        cfg.write_text("old")
        r = await soul_config_rewrite(
            cfg, content="new", sandbox_root=tmp_path, version_store=vs
        )
        assert r["status"] == "ok"
        assert cfg.read_text() == "new"
        assert r["rev_before"] and r["rev_after"]
        assert len(vs.list_versions()) == 2

    async def test_path_escape(self, tmp_path: pathlib.Path):
        from oprim import PathSecurityError, soul_config_rewrite

        with pytest.raises(PathSecurityError):
            await soul_config_rewrite("/etc/evil", content="x", sandbox_root=tmp_path)

    async def test_atomic_no_partial(self, tmp_path: pathlib.Path):
        from oprim import soul_config_rewrite

        cfg = tmp_path / "s.md"
        cfg.write_text("v1")
        r = await soul_config_rewrite(cfg, content="v2-long-content")
        assert r["bytes_written"] == len("v2-long-content")
        assert cfg.read_text() == "v2-long-content"


# ---------------------------------------------------------------------------
# replay_step_record
# ---------------------------------------------------------------------------


class TestReplayStepRecord:
    def test_record_to_jsonl(self, tmp_path: pathlib.Path):
        from oprim import replay_step_record

        log = tmp_path / "replay.jsonl"
        r = replay_step_record(1, run_id="run1", payload={"a": 1}, log_path=log)
        assert r["recorded"] is True and pathlib.Path(r["path"]).exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1 and '"run_id": "run1"' in lines[0]

    def test_recorder_injected(self):
        from oprim import replay_step_record

        captured = []

        class Rec:
            def append(self, entry):
                captured.append(entry)
                return True

        r = replay_step_record(2, run_id="r", payload={}, recorder=Rec())
        assert r["recorded"] is True and captured[0]["step"] == 2

    def test_requires_persistence(self):
        from oprim import ReplayRecordError, replay_step_record

        with pytest.raises(ReplayRecordError):
            replay_step_record(0, run_id="r", payload={})


# ---------------------------------------------------------------------------
# media_content_parse / media_publish_post / support_bundle_pack
# ---------------------------------------------------------------------------


class TestMediaAtoms:
    async def test_parse(self, tmp_path: pathlib.Path):
        from oprim import media_content_parse

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"jpg")
        r = await media_content_parse(img)
        assert r["kind"] == "image" and r["mime"] == "image/jpeg"
        assert r["size_bytes"] == 3

    async def test_parse_missing(self, tmp_path: pathlib.Path):
        from oprim import MediaParseError, media_content_parse

        with pytest.raises(MediaParseError):
            await media_content_parse(tmp_path / "nope.jpg")

    async def test_publish(self, tmp_path: pathlib.Path):
        from oprim import media_publish_post

        img = tmp_path / "p.jpg"
        img.write_bytes(b"jpg")
        r = await media_publish_post(
            "x", content="hi", media_paths=[str(img)], publisher=FakePublisher()
        )
        assert r["post_id"] == "p-1" and r["media_count"] == 1

    async def test_publish_missing_media(self):
        from oprim import MediaPublishError, media_publish_post

        with pytest.raises(MediaPublishError):
            await media_publish_post(
                "x", content="hi", media_paths=["/nope.jpg"],
                publisher=FakePublisher(),
            )

    async def test_bundle_pack(self, tmp_path: pathlib.Path):
        from oprim import support_bundle_pack

        f1 = tmp_path / "a.log"
        f1.write_text("log")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("data")
        r = await support_bundle_pack(
            "b1", paths=[str(f1), str(sub)], output_dir=str(tmp_path / "out")
        )
        assert r["entries"] == 2
        assert pathlib.Path(r["archive"]).exists()

    async def test_bundle_max_bytes(self, tmp_path: pathlib.Path):
        from oprim import SupportBundleError, support_bundle_pack

        big = tmp_path / "big.bin"
        big.write_bytes(b"\x00" * 1024)
        with pytest.raises(SupportBundleError):
            await support_bundle_pack("b2", paths=[str(big)], max_bytes=100)
