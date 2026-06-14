"""Tests for CC supplement oprim elements (P-NEW1..8)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _skill(name: str, description: str = "") -> "object":
    from oprim._hicode_types import SkillSpec

    return SkillSpec(name=name, description=description, body="")


def _tool_call(name: str) -> "object":
    from oprim._hicode_types import ToolCall

    return ToolCall(id="t1", name=name, args={})


# ── P-NEW1 match_skill_trigger ────────────────────────────────────────────────

class TestMatchSkillTrigger:
    def test_single_match_by_name(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        skills = [_skill("deploy"), _skill("test")]
        result = match_skill_trigger("please deploy my app", skills=skills)
        assert result is not None
        assert result.name == "deploy"

    def test_no_match_returns_none(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        skills = [_skill("deploy"), _skill("test")]
        result = match_skill_trigger("just do something", skills=skills)
        assert result is None

    def test_empty_skills_returns_none(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        assert match_skill_trigger("deploy app", skills=[]) is None

    def test_empty_input_returns_none(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        assert match_skill_trigger("", skills=[_skill("deploy")]) is None

    def test_case_insensitive_name_match(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        skills = [_skill("Deploy")]
        result = match_skill_trigger("DEPLOY my app", skills=skills)
        assert result is not None and result.name == "Deploy"

    def test_keyword_in_description_matches(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        skills = [_skill("ci", "build pipeline lint")]
        result = match_skill_trigger("run the pipeline", skills=skills)
        assert result is not None and result.name == "ci"

    def test_no_overlap_returns_none(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        skills = [_skill("abc", "xyz")]
        result = match_skill_trigger("no overlap at all", skills=skills)
        assert result is None

    def test_priority_longer_description_wins(self) -> None:
        from oprim._match_skill_trigger import match_skill_trigger

        short = _skill("run", "run")
        long_skill = _skill("run_tests", "run unit integration tests thoroughly")
        result = match_skill_trigger("run unit tests", skills=[short, long_skill])
        assert result is not None


# ── P-NEW2 interpolate_skill_args ─────────────────────────────────────────────

class TestInterpolateSkillArgs:
    def test_arguments_placeholder_joins_all(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        result = interpolate_skill_args("Run: $ARGUMENTS", args={"0": "foo", "1": "bar"})
        assert "foo bar" in result

    def test_positional_by_index(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        result = interpolate_skill_args("First: $0, Second: $1", args={"0": "a", "1": "b"})
        assert result == "First: a, Second: b"

    def test_named_placeholder(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        result = interpolate_skill_args("Dir: ${SKILL_DIR}", args={"SKILL_DIR": "/tmp/skill"})
        assert result == "Dir: /tmp/skill"

    def test_no_placeholder_returns_original(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        body = "No placeholders here"
        assert interpolate_skill_args(body, args={"0": "x"}) == body

    def test_missing_named_raises_value_error(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        with pytest.raises(ValueError, match="SKILL_DIR"):
            interpolate_skill_args("${SKILL_DIR}", args={})

    def test_missing_positional_raises_value_error(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        with pytest.raises(ValueError):
            interpolate_skill_args("$0 and $1", args={"0": "only_zero"})

    def test_multiple_named_all_replaced(self) -> None:
        from oprim._interpolate_skill_args import interpolate_skill_args

        result = interpolate_skill_args("${A} ${B}", args={"A": "hello", "B": "world"})
        assert result == "hello world"


# ── P-NEW3 resolve_slash_command ──────────────────────────────────────────────

class TestResolveSlashCommand:
    def test_registered_command_returns_skill_ref(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        registry = {"deploy": {"skill": "deploy"}}
        result = resolve_slash_command("/deploy", registry=registry)
        assert result is not None and result.name == "deploy"

    def test_command_with_args_splits_correctly(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        registry = {"deploy": {}}
        result = resolve_slash_command("/deploy --env prod --dry-run", registry=registry)
        assert result is not None
        assert result.args == ["--env", "prod", "--dry-run"]

    def test_non_slash_input_returns_none(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        assert resolve_slash_command("deploy", registry={"deploy": {}}) is None

    def test_unregistered_command_returns_none(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        assert resolve_slash_command("/unknown", registry={"deploy": {}}) is None

    def test_empty_slash_returns_none(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        assert resolve_slash_command("/", registry={"deploy": {}}) is None

    def test_raw_input_preserved(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        registry = {"foo": {}}
        result = resolve_slash_command("/foo bar", registry=registry)
        assert result is not None and "/foo" in result.raw_input

    def test_empty_registry_returns_none(self) -> None:
        from oprim._resolve_slash_command import resolve_slash_command

        assert resolve_slash_command("/deploy", registry={}) is None


# ── P-NEW4 parse_plugin_manifest ──────────────────────────────────────────────

class TestParsePluginManifest:
    def test_full_manifest_parsed(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        raw = json.dumps({
            "name": "p", "version": "2.0",
            "skills": [{"name": "s1"}],
            "subagents": [{"name": "ag1"}],
            "commands": [{"name": "cmd"}],
            "hooks": [{"event": "preToolUse"}],
            "mcp_defs": [{"name": "mcp1"}],
        })
        m = parse_plugin_manifest(raw)
        assert m.name == "p" and m.version == "2.0"
        assert len(m.skills) == 1

    def test_missing_name_raises(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        with pytest.raises(ValueError, match="name"):
            parse_plugin_manifest(json.dumps({"version": "1.0"}))

    def test_missing_version_raises(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        with pytest.raises(ValueError, match="version"):
            parse_plugin_manifest(json.dumps({"name": "p"}))

    def test_invalid_json_raises(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        with pytest.raises(ValueError):
            parse_plugin_manifest("{invalid json}")

    def test_empty_component_lists_valid(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        m = parse_plugin_manifest(json.dumps({"name": "p", "version": "1.0"}))
        assert m.skills == [] and m.hooks == []

    def test_non_object_json_raises(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        with pytest.raises(ValueError):
            parse_plugin_manifest("[1, 2, 3]")

    def test_version_string_preserved(self) -> None:
        from oprim._parse_plugin_manifest import parse_plugin_manifest

        m = parse_plugin_manifest(json.dumps({"name": "p", "version": "1.0.0"}))
        assert m.version == "1.0.0"


# ── P-NEW5 load_plugin_raw ────────────────────────────────────────────────────

class TestLoadPluginRaw:
    @pytest.mark.asyncio
    async def test_reads_plugin_json(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text('{"name":"p","version":"1"}')
        result = await load_plugin_raw(plugin_dir)
        assert '"name"' in result

    @pytest.mark.asyncio
    async def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        with pytest.raises(FileNotFoundError):
            await load_plugin_raw(tmp_path / "nope")

    @pytest.mark.asyncio
    async def test_no_manifest_raises(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            await load_plugin_raw(plugin_dir)

    @pytest.mark.asyncio
    async def test_empty_manifest_returns_empty_string(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        plugin_dir = tmp_path / "p"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("")
        result = await load_plugin_raw(plugin_dir)
        assert result == ""

    @pytest.mark.asyncio
    async def test_does_not_parse_json(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        plugin_dir = tmp_path / "p"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("not json at all")
        result = await load_plugin_raw(plugin_dir)
        assert result == "not json at all"

    @pytest.mark.asyncio
    async def test_large_manifest_readable(self, tmp_path: Path) -> None:
        from oprim._load_plugin_raw import load_plugin_raw

        plugin_dir = tmp_path / "p"
        plugin_dir.mkdir()
        skills = [{"x": "y"} for _ in range(1000)]
        content = json.dumps({"name": "p", "version": "1", "skills": skills})
        (plugin_dir / "plugin.json").write_text(content)
        result = await load_plugin_raw(plugin_dir)
        assert len(result) > 100


# ── P-NEW6 check_plan_mode_allowed ────────────────────────────────────────────

class TestCheckPlanModeAllowed:
    def test_plan_mode_read_tool_allowed(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("read"), mode="plan") is True

    def test_plan_mode_write_tool_blocked(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("write"), mode="plan") is False

    def test_execute_mode_all_allowed(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        for name in ("write", "bash", "edit", "delete"):
            assert check_plan_mode_allowed(_tool_call(name), mode="execute") is True

    def test_plan_mode_bash_blocked(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("bash"), mode="plan") is False

    def test_plan_mode_lsp_prefix_allowed(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("lsp_hover"), mode="plan") is True
        assert check_plan_mode_allowed(_tool_call("lsp_find_references"), mode="plan") is True

    def test_unknown_mode_read_allowed(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("grep"), mode="unknown") is True

    def test_unknown_mode_write_blocked(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("edit"), mode="unknown") is False

    def test_plan_mode_glob_allowed(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("glob"), mode="plan") is True

    def test_plan_mode_unknown_tool_blocked(self) -> None:
        from oprim._check_plan_mode_allowed import check_plan_mode_allowed

        assert check_plan_mode_allowed(_tool_call("mystery_tool"), mode="plan") is False


# ── P-NEW7 make_checkpoint ────────────────────────────────────────────────────

class TestMakeCheckpoint:
    def _state(self, **kw: object) -> "object":
        from oprim._cc_types import RunState

        defaults: dict = {"step": 0, "data": {}, "completed_steps": []}
        defaults.update(kw)
        return RunState(session_id="sess-1", **defaults)  # type: ignore[arg-type]

    def test_normal_serialization(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        cp = make_checkpoint(self._state(step=3, data={"x": 1}), session_id="s1")
        assert cp.session_id == "s1"
        assert cp.payload["step"] == 3
        assert cp.payload["data"] == {"x": 1}

    def test_empty_state_produces_checkpoint(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        cp = make_checkpoint(self._state(), session_id="s1")
        assert cp.session_id == "s1" and cp.version == "1"

    def test_empty_session_id_raises(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        with pytest.raises(ValueError, match="session_id"):
            make_checkpoint(self._state(), session_id="")

    def test_non_serializable_raises(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        state = self._state(data={"fn": lambda: None})
        with pytest.raises(ValueError, match="non-serializable"):
            make_checkpoint(state, session_id="s1")

    def test_timestamp_in_checkpoint(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        cp = make_checkpoint(self._state(), session_id="s1")
        assert cp.timestamp and "Z" in cp.timestamp

    def test_completed_steps_preserved(self) -> None:
        from oprim._make_checkpoint import make_checkpoint

        state = self._state(completed_steps=["step_a", "step_b"])
        cp = make_checkpoint(state, session_id="s1")
        assert cp.payload["completed_steps"] == ["step_a", "step_b"]

    def test_no_file_written(self, tmp_path: Path) -> None:
        import os

        from oprim._make_checkpoint import make_checkpoint

        before = set(os.listdir(tmp_path))
        make_checkpoint(self._state(), session_id="s1")
        after = set(os.listdir(tmp_path))
        assert before == after


# ── P-NEW8 restore_from_checkpoint ───────────────────────────────────────────

class TestRestoreFromCheckpoint:
    def _checkpoint(self, **kw: object) -> "object":
        from oprim._cc_types import CheckpointData

        base = {
            "session_id": "s1",
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "1",
            "payload": {
                "step": 5,
                "data": {"key": "val"},
                "completed_steps": ["a"],
                "state_session_id": "orig-sess",
            },
        }
        base.update(kw)  # type: ignore[arg-type]
        return CheckpointData(**base)  # type: ignore[arg-type]

    def test_normal_restore(self) -> None:
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        state = restore_from_checkpoint(self._checkpoint())
        assert state.step == 5
        assert state.data == {"key": "val"}
        assert state.session_id == "orig-sess"

    def test_roundtrip_with_make_checkpoint(self) -> None:
        from oprim._cc_types import RunState
        from oprim._make_checkpoint import make_checkpoint
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        original = RunState(session_id="sess-x", step=7, data={"n": 42}, completed_steps=["s1"])
        cp = make_checkpoint(original, session_id="sess-x")
        restored = restore_from_checkpoint(cp)
        assert restored.step == original.step
        assert restored.data == original.data
        assert restored.completed_steps == original.completed_steps

    def test_invalid_version_raises(self) -> None:
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        with pytest.raises(ValueError, match="version"):
            restore_from_checkpoint(self._checkpoint(version="99"))

    def test_empty_session_id_raises(self) -> None:
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        with pytest.raises(ValueError):
            restore_from_checkpoint(self._checkpoint(session_id=""))

    def test_missing_payload_field_raises(self) -> None:
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        bad_payload = {"step": 1}
        with pytest.raises(ValueError, match="missing"):
            restore_from_checkpoint(self._checkpoint(payload=bad_payload))

    def test_completed_steps_restored(self) -> None:
        from oprim._cc_types import CheckpointData
        from oprim._restore_from_checkpoint import restore_from_checkpoint

        cp = CheckpointData(
            session_id="s1",
            timestamp="2025-01-01T00:00:00Z",
            version="1",
            payload={
                "step": 0,
                "data": {},
                "completed_steps": ["x", "y", "z"],
                "state_session_id": "s1",
            },
        )
        state = restore_from_checkpoint(cp)
        assert state.completed_steps == ["x", "y", "z"]

    def test_no_file_read(self, tmp_path: Path) -> None:
        import os

        from oprim._restore_from_checkpoint import restore_from_checkpoint

        before = set(os.listdir(tmp_path))
        restore_from_checkpoint(self._checkpoint())
        after = set(os.listdir(tmp_path))
        assert before == after
