"""tests/test_batch_hevi_f11.py — hevi Phase 10/11 oprim batch tests."""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_caller(json_text: str):
    """Return an async caller that always replies with json_text."""
    async def caller(**kwargs):
        return {
            "content": [{"type": "text", "text": json_text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    return caller


def make_error_caller(exc: Exception):
    """Return an async caller that always raises exc."""
    async def caller(**kwargs):
        raise exc
    return caller


# ---------------------------------------------------------------------------
# 1. storyboard_grid
# ---------------------------------------------------------------------------

from oprim.storyboard_grid import StoryboardGridResult, storyboard_grid

_GRID_JSON = json.dumps({
    "shots": [
        {"index": 0, "description": "Wide shot of city", "duration_s": 3.0, "camera_angle": "wide"},
        {"index": 1, "description": "Close up on hero", "duration_s": 2.5, "camera_angle": "close-up"},
    ],
    "total_duration_s": 5.5,
})


def test_storyboard_grid_returns_result():
    result = asyncio.run(storyboard_grid("A hero walks into town.", caller=make_caller(_GRID_JSON)))
    assert isinstance(result, StoryboardGridResult)


def test_storyboard_grid_shot_count():
    result = asyncio.run(storyboard_grid("A hero walks into town.", caller=make_caller(_GRID_JSON)))
    assert len(result.shots) == 2


def test_storyboard_grid_total_duration():
    result = asyncio.run(storyboard_grid("A hero walks into town.", caller=make_caller(_GRID_JSON)))
    assert result.total_duration_s == pytest.approx(5.5)


def test_storyboard_grid_shot_fields():
    result = asyncio.run(storyboard_grid("A hero walks into town.", caller=make_caller(_GRID_JSON)))
    shot = result.shots[0]
    assert shot["camera_angle"] == "wide"
    assert shot["description"] == "Wide shot of city"


def test_storyboard_grid_grid_description_nonempty():
    result = asyncio.run(storyboard_grid("A hero walks into town.", caller=make_caller(_GRID_JSON)))
    assert result.grid_description != ""


def test_storyboard_grid_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(storyboard_grid("script", caller=make_caller("not json")))


def test_storyboard_grid_empty_shots():
    payload = json.dumps({"shots": [], "total_duration_s": 0.0})
    result = asyncio.run(storyboard_grid("empty script", caller=make_caller(payload)))
    assert result.shots == []
    assert result.total_duration_s == 0.0


# ---------------------------------------------------------------------------
# 2. multi_angle
# ---------------------------------------------------------------------------

from oprim.multi_angle import MultiAngleResult, multi_angle

_ANGLE_JSON = json.dumps({
    "angle_prompts": {
        "front": "front view of a warrior",
        "side": "side view of a warrior",
        "back": "back view of a warrior",
        "three-quarter": "three-quarter view of a warrior",
    }
})


def test_multi_angle_returns_result():
    result = asyncio.run(multi_angle("a warrior in armour", caller=make_caller(_ANGLE_JSON)))
    assert isinstance(result, MultiAngleResult)


def test_multi_angle_angle_prompts_keys():
    result = asyncio.run(multi_angle("a warrior in armour", caller=make_caller(_ANGLE_JSON)))
    assert set(result.angle_prompts.keys()) == {"front", "side", "back", "three-quarter"}


def test_multi_angle_subject_description_preserved():
    result = asyncio.run(multi_angle("a warrior in armour", caller=make_caller(_ANGLE_JSON)))
    assert result.subject_description == "a warrior in armour"


def test_multi_angle_custom_angles():
    payload = json.dumps({"angle_prompts": {"top": "top view"}})
    result = asyncio.run(
        multi_angle("a car", caller=make_caller(payload), angles=["top"])
    )
    assert "top" in result.angle_prompts


def test_multi_angle_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        asyncio.run(multi_angle("subject", caller=make_caller("bad")))


def test_multi_angle_prompts_are_strings():
    result = asyncio.run(multi_angle("a warrior in armour", caller=make_caller(_ANGLE_JSON)))
    for v in result.angle_prompts.values():
        assert isinstance(v, str)


def test_multi_angle_nonempty_prompts():
    result = asyncio.run(multi_angle("a warrior in armour", caller=make_caller(_ANGLE_JSON)))
    for v in result.angle_prompts.values():
        assert len(v) > 0


# ---------------------------------------------------------------------------
# 3. inject_visual_style  (SYNC)
# ---------------------------------------------------------------------------

from oprim.inject_visual_style import inject_visual_style


def test_inject_visual_style_is_sync():
    assert not inspect.iscoroutinefunction(inject_visual_style)


def test_inject_visual_style_all_none_returns_original():
    prompt = "a cat sitting on a roof"
    result = inject_visual_style(prompt)
    assert result == prompt


def test_inject_visual_style_style_only():
    result = inject_visual_style("a cat", style="anime")
    assert result == "a cat, anime style"


def test_inject_visual_style_combo():
    result = inject_visual_style("a cat", style="anime", lighting="sunset")
    assert "anime style" in result
    assert "sunset lighting" in result


def test_inject_visual_style_all_params():
    result = inject_visual_style(
        "a cat",
        style="photorealistic",
        lighting="golden hour",
        color_grade="warm tones",
        camera="macro",
    )
    assert "photorealistic style" in result
    assert "golden hour lighting" in result
    assert "warm tones color grade" in result
    assert "macro camera" in result


def test_inject_visual_style_starts_with_prompt():
    result = inject_visual_style("a dog", style="oil painting")
    assert result.startswith("a dog")


def test_inject_visual_style_camera_only():
    result = inject_visual_style("a landscape", camera="drone")
    assert result == "a landscape, drone camera"


# ---------------------------------------------------------------------------
# 4. video_element_edit
# ---------------------------------------------------------------------------

from oprim.video_element_edit import VideoEditOperation, video_element_edit

_ELEMENTS = [
    {"id": "e0", "text": "Hello"},
    {"id": "e1", "text": "World"},
    {"id": "e2", "text": "Bye"},
]

async def _noop_caller(**kwargs):
    return {}


def test_video_element_edit_replace_no_replacement_raises():
    with pytest.raises(ValueError):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="replace",
                target_index=0,
                replacement=None,
                caller=_noop_caller,
            )
        )


def test_video_element_edit_replace_empty_dict_raises():
    with pytest.raises(ValueError):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="replace",
                target_index=0,
                replacement={},
                caller=_noop_caller,
            )
        )


def test_video_element_edit_replace_success():
    new_el = {"id": "e0", "text": "Replaced"}
    result = asyncio.run(
        video_element_edit(
            elements=list(_ELEMENTS),
            operation="replace",
            target_index=0,
            replacement=new_el,
            caller=_noop_caller,
        )
    )
    assert result[0]["text"] == "Replaced"
    assert len(result) == 3


def test_video_element_edit_replace_out_of_range_raises():
    with pytest.raises(IndexError):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="replace",
                target_index=99,
                replacement={"id": "x"},
                caller=_noop_caller,
            )
        )


def test_video_element_edit_insert():
    new_el = {"id": "eX", "text": "Inserted"}
    result = asyncio.run(
        video_element_edit(
            elements=list(_ELEMENTS),
            operation="insert",
            target_index=1,
            replacement=new_el,
            caller=_noop_caller,
        )
    )
    assert len(result) == 4
    assert result[1]["text"] == "Inserted"


def test_video_element_edit_insert_no_replacement_raises():
    with pytest.raises(ValueError):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="insert",
                target_index=1,
                replacement=None,
                caller=_noop_caller,
            )
        )


def test_video_element_edit_delete():
    result = asyncio.run(
        video_element_edit(
            elements=list(_ELEMENTS),
            operation="delete",
            target_index=1,
            caller=_noop_caller,
        )
    )
    assert len(result) == 2
    assert result[0]["text"] == "Hello"
    assert result[1]["text"] == "Bye"


def test_video_element_edit_delete_out_of_range_raises():
    with pytest.raises(IndexError):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="delete",
                target_index=50,
                caller=_noop_caller,
            )
        )


def test_video_element_edit_unknown_operation_raises():
    with pytest.raises(ValueError, match="unknown operation"):
        asyncio.run(
            video_element_edit(
                elements=list(_ELEMENTS),
                operation="flip",
                target_index=0,
                caller=_noop_caller,
            )
        )


def test_video_element_edit_original_not_mutated():
    original = [{"id": "e0", "text": "Hello"}, {"id": "e1", "text": "World"}]
    copy = [dict(e) for e in original]
    asyncio.run(
        video_element_edit(
            elements=original,
            operation="replace",
            target_index=0,
            replacement={"id": "e0", "text": "New"},
            caller=_noop_caller,
        )
    )
    assert original == copy


# ---------------------------------------------------------------------------
# 5. subject_create / retrieve / update
# ---------------------------------------------------------------------------

from oprim._hevi_types import Subject
from oprim.subject_create import subject_create
from oprim.subject_retrieve import subject_retrieve
from oprim.subject_update import subject_update


def _make_subject(**overrides) -> Subject:
    defaults = dict(subject_id="s-001", name="Hero", description="A brave hero")
    defaults.update(overrides)
    return Subject(**defaults)


def test_subject_create_returns_subject():
    store: dict = {}
    s = _make_subject()
    result = asyncio.run(subject_create(s, store=store))
    assert isinstance(result, Subject)
    assert result.subject_id == "s-001"


def test_subject_create_persisted_in_store():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    assert "s-001" in store


def test_subject_create_multiple():
    store: dict = {}
    s1 = _make_subject(subject_id="s-001", name="Hero")
    s2 = _make_subject(subject_id="s-002", name="Villain")
    asyncio.run(subject_create(s1, store=store))
    asyncio.run(subject_create(s2, store=store))
    assert len(store) == 2


def test_subject_create_overwrites_same_id():
    store: dict = {}
    s1 = _make_subject(name="Old")
    s2 = _make_subject(name="New")
    asyncio.run(subject_create(s1, store=store))
    asyncio.run(subject_create(s2, store=store))
    assert store["s-001"].name == "New"


def test_subject_retrieve_found():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    result = asyncio.run(subject_retrieve("s-001", store=store))
    assert result is not None
    assert result.name == "Hero"


def test_subject_retrieve_not_found_returns_none():
    store: dict = {}
    result = asyncio.run(subject_retrieve("missing-id", store=store))
    assert result is None


def test_subject_retrieve_returns_subject_type():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    result = asyncio.run(subject_retrieve("s-001", store=store))
    assert isinstance(result, Subject)


def test_subject_retrieve_empty_store():
    result = asyncio.run(subject_retrieve("any-id", store={}))
    assert result is None


def test_subject_update_returns_updated():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    updated = asyncio.run(subject_update("s-001", {"name": "Updated Hero"}, store=store))
    assert updated is not None
    assert updated.name == "Updated Hero"


def test_subject_update_increments_version():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    updated = asyncio.run(subject_update("s-001", {"description": "changed"}, store=store))
    assert updated.version == 2


def test_subject_update_not_found_returns_none():
    store: dict = {}
    result = asyncio.run(subject_update("ghost", {"name": "X"}, store=store))
    assert result is None


def test_subject_update_persisted_in_store():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    asyncio.run(subject_update("s-001", {"tags": ["hero", "main"]}, store=store))
    assert store["s-001"].tags == ["hero", "main"]


def test_subject_update_double_increment():
    store: dict = {}
    s = _make_subject()
    asyncio.run(subject_create(s, store=store))
    asyncio.run(subject_update("s-001", {"name": "v2"}, store=store))
    asyncio.run(subject_update("s-001", {"name": "v3"}, store=store))
    assert store["s-001"].version == 3


# ---------------------------------------------------------------------------
# 6. canvas_node_execute
# ---------------------------------------------------------------------------

from oprim._hevi_types import CanvasNode
from oprim.canvas_node_execute import CanvasNodeResult, canvas_node_execute


def _make_node(**overrides) -> CanvasNode:
    defaults = dict(node_id="n-1", node_type="text", label="Test Node")
    defaults.update(overrides)
    return CanvasNode(**defaults)


def test_canvas_node_execute_no_executor_fails_gracefully():
    node = _make_node()
    result = asyncio.run(canvas_node_execute(node=node))
    assert result.success is False
    assert result.error == "no executor"
    assert result.output is None


def test_canvas_node_execute_no_executor_node_id():
    node = _make_node(node_id="abc")
    result = asyncio.run(canvas_node_execute(node=node))
    assert result.node_id == "abc"


def test_canvas_node_execute_sync_executor():
    node = _make_node()

    def sync_exec(n, upstream):
        return {"rendered": n.label}

    result = asyncio.run(canvas_node_execute(node=node, executor=sync_exec))
    assert result.success is True
    assert result.output == {"rendered": "Test Node"}


def test_canvas_node_execute_async_executor():
    node = _make_node(node_type="image")

    async def async_exec(n, upstream):
        return f"image_output_for_{n.node_id}"

    result = asyncio.run(canvas_node_execute(node=node, executor=async_exec))
    assert result.success is True
    assert "n-1" in result.output


def test_canvas_node_execute_executor_exception_captured():
    node = _make_node()

    def bad_exec(n, upstream):
        raise RuntimeError("something broke")

    result = asyncio.run(canvas_node_execute(node=node, executor=bad_exec))
    assert result.success is False
    assert "something broke" in result.error


def test_canvas_node_execute_upstream_passed_to_executor():
    node = _make_node()
    captured = {}

    def capture_exec(n, upstream):
        captured.update(upstream)
        return "ok"

    upstream = {"n-0": "prev_output"}
    asyncio.run(canvas_node_execute(node=node, upstream_outputs=upstream, executor=capture_exec))
    assert captured == {"n-0": "prev_output"}


def test_canvas_node_execute_node_type_in_result():
    node = _make_node(node_type="video")
    result = asyncio.run(canvas_node_execute(node=node))
    assert result.node_type == "video"


def test_canvas_node_execute_result_model():
    node = _make_node()
    result = asyncio.run(canvas_node_execute(node=node))
    assert isinstance(result, CanvasNodeResult)


# ---------------------------------------------------------------------------
# 7. canvas_edge_validate  (SYNC)
# ---------------------------------------------------------------------------

from oprim.canvas_edge_validate import COMPATIBLE, canvas_edge_validate


def test_canvas_edge_validate_is_sync():
    assert not inspect.iscoroutinefunction(canvas_edge_validate)


def test_canvas_edge_validate_true_cases():
    true_pairs = [
        ("text", "image"),
        ("text", "video"),
        ("image", "video"),
        ("audio", "video"),
        ("script", "video"),
    ]
    for from_type, to_type in true_pairs:
        assert canvas_edge_validate(from_type=from_type, to_type=to_type) is True, \
            f"Expected True for ({from_type}, {to_type})"


def test_canvas_edge_validate_additional_true_cases():
    assert canvas_edge_validate(from_type="text", to_type="audio") is True
    assert canvas_edge_validate(from_type="text", to_type="script") is True
    assert canvas_edge_validate(from_type="image", to_type="image") is True
    assert canvas_edge_validate(from_type="image", to_type="script") is True
    assert canvas_edge_validate(from_type="video", to_type="video") is True
    assert canvas_edge_validate(from_type="script", to_type="image") is True


def test_canvas_edge_validate_false_cases():
    false_pairs = [
        ("audio", "text"),
        ("video", "text"),
        ("video", "audio"),
    ]
    for from_type, to_type in false_pairs:
        assert canvas_edge_validate(from_type=from_type, to_type=to_type) is False, \
            f"Expected False for ({from_type}, {to_type})"


def test_canvas_edge_validate_unknown_types_false():
    assert canvas_edge_validate(from_type="banana", to_type="apple") is False


def test_canvas_edge_validate_empty_strings_false():
    assert canvas_edge_validate(from_type="", to_type="") is False


def test_canvas_edge_validate_compatible_dict_has_correct_entries():
    assert len(COMPATIBLE) == 11
    assert all(v is True for v in COMPATIBLE.values())


def test_canvas_edge_validate_symmetric_not_assumed():
    # audio->video is True but video->audio is not in matrix
    assert canvas_edge_validate(from_type="audio", to_type="video") is True
    assert canvas_edge_validate(from_type="video", to_type="audio") is False


# ---------------------------------------------------------------------------
# 8. adapt_prompt_for_provider
# ---------------------------------------------------------------------------

from oprim.adapt_prompt_for_provider import adapt_prompt_for_provider


def test_adapt_prompt_known_provider_wan22():
    result = asyncio.run(adapt_prompt_for_provider("a sunset", provider="wan22"))
    assert result["provider"] == "wan22"
    assert "a sunset" in result["prompt"]
    assert "电影级画质" in result["prompt"]


def test_adapt_prompt_known_provider_ltx2():
    result = asyncio.run(adapt_prompt_for_provider("a forest", provider="ltx2"))
    assert "4K" in result["prompt"]


def test_adapt_prompt_known_provider_flux():
    result = asyncio.run(adapt_prompt_for_provider("a portrait", provider="flux"))
    assert "masterpiece" in result["prompt"]


def test_adapt_prompt_unknown_provider_passthrough():
    result = asyncio.run(adapt_prompt_for_provider("hello world", provider="unknown"))
    assert result["prompt"] == "hello world"
    assert result["provider"] == "unknown"


def test_adapt_prompt_negative_prompt_preserved():
    result = asyncio.run(
        adapt_prompt_for_provider("a dog", provider="flux", negative_prompt="blurry")
    )
    assert result["negative_prompt"] == "blurry"


def test_adapt_prompt_returns_dict_keys():
    result = asyncio.run(adapt_prompt_for_provider("x", provider="flux"))
    assert {"prompt", "negative_prompt", "provider"} <= result.keys()


# ---------------------------------------------------------------------------
# 9. provider_health_check
# ---------------------------------------------------------------------------

from oprim.provider_health_check import provider_health_check


def test_provider_health_check_returns_bool():
    result = asyncio.run(provider_health_check("nonexistent_provider"))
    assert isinstance(result, bool)


def test_provider_health_check_unregistered_returns_false():
    result = asyncio.run(provider_health_check("__totally_unknown_xyz__"))
    assert result is False


def test_provider_health_check_never_raises():
    # Must not raise regardless of provider name
    try:
        asyncio.run(provider_health_check(""))
    except Exception as exc:
        pytest.fail(f"provider_health_check raised unexpectedly: {exc}")


def test_provider_health_check_with_timeout():
    result = asyncio.run(provider_health_check("no_such_provider", timeout_s=0.001))
    assert result is False


def test_provider_health_check_false_for_empty_string():
    result = asyncio.run(provider_health_check(""))
    assert result is False


def test_provider_health_check_false_for_numeric_name():
    result = asyncio.run(provider_health_check("12345"))
    assert result is False


# ---------------------------------------------------------------------------
# 10. character_three_view
# ---------------------------------------------------------------------------

from oprim.character_three_view import ThreeViewError, ThreeViewResult, character_three_view

_THREE_VIEW_JSON = json.dumps({
    "front": "front view of an elf warrior",
    "side": "side view of an elf warrior",
    "back": "back view of an elf warrior",
    "style_tags": ["fantasy", "detailed"],
})


def test_character_three_view_returns_result():
    result = asyncio.run(
        character_three_view("an elf warrior", caller=make_caller(_THREE_VIEW_JSON))
    )
    assert isinstance(result, ThreeViewResult)


def test_character_three_view_front_prompt():
    result = asyncio.run(
        character_three_view("an elf warrior", caller=make_caller(_THREE_VIEW_JSON))
    )
    assert "elf warrior" in result.front_prompt


def test_character_three_view_side_prompt():
    result = asyncio.run(
        character_three_view("an elf warrior", caller=make_caller(_THREE_VIEW_JSON))
    )
    assert result.side_prompt != ""


def test_character_three_view_back_prompt():
    result = asyncio.run(
        character_three_view("an elf warrior", caller=make_caller(_THREE_VIEW_JSON))
    )
    assert result.back_prompt != ""


def test_character_three_view_style_tags():
    result = asyncio.run(
        character_three_view("an elf warrior", caller=make_caller(_THREE_VIEW_JSON))
    )
    assert "fantasy" in result.style_tags


def test_character_three_view_raises_on_bad_json():
    with pytest.raises(ThreeViewError):
        asyncio.run(
            character_three_view("a robot", caller=make_caller("not valid json"))
        )


def test_character_three_view_raises_on_caller_error():
    async def bad_caller(**kwargs):
        raise RuntimeError("network error")

    with pytest.raises(ThreeViewError):
        asyncio.run(character_three_view("a robot", caller=bad_caller))
