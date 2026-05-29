"""Tests for oprim.camera_motion_prompt."""

from __future__ import annotations

import pytest
from oprim.camera_motion_prompt import camera_motion_prompt


@pytest.mark.parametrize(
    "motion_type,expected_descriptor",
    [
        ("pan_left", "camera pans left"),
        ("pan_right", "camera pans right"),
        ("tilt_up", "camera tilts up"),
        ("tilt_down", "camera tilts down"),
        ("dolly_in", "camera moves forward (dolly in)"),
        ("dolly_out", "camera pulls back (dolly out)"),
        ("rotate", "camera rotates around subject"),
        ("static", "static locked-off shot"),
    ],
)
def test_all_eight_motion_types(motion_type: str, expected_descriptor: str) -> None:
    """All 8 motion types produce their correct descriptor."""
    result = camera_motion_prompt(
        base_motion=None,
        motion_type=motion_type,  # type: ignore[arg-type]
    )
    assert expected_descriptor in result


@pytest.mark.parametrize(
    "intensity,expected_word",
    [
        (0.0, "slow"),
        (0.33, "slow"),
        (0.34, "medium"),
        (0.5, "medium"),
        (0.67, "medium"),
        (0.68, "fast"),
        (1.0, "fast"),
    ],
)
def test_intensity_mapping(intensity: float, expected_word: str) -> None:
    """Intensity maps to correct slow/medium/fast label."""
    result = camera_motion_prompt(base_motion=None, motion_type="pan_left", intensity=intensity)
    assert expected_word in result


def test_base_motion_none_format() -> None:
    """base_motion=None produces '{descriptor}, {intensity} motion' format."""
    result = camera_motion_prompt(base_motion=None, motion_type="tilt_up", intensity=0.1)
    assert result == "camera tilts up, slow motion"


def test_base_motion_non_empty_join_format() -> None:
    """Non-empty base_motion is prepended: '{base_motion}, {descriptor}, {intensity} motion'."""
    result = camera_motion_prompt(base_motion="forest scene", motion_type="dolly_in", intensity=0.5)
    assert result == "forest scene, camera moves forward (dolly in), medium motion"


def test_unknown_motion_type_raises() -> None:
    """Unknown motion_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown motion_type"):
        camera_motion_prompt(base_motion=None, motion_type="zoom")  # type: ignore[arg-type]


def test_intensity_below_zero_raises() -> None:
    """intensity < 0 raises ValueError."""
    with pytest.raises(ValueError, match="intensity must be in"):
        camera_motion_prompt(base_motion=None, motion_type="pan_left", intensity=-0.1)


def test_intensity_above_one_raises() -> None:
    """intensity > 1 raises ValueError."""
    with pytest.raises(ValueError, match="intensity must be in"):
        camera_motion_prompt(base_motion=None, motion_type="pan_left", intensity=1.1)


def test_output_not_empty() -> None:
    """Output is always a non-empty string."""
    result = camera_motion_prompt(base_motion=None, motion_type="static", intensity=0.5)
    assert isinstance(result, str)
    assert len(result) > 0
