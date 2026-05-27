"""Tests for oprim.lighting_control_prompt."""

from __future__ import annotations

import pytest
from oprim.lighting_control_prompt import lighting_control_prompt


@pytest.mark.parametrize(
    "lighting,expected_descriptor",
    [
        ("暖", "warm and cozy light"),
        ("冷", "cool and crisp light"),
        ("戏剧", "dramatic chiaroscuro lighting"),
        ("自然", "soft natural daylight"),
        ("高对比", "high contrast hard light"),
        ("柔和", "soft diffused light"),
    ],
)
def test_all_six_lightings_contain_correct_descriptor(
    lighting: str, expected_descriptor: str
) -> None:
    """All 6 lightings inject the correct descriptor."""
    result = lighting_control_prompt(base_prompt="室内场景", lighting=lighting)  # type: ignore[arg-type]
    assert expected_descriptor in result


def test_base_prompt_embedded_first() -> None:
    """base_prompt is the first part of the output."""
    result = lighting_control_prompt(base_prompt="一段走廊", lighting="戏剧")
    assert result.startswith("一段走廊")


def test_unknown_lighting_raises_value_error() -> None:
    """Unknown lighting raises ValueError."""
    with pytest.raises(ValueError, match="Unknown lighting"):
        lighting_control_prompt(base_prompt="test", lighting="霓虹")  # type: ignore[arg-type]


def test_empty_base_prompt_raises_value_error() -> None:
    """Empty base_prompt raises ValueError."""
    with pytest.raises(ValueError, match="base_prompt must not be empty"):
        lighting_control_prompt(base_prompt="", lighting="自然")


def test_output_contains_lighting_prefix() -> None:
    """Output contains 'lighting:' prefix before the descriptor."""
    result = lighting_control_prompt(base_prompt="场景", lighting="柔和")
    assert "lighting:" in result


def test_single_lighting_value_no_combination() -> None:
    """Only one lighting descriptor appears in the output (no multi-value)."""
    result = lighting_control_prompt(base_prompt="scene", lighting="暖")
    # Should contain exactly one "lighting:" occurrence
    assert result.count("lighting:") == 1
