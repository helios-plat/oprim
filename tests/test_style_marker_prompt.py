"""Tests for oprim.style_marker_prompt."""

from __future__ import annotations

import pytest
from oprim.style_marker_prompt import style_marker_prompt


@pytest.mark.parametrize(
    "style,expected_keywords,expected_tone",
    [
        ("科普", "科普风格", "通俗易懂"),
        ("严肃", "严肃风格", "庄重权威"),
        ("搞笑", "搞笑风格", "轻松幽默"),
        ("治愈", "治愈风格", "温暖柔和"),
        ("悬疑", "悬疑风格", "紧张神秘"),
        ("热血", "热血风格", "激昂澎湃"),
        ("温暖", "温暖风格", "亲切温馨"),
    ],
)
def test_all_seven_styles_contain_correct_keywords(
    style: str, expected_keywords: str, expected_tone: str
) -> None:
    """All 7 styles inject the correct keyword pair."""
    result = style_marker_prompt(base_prompt="一只猫", style=style)  # type: ignore[arg-type]
    assert expected_keywords in result
    assert expected_tone in result


def test_base_prompt_appears_first() -> None:
    """base_prompt is the first part of the output."""
    result = style_marker_prompt(base_prompt="一只狗在跑", style="热血")
    assert result.startswith("一只狗在跑")


def test_unknown_style_raises_value_error() -> None:
    """Unknown style raises ValueError at runtime."""
    with pytest.raises(ValueError, match="Unknown style"):
        style_marker_prompt(base_prompt="test", style="未知风格")  # type: ignore[arg-type]


def test_empty_base_prompt_raises_value_error() -> None:
    """Empty base_prompt raises ValueError."""
    with pytest.raises(ValueError, match="base_prompt must not be empty"):
        style_marker_prompt(base_prompt="", style="科普")


def test_long_base_prompt_not_truncated() -> None:
    """Long base_prompt is preserved without truncation."""
    long_prompt = "a" * 1000
    result = style_marker_prompt(base_prompt=long_prompt, style="科普")
    assert result.startswith(long_prompt)
    assert len(result) > 1000


def test_unicode_base_prompt_handled_correctly() -> None:
    """Unicode characters in base_prompt are preserved."""
    unicode_prompt = "🌟✨🎬 电影感开场镜头"
    result = style_marker_prompt(base_prompt=unicode_prompt, style="治愈")
    assert result.startswith(unicode_prompt)
    assert "治愈风格" in result


def test_output_format_is_comma_separated_fixed_order() -> None:
    """Output is exactly '{base_prompt}, {keywords}, {tone}'."""
    result = style_marker_prompt(base_prompt="场景描述", style="科普")
    parts = result.split(", ")
    assert len(parts) == 3
    assert parts[0] == "场景描述"
    assert parts[1] == "科普风格"
    assert parts[2] == "通俗易懂"
