"""Focused backward-compatibility tests for oprim.probe_duration (HEVI contract)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from oprim import ProbeDurationError, probe_duration
from oprim._exceptions import OprimError
from oprim._media_probe import MediaInfo

_MEDIA_PROBE = "oprim._probe_duration.media_probe"


def test_public_import_contract() -> None:
    """`from oprim import probe_duration, ProbeDurationError` is the consumer contract."""
    assert callable(probe_duration)
    assert issubclass(ProbeDurationError, Exception)


def test_success_with_path(tmp_path: Path) -> None:
    media = tmp_path / "a.mp4"
    with patch(
        _MEDIA_PROBE, return_value=MediaInfo(duration_seconds=1.5)
    ) as mocked:
        assert probe_duration(media) == pytest.approx(1.5)
    mocked.assert_called_once_with(path=str(media))


def test_success_with_str() -> None:
    with patch(
        _MEDIA_PROBE, return_value=MediaInfo(duration_seconds=2.25)
    ) as mocked:
        assert probe_duration("/tmp/a.mp4") == pytest.approx(2.25)
    mocked.assert_called_once_with(path="/tmp/a.mp4")


def test_media_probe_error_is_wrapped() -> None:
    cause = OprimError("ffprobe produced no output")
    with (
        patch(_MEDIA_PROBE, side_effect=cause),
        pytest.raises(ProbeDurationError) as excinfo,
    ):
        probe_duration("/tmp/a.mp4")
    assert excinfo.value.__cause__ is cause


def test_missing_duration_raises() -> None:
    with (
        patch(_MEDIA_PROBE, return_value=MediaInfo(duration_seconds=None)),
        pytest.raises(ProbeDurationError, match="missing"),
    ):
        probe_duration("/tmp/a.mp4")


def test_unparsable_duration_raises() -> None:
    fake = SimpleNamespace(duration_seconds="N/A")
    with (
        patch(_MEDIA_PROBE, return_value=fake),
        pytest.raises(ProbeDurationError, match="unparsable"),
    ):
        probe_duration("/tmp/a.mp4")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_duration_raises(bad: float) -> None:
    with (
        patch(_MEDIA_PROBE, return_value=MediaInfo(duration_seconds=bad)),
        pytest.raises(ProbeDurationError, match="invalid"),
    ):
        probe_duration("/tmp/a.mp4")
