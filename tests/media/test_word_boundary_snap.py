"""word_boundary_snap — contract and correctness.

The point of the element is that a cut cannot land inside a word. These tests
check the move, the idempotence that makes it safe to apply twice, and the
fail-closed error contract.
"""

from __future__ import annotations

import pytest

from oprim._word_boundary_snap import WordBoundarySnapError, word_boundary_snap

WORDS = [
    {"text": "hello", "start": 1.00, "end": 1.40},
    {"text": "world", "start": 1.60, "end": 2.00},
]


def test_snap_moves_boundary_out_of_a_word() -> None:
    # 1.2s is inside "hello" (1.00-1.40).
    assert word_boundary_snap(boundaries_s=[1.20], words=WORDS) == [1.40]


def test_snap_is_idempotent() -> None:
    once = word_boundary_snap(boundaries_s=[1.20], words=WORDS)
    twice = word_boundary_snap(boundaries_s=once, words=WORDS)
    assert once == twice


def test_clean_boundary_is_left_alone() -> None:
    assert word_boundary_snap(boundaries_s=[1.60], words=WORDS) == [1.60]


def test_direction_selects_the_edge() -> None:
    assert word_boundary_snap(
        boundaries_s=[1.20], words=WORDS, direction="out") == [1.00]
    assert word_boundary_snap(
        boundaries_s=[1.20], words=WORDS, direction="in") == [1.40]


def test_nearest_prefers_the_closer_edge() -> None:
    # 1.35 is closer to the end (1.40) than to the start (1.00).
    assert word_boundary_snap(boundaries_s=[1.35], words=WORDS) == [1.40]
    # 1.05 is closer to the start.
    assert word_boundary_snap(boundaries_s=[1.05], words=WORDS) == [1.00]


def test_no_boundary_is_left_inside_a_word() -> None:
    probes = [1.05, 1.10, 1.20, 1.30, 1.38]
    for value in word_boundary_snap(boundaries_s=probes, words=WORDS):
        inside = any(
            start + 0.02 < value < end - 0.02 for start, end in (
                (1.00, 1.40), (1.60, 2.00),
            )
        )
        assert not inside, f"{value} is still inside a word"


def test_order_and_length_are_preserved() -> None:
    out = word_boundary_snap(boundaries_s=[0.0, 1.20, 1.70, 5.0], words=WORDS)
    assert len(out) == 4
    assert out[0] == 0.0
    assert out[3] == 5.0


def test_media_duration_clamps() -> None:
    out = word_boundary_snap(
        boundaries_s=[1.20, 99.0], words=WORDS, media_duration_s=10.0)
    assert out == [1.40, 10.0]


def test_malformed_words_are_ignored_not_fatal() -> None:
    words = [{"text": "x"}, {"text": "y", "start": 1.0}, *WORDS]
    assert word_boundary_snap(boundaries_s=[1.20], words=words) == [1.40]


def test_unsorted_words_still_snap() -> None:
    shuffled = [WORDS[1], WORDS[0]]
    assert word_boundary_snap(boundaries_s=[1.20], words=shuffled) == [1.40]


def test_empty_boundaries_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        word_boundary_snap(boundaries_s=[], words=WORDS)


def test_bad_direction_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        word_boundary_snap(boundaries_s=[1.0], words=WORDS, direction="sideways")


def test_negative_tolerance_is_a_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        word_boundary_snap(boundaries_s=[1.0], words=WORDS, tolerance_s=-1.0)


def test_no_words_fails_closed() -> None:
    """An unsnappable input must not return the boundary as if it were clean."""
    with pytest.raises(WordBoundarySnapError):
        word_boundary_snap(boundaries_s=[1.20], words=[])
    with pytest.raises(WordBoundarySnapError):
        word_boundary_snap(boundaries_s=[1.20], words=[{"start": 1.0}])


def test_keyword_only_signature() -> None:
    with pytest.raises(TypeError):
        word_boundary_snap([1.0], WORDS)  # type: ignore[misc]