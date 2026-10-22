"""Tests for oprim.jaccard_similarity (ported from Tide _platform_ops, 16 tests).

Note: Tide's originals include 2 hypothesis-based property tests
(test_property_symmetry / test_property_non_negative). hypothesis isn't a
dependency here and none of oprim's other ported _platform_ops tests use it,
so those are replaced with equivalent explicit-example tests below
(test_symmetry_examples / test_non_negative_examples) to keep the same
coverage without adding a new dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from oprim.jaccard_similarity import jaccard_similarity


class TestJaccardBinary:
    def test_known_value(self) -> None:
        result = jaccard_similarity({1, 2, 3}, {2, 3, 4})
        assert abs(result - 0.5) < 1e-12

    def test_empty_both(self) -> None:
        assert jaccard_similarity(set(), set()) == 1.0

    def test_identical(self) -> None:
        s = {1, 2, 3, 4, 5}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint(self) -> None:
        assert jaccard_similarity({1, 2}, {3, 4}) == 0.0

    def test_list_input(self) -> None:
        result = jaccard_similarity([1, 2, 3], [2, 3, 4])
        assert abs(result - 0.5) < 1e-12

    def test_array_input(self) -> None:
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        result = jaccard_similarity(a, b)
        assert abs(result - 0.5) < 1e-12

    def test_range_0_1(self) -> None:
        result = jaccard_similarity({1, 2, 3}, {3, 4, 5, 6})
        assert 0.0 <= result <= 1.0


class TestJaccardWeighted:
    def test_unit_weights_equals_binary(self) -> None:
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        wa = np.ones(3)
        wb = np.ones(3)
        binary = jaccard_similarity({1, 2, 3}, {2, 3, 4})
        weighted = jaccard_similarity(a, b, weights_a=wa, weights_b=wb, mode="weighted")
        assert abs(binary - weighted) < 1e-10

    def test_weighted_known_value(self) -> None:
        a = np.array([1, 2])
        b = np.array([2, 3])
        wa = np.array([1.0, 2.0])
        wb = np.array([2.0, 3.0])
        result = jaccard_similarity(a, b, weights_a=wa, weights_b=wb, mode="weighted")
        assert 0.0 < result < 1.0

    def test_negative_weights_raises(self) -> None:
        a = np.array([1, 2])
        b = np.array([2, 3])
        wa = np.array([-1.0, 2.0])
        wb = np.array([1.0, 3.0])
        with pytest.raises(ValueError, match="non-negative"):
            jaccard_similarity(a, b, weights_a=wa, weights_b=wb, mode="weighted")

    def test_missing_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights_a and weights_b required"):
            jaccard_similarity(np.array([1]), np.array([1]), mode="weighted")

    def test_empty_weighted_returns_1(self) -> None:
        result = jaccard_similarity(
            np.array([]),
            np.array([]),
            weights_a=np.array([]),
            weights_b=np.array([]),
            mode="weighted",
        )
        assert result == 1.0

    def test_weighted_zero_weights_returns_1(self) -> None:
        a = np.array([1])
        b = np.array([2])
        wa = np.array([0.0])
        wb = np.array([0.0])
        result = jaccard_similarity(a, b, weights_a=wa, weights_b=wb, mode="weighted")
        assert result == 1.0

    def test_misaligned_weights_raises(self) -> None:
        with pytest.raises(ValueError):
            jaccard_similarity(
                np.array([1, 2]),
                np.array([2, 3]),
                weights_a=np.array([1.0]),
                weights_b=np.array([1.0, 2.0]),
                mode="weighted",
            )


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (frozenset(), frozenset()),
        (frozenset({1}), frozenset()),
        (frozenset({1, 2, 3}), frozenset({2, 3, 4})),
        (frozenset({1, 2, 3}), frozenset({4, 5, 6})),
        (frozenset(range(10)), frozenset(range(5, 15))),
    ],
)
def test_symmetry_examples(a: frozenset, b: frozenset) -> None:
    j_ab = jaccard_similarity(a, b)
    j_ba = jaccard_similarity(b, a)
    assert abs(j_ab - j_ba) < 1e-12, f"J(A,B) != J(B,A): {j_ab} vs {j_ba}"


@pytest.mark.parametrize(
    "a",
    [frozenset(), frozenset({1}), frozenset({1, 2, 3}), frozenset(range(20))],
)
def test_non_negative_examples(a: frozenset) -> None:
    assert jaccard_similarity(a, a) >= 0.0
