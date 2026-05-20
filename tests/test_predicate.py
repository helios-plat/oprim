"""Tests for oprim.predicate — Sprint 0 threshold predicate primitives."""

from __future__ import annotations

import pytest

from oprim.predicate import evaluate_threshold_condition


class TestEvaluateThresholdCondition:
    def test_gt_true(self) -> None:
        assert evaluate_threshold_condition(0.8, 0.5, "gt") is True

    def test_gt_false(self) -> None:
        assert evaluate_threshold_condition(0.3, 0.5, "gt") is False

    def test_gte_equal(self) -> None:
        assert evaluate_threshold_condition(0.5, 0.5, "gte") is True

    def test_gte_greater(self) -> None:
        assert evaluate_threshold_condition(0.6, 0.5, "gte") is True

    def test_gte_less(self) -> None:
        assert evaluate_threshold_condition(0.4, 0.5, "gte") is False

    def test_lt_true(self) -> None:
        assert evaluate_threshold_condition(0.2, 0.5, "lt") is True

    def test_lt_false(self) -> None:
        assert evaluate_threshold_condition(0.7, 0.5, "lt") is False

    def test_lte_equal(self) -> None:
        assert evaluate_threshold_condition(0.5, 0.5, "lte") is True

    def test_lte_less(self) -> None:
        assert evaluate_threshold_condition(0.4, 0.5, "lte") is True

    def test_eq_true(self) -> None:
        assert evaluate_threshold_condition(1.0, 1.0, "eq") is True

    def test_eq_false(self) -> None:
        assert evaluate_threshold_condition(1.0, 2.0, "eq") is False

    def test_ne_true(self) -> None:
        assert evaluate_threshold_condition(1.0, 2.0, "ne") is True

    def test_ne_false(self) -> None:
        assert evaluate_threshold_condition(1.0, 1.0, "ne") is False

    def test_raises_on_invalid_op(self) -> None:
        with pytest.raises(ValueError, match="Unknown operator"):
            evaluate_threshold_condition(1.0, 0.5, "invalid")  # type: ignore

    def test_negative_values(self) -> None:
        assert evaluate_threshold_condition(-1.0, -2.0, "gt") is True
        assert evaluate_threshold_condition(-2.0, -1.0, "lt") is True

    def test_zero_threshold(self) -> None:
        assert evaluate_threshold_condition(0.001, 0.0, "gt") is True
        assert evaluate_threshold_condition(-0.001, 0.0, "lt") is True

    @pytest.mark.academic_reference
    def test_academic_all_operators(self) -> None:
        """Verify all 6 comparison operators against manual checks.

        Reference: standard comparison predicate semantics.
        """
        v, t = 5.0, 3.0
        assert evaluate_threshold_condition(v, t, "gt") is True   # 5 > 3
        assert evaluate_threshold_condition(v, t, "gte") is True  # 5 >= 3
        assert evaluate_threshold_condition(v, t, "lt") is False  # 5 < 3
        assert evaluate_threshold_condition(v, t, "lte") is False # 5 <= 3
        assert evaluate_threshold_condition(v, t, "eq") is False  # 5 == 3
        assert evaluate_threshold_condition(v, t, "ne") is True   # 5 != 3
        # Equal case
        v2 = 3.0
        assert evaluate_threshold_condition(v2, t, "gte") is True
        assert evaluate_threshold_condition(v2, t, "lte") is True
        assert evaluate_threshold_condition(v2, t, "eq") is True
        assert evaluate_threshold_condition(v2, t, "ne") is False
