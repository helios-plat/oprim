from __future__ import annotations

import pytest

from oprim.epistemic_confidence_compute import epistemic_confidence_compute


def test_confidence_normal():
    # ["proven","high","moderate"] -> (1.0 + 0.8 + 0.6) / 3 = 0.8
    result = epistemic_confidence_compute(grades=["proven", "high", "moderate"])
    assert result == pytest.approx(0.8)


def test_confidence_empty():
    assert epistemic_confidence_compute(grades=[]) == 0.0


def test_confidence_all_contradicted():
    assert epistemic_confidence_compute(grades=["contradicted", "contradicted"]) == 0.0


def test_confidence_custom_weights():
    custom_weights = {"custom_grade": 0.5}
    result = epistemic_confidence_compute(
        grades=["custom_grade", "custom_grade"],
        weights=custom_weights
    )
    assert result == 0.5


def test_confidence_unknown_grade():
    # "unknown" not in default weights, should use unknown_grade_weight (default 0.2)
    result = epistemic_confidence_compute(grades=["proven", "unknown"])
    assert result == pytest.approx((1.0 + 0.2) / 2)


def test_confidence_unknown_grade_custom_fallback():
    result = epistemic_confidence_compute(
        grades=["proven", "unknown"],
        unknown_grade_weight=0.5
    )
    assert result == pytest.approx((1.0 + 0.5) / 2)


def test_confidence_out_of_bounds_weights():
    with pytest.raises(ValueError, match="is out of bounds"):
        epistemic_confidence_compute(
            grades=["proven"],
            weights={"proven": 1.5}
        )


def test_confidence_single_element():
    assert epistemic_confidence_compute(grades=["proven"]) == 1.0
