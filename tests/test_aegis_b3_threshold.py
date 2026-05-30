from __future__ import annotations

import pytest

from oprim.evaluate_threshold_rule import (
    ThresholdResult,
    ThresholdRuleError,
    evaluate_threshold_rule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(metric: str, operator: str, warn: float, critical: float) -> dict:
    return {
        "metric": metric,
        "operator": operator,
        "threshold": {"warn": warn, "critical": critical},
    }


# ---------------------------------------------------------------------------
# >= operator (higher = more severe, e.g. CPU usage)
# ---------------------------------------------------------------------------


def test_gte_below_warn_is_ok() -> None:
    result = evaluate_threshold_rule(
        current_value=50.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert result.severity == "ok"
    assert result.triggered is False
    assert result.threshold_breached is None


def test_gte_between_warn_and_critical_is_warn() -> None:
    result = evaluate_threshold_rule(
        current_value=80.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert result.severity == "warn"
    assert result.triggered is True
    assert result.threshold_breached == 70.0


def test_gte_above_critical_is_critical() -> None:
    result = evaluate_threshold_rule(
        current_value=95.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert result.severity == "critical"
    assert result.triggered is True
    assert result.threshold_breached == 90.0


def test_gte_exactly_at_critical_is_critical() -> None:
    result = evaluate_threshold_rule(
        current_value=90.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert result.severity == "critical"
    assert result.triggered is True


# ---------------------------------------------------------------------------
# <= operator (lower = more severe, e.g. account balance, health score)
# ---------------------------------------------------------------------------


def test_lte_above_warn_is_ok() -> None:
    result = evaluate_threshold_rule(
        current_value=500.0,
        rule_spec=_spec("balance", "<=", warn=200.0, critical=50.0),
    )
    assert result.severity == "ok"
    assert result.triggered is False
    assert result.threshold_breached is None


def test_lte_between_critical_and_warn_is_warn() -> None:
    result = evaluate_threshold_rule(
        current_value=100.0,
        rule_spec=_spec("balance", "<=", warn=200.0, critical=50.0),
    )
    assert result.severity == "warn"
    assert result.triggered is True
    assert result.threshold_breached == 200.0


def test_lte_below_critical_is_critical() -> None:
    result = evaluate_threshold_rule(
        current_value=10.0,
        rule_spec=_spec("balance", "<=", warn=200.0, critical=50.0),
    )
    assert result.severity == "critical"
    assert result.triggered is True
    assert result.threshold_breached == 50.0


# ---------------------------------------------------------------------------
# Strict > and < operators
# ---------------------------------------------------------------------------


def test_gt_strict_exactly_at_warn_is_ok() -> None:
    """Strict >: current == warn should NOT trigger."""
    result = evaluate_threshold_rule(
        current_value=70.0,
        rule_spec=_spec("latency_ms", ">", warn=70.0, critical=90.0),
    )
    assert result.severity == "ok"
    assert result.triggered is False


def test_lt_strict_exactly_at_warn_is_ok() -> None:
    """Strict <: current == warn should NOT trigger."""
    result = evaluate_threshold_rule(
        current_value=200.0,
        rule_spec=_spec("health_score", "<", warn=200.0, critical=50.0),
    )
    assert result.severity == "ok"
    assert result.triggered is False


def test_gt_strict_above_warn_triggers_warn() -> None:
    result = evaluate_threshold_rule(
        current_value=71.0,
        rule_spec=_spec("latency_ms", ">", warn=70.0, critical=90.0),
    )
    assert result.severity == "warn"
    assert result.triggered is True


def test_lt_strict_below_warn_triggers_warn() -> None:
    result = evaluate_threshold_rule(
        current_value=199.0,
        rule_spec=_spec("health_score", "<", warn=200.0, critical=50.0),
    )
    assert result.severity == "warn"
    assert result.triggered is True


# ---------------------------------------------------------------------------
# Missing required fields — ThresholdRuleError
# ---------------------------------------------------------------------------


def test_missing_metric_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="metric"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"operator": ">=", "threshold": {"warn": 5.0, "critical": 8.0}},
        )


def test_missing_operator_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="operator"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"metric": "foo", "threshold": {"warn": 5.0, "critical": 8.0}},
        )


def test_missing_threshold_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="threshold"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"metric": "foo", "operator": ">="},
        )


# ---------------------------------------------------------------------------
# Bad operator
# ---------------------------------------------------------------------------


def test_unsupported_operator_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="unsupported operator"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec=_spec("foo", "!=", warn=5.0, critical=8.0),
        )


# ---------------------------------------------------------------------------
# Semantic order violations
# ---------------------------------------------------------------------------


def test_gte_warn_gte_critical_raises() -> None:
    """For >=, warn must be < critical."""
    with pytest.raises(ThresholdRuleError, match="warn.*must be < critical"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec=_spec("cpu", ">=", warn=90.0, critical=70.0),
        )


def test_gte_warn_equals_critical_raises() -> None:
    with pytest.raises(ThresholdRuleError):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec=_spec("cpu", ">=", warn=90.0, critical=90.0),
        )


def test_lte_warn_lte_critical_raises() -> None:
    """For <=, warn must be > critical."""
    with pytest.raises(ThresholdRuleError, match="warn.*must be > critical"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec=_spec("balance", "<=", warn=50.0, critical=200.0),
        )


# ---------------------------------------------------------------------------
# threshold not a dict
# ---------------------------------------------------------------------------


def test_threshold_not_dict_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="threshold must be dict"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"metric": "foo", "operator": ">=", "threshold": 99.0},
        )


# ---------------------------------------------------------------------------
# threshold dict missing sub-fields
# ---------------------------------------------------------------------------


def test_threshold_missing_warn_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="warn"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"metric": "foo", "operator": ">=", "threshold": {"critical": 90.0}},
        )


def test_threshold_missing_critical_raises() -> None:
    with pytest.raises(ThresholdRuleError, match="critical"):
        evaluate_threshold_rule(
            current_value=10.0,
            rule_spec={"metric": "foo", "operator": ">=", "threshold": {"warn": 70.0}},
        )


# ---------------------------------------------------------------------------
# reason field content
# ---------------------------------------------------------------------------


def test_reason_contains_metric_and_operator_on_warn() -> None:
    result = evaluate_threshold_rule(
        current_value=80.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert "cpu_usage" in result.reason
    assert ">=" in result.reason


def test_reason_contains_metric_on_ok() -> None:
    result = evaluate_threshold_rule(
        current_value=50.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert "cpu_usage" in result.reason
    assert "ok" in result.reason


# ---------------------------------------------------------------------------
# ThresholdResult model integrity
# ---------------------------------------------------------------------------


def test_result_is_threshold_result_instance() -> None:
    result = evaluate_threshold_rule(
        current_value=80.0,
        rule_spec=_spec("cpu_usage", ">=", warn=70.0, critical=90.0),
    )
    assert isinstance(result, ThresholdResult)
    assert result.metric == "cpu_usage"
    assert result.current_value == 80.0
