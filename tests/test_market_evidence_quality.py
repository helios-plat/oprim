from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from oprim.market_evidence_quality import (
    detect_claim_conflicts,
    score_evidence_source,
    validate_market_date_window,
)

TODAY = date(2026, 10, 22)
NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def test_valid_window_is_inclusive_and_normalized():
    assert validate_market_date_window("2026-10-20", end_date="2026-10-22", today=TODAY) == {
        "start_date": "2026-10-20", "end_date": "2026-10-22", "calendar_days": 3, "inclusive": True,
    }


@pytest.mark.parametrize("start,end", [
    ("2026-10-23", "2026-10-22"),
    ("2026-10-22", "2026-10-23"),
    ("2026-02-30", "2026-03-01"),
    ("2026-1-01", "2026-01-02"),
    ("2026-01-01T00:00:00Z", "2026-01-02"),
])
def test_invalid_or_future_window_rejected(start, end):
    with pytest.raises(ValueError):
        validate_market_date_window(start, end_date=end, today=TODAY)


def test_max_days_is_enforced():
    with pytest.raises(ValueError, match="max_days"):
        validate_market_date_window("2026-10-01", end_date="2026-10-22", max_days=20, today=TODAY)


@pytest.mark.parametrize("max_days", [0, -1, 3661, True, 1.5])
def test_max_days_must_be_bounded_integer(max_days):
    with pytest.raises(ValueError, match="max_days"):
        validate_market_date_window("2026-10-20", end_date="2026-10-22", max_days=max_days, today=TODAY)


def test_primary_and_current_source_scores_above_blog():
    primary = score_evidence_source({
        "source_type": "primary_exchange", "source_uri": "https://exchange.example/data",
        "retrieved_at_utc": "2026-10-22T11:00:00Z", "directness": 1.0,
    }, now_utc=NOW)
    blog = score_evidence_source({
        "source_type": "blog", "source_uri": "https://blog.example/post",
        "retrieved_at_utc": "2026-10-22T11:00:00Z", "directness": 0.5,
    }, now_utc=NOW)
    assert primary["score"] > blog["score"]
    assert primary["status"] == "current"


def test_stale_source_is_flagged_and_freshness_zero():
    result = score_evidence_source({
        "source_type": "news", "url": "https://news.example/article",
        "retrieved_at_utc": "2026-10-20T00:00:00Z",
    }, now_utc=NOW)
    assert result["status"] == "stale"
    assert result["freshness_points"] == 0
    assert "source_is_stale" in result["reasons"]


@pytest.mark.parametrize("source", [
    {},
    {"source_uri": "https://x.example", "retrieved_at_utc": "bad"},
    {"source_uri": "https://x.example", "retrieved_at_utc": "2026-10-22T00:00:00"},
    {"source_uri": "https://x.example", "retrieved_at_utc": "2026-10-23T00:00:00Z"},
])
def test_missing_or_unsafe_time_is_unverified(source):
    result = score_evidence_source(source, now_utc=NOW)
    assert result["status"] == "unverified"
    assert result["score"] == 0


def test_source_error_is_not_scored_as_supported():
    result = score_evidence_source({
        "source_type": "market_api", "source_uri": "https://exchange.example/data",
        "retrieved_at_utc": "2026-10-22T11:00:00Z", "error": "upstream unavailable",
    }, now_utc=NOW)
    assert result["status"] == "unverified"
    assert "source_has_error" in result["reasons"]


def test_numeric_disagreement_is_reported_with_provenance():
    conflicts = detect_claim_conflicts([
        {"subject": "BTC-USDT", "predicate": "close", "value": "100.0", "unit": "USDT", "source_id": "a"},
        {"subject": "btc-usdt", "predicate": "close", "value": "101", "unit": "usdt", "source_id": "b"},
    ])
    assert len(conflicts) == 1
    assert conflicts[0]["status"] == "conflict"
    assert conflicts[0]["source_ids"] == ["a", "b"]


def test_values_within_numeric_tolerance_do_not_conflict():
    assert detect_claim_conflicts([
        {"subject": "A", "predicate": "close", "value": 100.0, "source_id": "x"},
        {"subject": "a", "predicate": "close", "value": 100.01, "source_id": "y"},
    ], numeric_tolerance=0.02) == []


def test_categorical_disagreement_is_reported_but_duplicates_are_not():
    base = {"subject": "issuer", "predicate": "status", "unit": "", "value": "active"}
    assert detect_claim_conflicts([{**base, "source_id": "x"}, {**base, "value": " Active ", "source_id": "y"}]) == []
    conflicts = detect_claim_conflicts([{**base, "source_id": "x"}, {**base, "value": "inactive", "source_id": "y"}])
    assert conflicts[0]["reason"] == "categorical_values_disagree"


def test_different_subjects_predicates_or_units_are_not_compared():
    assert detect_claim_conflicts([
        {"subject": "A", "predicate": "close", "value": 100, "unit": "USD", "source_id": "x"},
        {"subject": "B", "predicate": "close", "value": 200, "unit": "USD", "source_id": "y"},
        {"subject": "A", "predicate": "close", "value": 100, "unit": "EUR", "source_id": "z"},
    ]) == []


@pytest.mark.parametrize("claims", ["not a sequence", [None], [{}]])
def test_malformed_claim_sets_are_rejected(claims):
    with pytest.raises(ValueError):
        detect_claim_conflicts(claims)


def test_bad_tolerance_is_rejected():
    with pytest.raises(ValueError, match="numeric_tolerance"):
        detect_claim_conflicts([], numeric_tolerance=-1)


def test_unsafe_source_origin_remains_unverified():
    result = score_evidence_source({
        "source_type": "primary_exchange", "source_uri": "javascript:alert(1)",
        "retrieved_at_utc": "2026-10-22T11:00:00Z",
    }, now_utc=NOW)
    assert result["status"] == "unverified"
    assert result["score"] == 0


def test_datetime_is_not_accepted_for_date_only_reference():
    with pytest.raises(ValueError, match="not a datetime"):
        validate_market_date_window("2026-10-20", end_date="2026-10-22", today=datetime(2026, 10, 22, tzinfo=timezone.utc))
