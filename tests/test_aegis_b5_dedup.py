# tests/test_aegis_b5_dedup.py
"""Tests for oprim.compute_dedup_key — time-bucket dedup key (B5)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timezone

import pytest

from oprim.compute_dedup_key import compute_dedup_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_ANCHOR = datetime(2026, 5, 30, 12, 0, 0, tzinfo=UTC)  # top of an hour


# ---------------------------------------------------------------------------
# Test 1: deterministic — same inputs produce same key
# ---------------------------------------------------------------------------


def test_deterministic_same_inputs():
    key1 = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    key2 = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    assert key1 == key2


# ---------------------------------------------------------------------------
# Test 2: different rule_id → different key
# ---------------------------------------------------------------------------


def test_different_rule_id_gives_different_key():
    key1 = compute_dedup_key(rule_id="rule-A", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    key2 = compute_dedup_key(rule_id="rule-B", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    assert key1 != key2


# ---------------------------------------------------------------------------
# Test 3: different entity_id → different key
# ---------------------------------------------------------------------------


def test_different_entity_id_gives_different_key():
    key1 = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    key2 = compute_dedup_key(rule_id="r1", entity_id="host-b", bucket_anchor=FIXED_ANCHOR)
    assert key1 != key2


# ---------------------------------------------------------------------------
# Test 4: bucket_anchor across bucket boundary → different key
#         (1h+1s later, bucket=3600)
# ---------------------------------------------------------------------------


def test_different_key_across_bucket_boundary():
    anchor_before = FIXED_ANCHOR
    anchor_after = datetime(2026, 5, 30, 13, 0, 1, tzinfo=UTC)  # 1h + 1s later
    key1 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=3600, bucket_anchor=anchor_before
    )
    key2 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=3600, bucket_anchor=anchor_after
    )
    assert key1 != key2


# ---------------------------------------------------------------------------
# Test 5: different times within same bucket → same key
#         (5 min later within same 1h bucket)
# ---------------------------------------------------------------------------


def test_same_key_within_same_bucket():
    anchor_start = FIXED_ANCHOR
    anchor_plus5min = datetime(2026, 5, 30, 12, 5, 0, tzinfo=UTC)
    key1 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=3600, bucket_anchor=anchor_start
    )
    key2 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=3600, bucket_anchor=anchor_plus5min
    )
    assert key1 == key2


# ---------------------------------------------------------------------------
# Test 6: return value is 64-char lowercase hex string
# ---------------------------------------------------------------------------


def test_return_value_is_64_char_hex():
    key = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    assert isinstance(key, str)
    assert len(key) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", key) is not None


# ---------------------------------------------------------------------------
# Test 7: bucket_seconds=0 → ValueError
# ---------------------------------------------------------------------------


def test_bucket_seconds_zero_raises_value_error():
    with pytest.raises(ValueError, match="bucket_seconds must be > 0"):
        compute_dedup_key(
            rule_id="r1", entity_id="host-a", bucket_seconds=0, bucket_anchor=FIXED_ANCHOR
        )


# ---------------------------------------------------------------------------
# Test 8: naive bucket_anchor → ValueError mentioning "timezone-aware"
# ---------------------------------------------------------------------------


def test_naive_bucket_anchor_raises_value_error():
    naive = datetime(2026, 5, 30, 12, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=naive)


# ---------------------------------------------------------------------------
# Test 9: bucket_anchor=None → runs without error, returns a string
# ---------------------------------------------------------------------------


def test_bucket_anchor_none_runs_without_error():
    result = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=None)
    assert isinstance(result, str)
    assert len(result) == 64


# ---------------------------------------------------------------------------
# Test 10: negative bucket_seconds → ValueError
# ---------------------------------------------------------------------------


def test_negative_bucket_seconds_raises_value_error():
    with pytest.raises(ValueError, match="bucket_seconds must be > 0"):
        compute_dedup_key(
            rule_id="r1", entity_id="host-a", bucket_seconds=-1, bucket_anchor=FIXED_ANCHOR
        )


# ---------------------------------------------------------------------------
# Test 11: non-UTC timezone-aware anchor still works (not rejected)
# ---------------------------------------------------------------------------


def test_non_utc_timezone_aware_anchor_accepted():
    tz_plus8 = (
        timezone.from_utc(  # type: ignore[attr-defined]
            None
        )
        if False
        else UTC
    )  # fall back to UTC for safety
    # Use a fixed +08:00 offset manually
    from datetime import timedelta

    tz_plus8 = timezone(timedelta(hours=8))
    anchor = datetime(2026, 5, 30, 20, 0, 0, tzinfo=tz_plus8)  # same instant as FIXED_ANCHOR
    key1 = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=anchor)
    key2 = compute_dedup_key(rule_id="r1", entity_id="host-a", bucket_anchor=FIXED_ANCHOR)
    assert key1 == key2  # same UTC timestamp → same bucket → same key


# ---------------------------------------------------------------------------
# Test 12: custom bucket_seconds (60s) creates distinct buckets per minute
# ---------------------------------------------------------------------------


def test_custom_bucket_seconds_minute_granularity():
    anchor_t0 = datetime(2026, 5, 30, 12, 0, 0, tzinfo=UTC)
    anchor_t1 = datetime(2026, 5, 30, 12, 1, 0, tzinfo=UTC)  # +60s, new bucket
    key1 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=60, bucket_anchor=anchor_t0
    )
    key2 = compute_dedup_key(
        rule_id="r1", entity_id="host-a", bucket_seconds=60, bucket_anchor=anchor_t1
    )
    assert key1 != key2


# ---------------------------------------------------------------------------
# Test 13: pipe in rule_id / entity_id does NOT cause collision (null-byte sep)
# ---------------------------------------------------------------------------


def test_pipe_in_fields_no_collision():
    # "a|b" + "c"  vs  "a" + "b|c" — would collide with "|" separator
    key1 = compute_dedup_key(rule_id="a|b", entity_id="c", bucket_anchor=FIXED_ANCHOR)
    key2 = compute_dedup_key(rule_id="a", entity_id="b|c", bucket_anchor=FIXED_ANCHOR)
    assert key1 != key2
