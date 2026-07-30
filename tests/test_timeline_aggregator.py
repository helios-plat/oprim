"""Tests for oprim.timeline_aggregator."""

from __future__ import annotations

import pytest

from oprim.timeline_aggregator import timeline_aggregator

ITEMS_DAY = [
    {"pub_date": "2024-01-01T10:00:00", "title": "A"},
    {"pub_date": "2024-01-01T15:00:00", "title": "B"},
    {"pub_date": "2024-01-02T08:00:00", "title": "C"},
    {"pub_date": "2024-02-01T00:00:00", "title": "D"},
]


def test_day_granularity_groups_by_date():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day")
    periods = [b["period"] for b in result["buckets"]]
    assert "2024-01-01" in periods
    assert "2024-01-02" in periods
    assert "2024-02-01" in periods


def test_week_granularity_groups_by_iso_week():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="week")
    periods = [b["period"] for b in result["buckets"]]
    # All periods should be in ISO week format YYYY-Www
    for p in periods:
        assert "-W" in p


def test_month_granularity_groups_by_year_month():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="month")
    periods = [b["period"] for b in result["buckets"]]
    assert "2024-01" in periods
    assert "2024-02" in periods


def test_sort_desc_orders_newest_first():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day", sort_desc=True)
    periods = [b["period"] for b in result["buckets"]]
    assert periods == sorted(periods, reverse=True)


def test_sort_asc_orders_oldest_first():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day", sort_desc=False)
    periods = [b["period"] for b in result["buckets"]]
    assert periods == sorted(periods)


def test_items_without_timestamp_field_skipped():
    items = [
        {"pub_date": "2024-01-01T00:00:00", "title": "A"},
        {"title": "No date"},
        {"pub_date": None, "title": "Null date"},
    ]
    result = timeline_aggregator(items=items, granularity="day")
    assert result["total_items"] == 1


def test_total_items_correct():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day")
    assert result["total_items"] == len(ITEMS_DAY)


def test_earliest_latest_populated():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day")
    assert result["earliest"] is not None
    assert result["latest"] is not None
    assert result["earliest"] < result["latest"]


def test_empty_list_returns_empty_buckets():
    result = timeline_aggregator(items=[], granularity="day")
    assert result["buckets"] == []
    assert result["total_items"] == 0
    assert result["earliest"] is None
    assert result["latest"] is None


def test_multiple_items_same_day_in_one_bucket():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="day")
    jan1_bucket = next(b for b in result["buckets"] if b["period"] == "2024-01-01")
    assert jan1_bucket["count"] == 2


def test_cross_month_items_split_correctly():
    result = timeline_aggregator(items=ITEMS_DAY, granularity="month")
    jan_bucket = next(b for b in result["buckets"] if b["period"] == "2024-01")
    feb_bucket = next(b for b in result["buckets"] if b["period"] == "2024-02")
    assert jan_bucket["count"] == 3
    assert feb_bucket["count"] == 1


def test_custom_timestamp_field():
    items = [
        {"created_at": "2024-03-15T00:00:00", "val": 1},
        {"created_at": "2024-03-16T00:00:00", "val": 2},
    ]
    result = timeline_aggregator(items=items, timestamp_field="created_at", granularity="day")
    assert result["total_items"] == 2
