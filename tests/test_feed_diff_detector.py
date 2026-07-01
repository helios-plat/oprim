"""Tests for oprim.feed_diff_detector."""

from __future__ import annotations

import pytest

from oprim.feed_diff_detector import feed_diff_detector

ITEM_A = {"guid": "http://example.com/1", "title": "Item 1"}
ITEM_B = {"guid": "http://example.com/2", "title": "Item 2"}
ITEM_C = {"guid": "http://example.com/3", "title": "Item 3"}

LINK_A = {"link": "http://example.com/1", "title": "Item 1"}
LINK_B = {"link": "http://example.com/2", "title": "Item 2"}


def test_empty_old_all_new():
    result = feed_diff_detector(old_items=[], new_items=[ITEM_A, ITEM_B])
    assert len(result["new_items"]) == 2
    assert result["total_new"] == 2


def test_empty_new_all_removed():
    result = feed_diff_detector(old_items=[ITEM_A, ITEM_B], new_items=[])
    assert len(result["removed_items"]) == 2


def test_identical_feeds_zero_new():
    result = feed_diff_detector(old_items=[ITEM_A, ITEM_B], new_items=[ITEM_A, ITEM_B])
    assert result["total_new"] == 0
    assert len(result["new_items"]) == 0


def test_one_new_item_detected():
    result = feed_diff_detector(old_items=[ITEM_A], new_items=[ITEM_A, ITEM_B])
    assert result["total_new"] == 1
    assert result["new_items"][0]["guid"] == "http://example.com/2"


def test_one_removed_item_detected():
    result = feed_diff_detector(old_items=[ITEM_A, ITEM_B], new_items=[ITEM_A])
    assert len(result["removed_items"]) == 1
    assert result["removed_items"][0]["guid"] == "http://example.com/2"


def test_key_field_fallback_to_link():
    result = feed_diff_detector(old_items=[LINK_A], new_items=[LINK_A, LINK_B], key_field="guid")
    # guid absent — falls back to link
    assert result["total_new"] == 1


def test_unchanged_count_correct():
    result = feed_diff_detector(old_items=[ITEM_A, ITEM_B], new_items=[ITEM_A, ITEM_B, ITEM_C])
    assert result["unchanged_count"] == 2


def test_total_new_equals_len_new_items():
    result = feed_diff_detector(old_items=[ITEM_A], new_items=[ITEM_A, ITEM_B, ITEM_C])
    assert result["total_new"] == len(result["new_items"])


def test_multiple_new_items():
    result = feed_diff_detector(old_items=[ITEM_A], new_items=[ITEM_A, ITEM_B, ITEM_C])
    assert result["total_new"] == 2


def test_order_preserved_in_new_items():
    result = feed_diff_detector(old_items=[], new_items=[ITEM_A, ITEM_B, ITEM_C])
    guids = [i["guid"] for i in result["new_items"]]
    assert guids == [
        "http://example.com/1",
        "http://example.com/2",
        "http://example.com/3",
    ]


def test_custom_key_field():
    items_old = [{"uid": "abc", "title": "A"}]
    items_new = [{"uid": "abc", "title": "A"}, {"uid": "xyz", "title": "X"}]
    result = feed_diff_detector(old_items=items_old, new_items=items_new, key_field="uid")
    assert result["total_new"] == 1
    assert result["new_items"][0]["uid"] == "xyz"


def test_both_empty():
    result = feed_diff_detector(old_items=[], new_items=[])
    assert result["new_items"] == []
    assert result["removed_items"] == []
    assert result["unchanged_count"] == 0
    assert result["total_new"] == 0
