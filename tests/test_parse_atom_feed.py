"""Tests for oprim.parse_atom_feed."""

from __future__ import annotations

import pytest

from oprim.parse_atom_feed import parse_atom_feed

VALID_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <id>urn:uuid:atom-test-feed</id>
  <updated>2024-01-02T00:00:00Z</updated>
  <entry>
    <title>Entry 1</title>
    <id>urn:uuid:entry-1</id>
    <link href="http://example.com/entry/1"/>
    <summary>Summary of entry 1</summary>
    <updated>2024-01-01T12:00:00Z</updated>
    <author><name>Alice</name></author>
  </entry>
  <entry>
    <title>Entry 2</title>
    <id>urn:uuid:entry-2</id>
    <link href="http://example.com/entry/2"/>
    <summary>Summary of entry 2</summary>
    <updated>2024-01-02T08:00:00Z</updated>
    <author><name>Bob</name></author>
  </entry>
</feed>
"""

EMPTY_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Empty Atom</title>
  <id>urn:uuid:empty</id>
  <updated>2024-01-01T00:00:00Z</updated>
</feed>
"""


def test_valid_atom_returns_items():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert len(result["items"]) == 2


def test_feed_title_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["feed_title"] == "Atom Test Feed"


def test_feed_id_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["feed_id"] == "urn:uuid:atom-test-feed"


def test_link_extracted_from_href():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["items"][0]["link"] == "http://example.com/entry/1"


def test_summary_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["items"][0]["summary"] == "Summary of entry 1"


def test_updated_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["items"][0]["updated"] == "2024-01-01T12:00:00Z"


def test_author_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["items"][0]["author"] == "Alice"


def test_max_items_limits_results():
    result = parse_atom_feed(xml=VALID_ATOM, max_items=1)
    assert len(result["items"]) == 1


def test_malformed_xml_returns_error():
    result = parse_atom_feed(xml="<broken><<<<")
    assert result["error"] is not None
    assert result["items"] == []


def test_item_count_correct():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["item_count"] == 2


def test_empty_feed_returns_empty_items():
    result = parse_atom_feed(xml=EMPTY_ATOM)
    assert result["items"] == []
    assert result["item_count"] == 0


def test_feed_updated_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["updated"] == "2024-01-02T00:00:00Z"


def test_entry_id_extracted():
    result = parse_atom_feed(xml=VALID_ATOM)
    assert result["items"][0]["id"] == "urn:uuid:entry-1"
