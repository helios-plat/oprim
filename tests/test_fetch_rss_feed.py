"""Tests for oprim.fetch_rss_feed."""

from __future__ import annotations

import pytest

from oprim.fetch_rss_feed import _parse_rss_xml

VALID_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>http://example.com</link>
    <description>A test feed</description>
    <item>
      <title>Item 1</title>
      <link>http://example.com/1</link>
      <description>First item</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <guid>http://example.com/1</guid>
    </item>
    <item>
      <title>Item 2</title>
      <link>http://example.com/2</link>
      <description>Second item</description>
      <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
      <guid>http://example.com/2</guid>
    </item>
  </channel>
</rss>
"""

EMPTY_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
  </channel>
</rss>
"""


def test_valid_rss_returns_items():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert len(result["items"]) == 2


def test_empty_feed_returns_empty_list():
    result = _parse_rss_xml(xml_string=EMPTY_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["items"] == []


def test_item_count_correct():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["item_count"] == 2


def test_feed_title_extracted():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["feed_title"] == "Test Feed"


def test_item_link_extracted():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["items"][0]["link"] == "http://example.com/1"


def test_pub_date_extracted():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["items"][0]["pub_date"] is not None
    assert "2024" in result["items"][0]["pub_date"]


def test_guid_extracted():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["items"][0]["guid"] == "http://example.com/1"


def test_max_items_limits_results():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=1)
    assert len(result["items"]) == 1
    assert result["item_count"] == 1


def test_malformed_xml_returns_error():
    result = _parse_rss_xml(
        xml_string="<not valid xml><<<", feed_url="http://example.com/rss", max_items=100
    )
    assert result["error"] is not None


def test_feed_title_from_channel():
    result = _parse_rss_xml(xml_string=EMPTY_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["feed_title"] == "Empty Feed"


def test_item_count_equals_len_items():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["item_count"] == len(result["items"])


def test_description_extracted():
    result = _parse_rss_xml(xml_string=VALID_RSS, feed_url="http://example.com/rss", max_items=100)
    assert result["items"][0]["description"] == "First item"


def test_feed_url_preserved():
    result = _parse_rss_xml(
        xml_string=VALID_RSS, feed_url="http://example.com/feed.rss", max_items=100
    )
    assert result["feed_url"] == "http://example.com/feed.rss"
