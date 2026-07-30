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


# ---------------------------------------------------------------------------
# SSRF protection tests (fetch_rss_feed, not _parse_rss_xml)
# ---------------------------------------------------------------------------


class TestFetchRssFeedSSRF:
    """Verify that fetch_rss_feed uses SSRF-safe transport."""

    def test_file_scheme_rejected(self):
        from oprim.fetch_rss_feed import fetch_rss_feed

        result = fetch_rss_feed(url="file:///etc/passwd")
        assert result["error"] is not None
        assert "unsupported_scheme" in result["error"]

    def test_ftp_scheme_rejected(self):
        from oprim.fetch_rss_feed import fetch_rss_feed

        result = fetch_rss_feed(url="ftp://internal.corp/feed.rss")
        assert result["error"] is not None
        assert "unsupported_scheme" in result["error"]

    def test_ssrf_blocked_private_ip(self):
        from unittest.mock import patch
        import socket
        from oprim.fetch_rss_feed import fetch_rss_feed

        # Simulate private-IP DNS resolution for a URL with http scheme
        private_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 0))]
        with patch("socket.getaddrinfo", return_value=private_addrinfo):
            result = fetch_rss_feed(url="http://internal.corp/feed.rss")
        assert result["error"] is not None
        assert "ssrf_blocked" in result["error"]

    def test_uses_ssrf_safe_opener(self):
        """fetch_rss_feed must import make_ssrf_safe_opener (not raw urlopen)."""
        import inspect
        from oprim import fetch_rss_feed as _mod
        import oprim.fetch_rss_feed as rss_mod

        source = inspect.getsource(rss_mod)
        assert "make_ssrf_safe_opener" in source
        assert "urlopen(url" not in source  # raw urlopen call must be gone

    def test_http_scheme_accepted_with_mocked_fetch(self):
        from unittest.mock import MagicMock, patch
        from oprim.fetch_rss_feed import fetch_rss_feed

        mock_resp = MagicMock()
        mock_resp.read.return_value = VALID_RSS.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_opener = MagicMock()
        mock_opener.open.return_value = mock_resp

        with patch(
            "obase.http.dns_pinned_transport.make_ssrf_safe_opener",
            return_value=mock_opener,
        ):
            result = fetch_rss_feed(url="http://example.com/feed.rss")

        assert result["error"] is None
        assert result["feed_title"] is not None
