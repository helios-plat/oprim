"""Tests for oprim.detect_feed_url."""

from __future__ import annotations

import pytest

from oprim.detect_feed_url import detect_feed_url

HTML_RSS = """\
<html>
<head>
  <link rel="alternate" type="application/rss+xml" title="My RSS" href="http://example.com/rss">
</head>
<body></body>
</html>
"""

HTML_ATOM = """\
<html>
<head>
  <link rel="alternate" type="application/atom+xml" title="My Atom" href="http://example.com/atom">
</head>
<body></body>
</html>
"""

HTML_BOTH = """\
<html>
<head>
  <link rel="alternate" type="application/rss+xml" title="RSS" href="http://example.com/rss">
  <link rel="alternate" type="application/atom+xml" title="Atom" href="http://example.com/atom">
</head>
</html>
"""

HTML_NO_FEED = """\
<html><head><title>No feeds here</title></head><body></body></html>
"""

HTML_RELATIVE = """\
<html>
<head>
  <link rel="alternate" type="application/rss+xml" href="/feed.rss">
</head>
</html>
"""


def test_rss_link_found():
    result = detect_feed_url(html=HTML_RSS)
    assert len(result["feeds"]) == 1
    assert "rss" in result["feeds"][0]["url"]


def test_atom_link_found():
    result = detect_feed_url(html=HTML_ATOM)
    assert len(result["feeds"]) == 1
    assert "atom" in result["feeds"][0]["url"]


def test_multiple_links_returns_all():
    result = detect_feed_url(html=HTML_BOTH)
    assert len(result["feeds"]) == 2


def test_no_feed_link_returns_empty():
    result = detect_feed_url(html=HTML_NO_FEED)
    assert result["feeds"] == []
    assert result["primary_feed"] is None


def test_primary_feed_is_first_found():
    result = detect_feed_url(html=HTML_BOTH)
    assert result["primary_feed"] == "http://example.com/rss"


def test_type_extracted():
    result = detect_feed_url(html=HTML_RSS)
    assert result["feeds"][0]["type"] == "application/rss+xml"


def test_title_attribute_extracted():
    result = detect_feed_url(html=HTML_RSS)
    assert result["feeds"][0]["title"] == "My RSS"


def test_relative_url_joined_with_base_url():
    result = detect_feed_url(html=HTML_RELATIVE, base_url="http://example.com")
    assert result["feeds"][0]["url"] == "http://example.com/feed.rss"


def test_malformed_html_handled_gracefully():
    result = detect_feed_url(html="<<<<not html at all>>>>")
    assert "error" in result


def test_rss_type_recognized():
    result = detect_feed_url(html=HTML_RSS)
    types = [f["type"] for f in result["feeds"]]
    assert "application/rss+xml" in types


def test_atom_type_recognized():
    result = detect_feed_url(html=HTML_ATOM)
    types = [f["type"] for f in result["feeds"]]
    assert "application/atom+xml" in types


def test_no_base_url_keeps_relative():
    result = detect_feed_url(html=HTML_RELATIVE)
    assert result["feeds"][0]["url"] == "/feed.rss"


def test_error_none_on_valid_html():
    result = detect_feed_url(html=HTML_RSS)
    assert result["error"] is None
