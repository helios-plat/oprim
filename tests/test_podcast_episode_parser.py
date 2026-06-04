"""Tests for oprim.podcast_episode_parser."""

from __future__ import annotations

import pytest

from oprim.podcast_episode_parser import podcast_episode_parser

VALID_PODCAST_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>My Podcast</title>
    <description>A great podcast</description>
    <itunes:author>John Doe</itunes:author>
    <item>
      <title>Episode 1</title>
      <link>http://example.com/ep1</link>
      <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
      <itunes:duration>30:00</itunes:duration>
      <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg" length="15000000"/>
      <guid>http://example.com/ep1</guid>
    </item>
    <item>
      <title>Episode 2</title>
      <link>http://example.com/ep2</link>
      <pubDate>Mon, 08 Jan 2024 10:00:00 GMT</pubDate>
      <itunes:duration>45:30</itunes:duration>
      <enclosure url="http://example.com/ep2.mp3" type="audio/mpeg" length="22000000"/>
      <guid>http://example.com/ep2</guid>
    </item>
  </channel>
</rss>
"""

EMPTY_PODCAST_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Podcast</title>
  </channel>
</rss>
"""


def test_valid_podcast_rss_returns_episodes():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert len(result["episodes"]) == 2


def test_podcast_title_extracted():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["podcast_title"] == "My Podcast"


def test_enclosure_url_extracted():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episodes"][0]["enclosure_url"] == "http://example.com/ep1.mp3"


def test_enclosure_type_audio_mpeg():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episodes"][0]["enclosure_type"] == "audio/mpeg"


def test_enclosure_length_as_int():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episodes"][0]["enclosure_length"] == 15000000
    assert isinstance(result["episodes"][0]["enclosure_length"], int)


def test_duration_extracted():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episodes"][0]["duration"] == "30:00"


def test_guid_extracted():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episodes"][0]["guid"] == "http://example.com/ep1"


def test_pub_date_extracted():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert "2024" in result["episodes"][0]["pub_date"]


def test_max_episodes_limits():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS, max_episodes=1)
    assert len(result["episodes"]) == 1


def test_episode_count_equals_len_episodes():
    result = podcast_episode_parser(xml=VALID_PODCAST_RSS)
    assert result["episode_count"] == len(result["episodes"])


def test_malformed_xml_returns_error():
    result = podcast_episode_parser(xml="<broken><<<")
    assert result["error"] is not None
    assert result["episodes"] == []


def test_empty_podcast_returns_no_episodes():
    result = podcast_episode_parser(xml=EMPTY_PODCAST_RSS)
    assert result["episodes"] == []
    assert result["episode_count"] == 0
