"""Tests for oprim.loki_query."""
from __future__ import annotations

import json

import httpx
import pytest

from oprim._exceptions import OprimError
from oprim.loki_query import loki_query

LOKI_URL = "http://loki.test:3100/loki/api/v1/query_range"


class TestLokiQuery:
    def test_basic_query(self, mock_loki, loki_fixtures_dir):
        """Basic LogQL query returns parsed lines."""
        with open(loki_fixtures_dir / "helixa_normal.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(200, json=payload))

        result = loki_query(
            loki_url="http://loki.test:3100",
            query='{service="helixa-prob"}',
            start_unix=1715763600.0,
            end_unix=1715763700.0,
        )

        assert result.stream_count == 1
        assert result.total_lines == 3
        assert all(line.labels.get("service") == "helixa-prob" for line in result.log_lines)

    def test_rabbitmq_error_scenario(self, mock_loki, loki_fixtures_dir):
        """Error scenario logs are returned."""
        with open(loki_fixtures_dir / "helixa_rabbitmq_error.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(200, json=payload))

        result = loki_query(
            loki_url="http://loki.test:3100",
            query='{service="helixa-prob"} |= "ERROR"',
            start_unix=1715763600.0,
            end_unix=1715763700.0,
        )

        assert any("heartbeat" in line.line for line in result.log_lines)

    def test_empty_streams(self, mock_loki, loki_fixtures_dir):
        """Empty streams result returns total_lines=0."""
        with open(loki_fixtures_dir / "empty.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(200, json=payload))

        result = loki_query(
            loki_url="http://loki.test:3100",
            query='{service="nonexistent"}',
            start_unix=1715763600.0,
            end_unix=1715763700.0,
        )

        assert result.stream_count == 0
        assert result.total_lines == 0
        assert result.truncated is False

    def test_limit_truncation(self, mock_loki, loki_fixtures_dir):
        """truncated=True when result count equals limit."""
        with open(loki_fixtures_dir / "helixa_normal.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(200, json=payload))

        result = loki_query(
            loki_url="http://loki.test:3100",
            query='{service="helixa-prob"}',
            start_unix=1715763600.0,
            end_unix=1715763700.0,
            limit=3,
        )

        assert result.truncated is True

    def test_logql_parse_error_400(self, mock_loki, loki_fixtures_dir):
        """400 response with parse error raises OprimError."""
        with open(loki_fixtures_dir / "logql_parse_error.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(400, json=payload))

        with pytest.raises(OprimError, match="syntax error"):
            loki_query(
                loki_url="http://loki.test:3100",
                query="invalid logql !!!",
                start_unix=0.0,
                end_unix=1.0,
            )

    def test_psycopg2_pool_error_scenario(self, mock_loki, loki_fixtures_dir):
        """psycopg2 pool error logs are returned."""
        with open(loki_fixtures_dir / "helixa_psycopg2_pool_error.json") as f:
            payload = json.load(f)
        mock_loki.get(LOKI_URL).mock(return_value=httpx.Response(200, json=payload))

        result = loki_query(
            loki_url="http://loki.test:3100",
            query='{service="helixa-risk-spot"} |= "ERROR"',
            start_unix=1715763600.0,
            end_unix=1715763700.0,
        )

        assert any("psycopg2" in line.line for line in result.log_lines)

    def test_timeout(self, mock_loki):
        """TimeoutException raises OprimError with 'timeout'."""
        mock_loki.get(LOKI_URL).mock(side_effect=httpx.TimeoutException)

        with pytest.raises(OprimError, match="timeout"):
            loki_query(
                loki_url="http://loki.test:3100",
                query='{service="helixa-prob"}',
                start_unix=0.0,
                end_unix=1.0,
            )

    def test_connect_error(self, mock_loki):
        """ConnectError raises OprimError with 'connect'."""
        mock_loki.get(LOKI_URL).mock(side_effect=httpx.ConnectError)

        with pytest.raises(OprimError, match="Failed to connect"):
            loki_query(
                loki_url="http://loki.test:3100",
                query='{service="helixa-prob"}',
                start_unix=0.0,
                end_unix=1.0,
            )

    def test_non_json_error_body(self, mock_loki):
        """500 with plain-text body falls back to text in error message."""
        mock_loki.get(LOKI_URL).mock(
            return_value=httpx.Response(500, text="plain text error")
        )

        with pytest.raises(OprimError, match="plain text error"):
            loki_query(
                loki_url="http://loki.test:3100",
                query='{service="helixa-prob"}',
                start_unix=0.0,
                end_unix=1.0,
            )

    def test_error_status_in_response_body(self, mock_loki):
        """200 with status=error in body raises OprimError."""
        mock_loki.get(LOKI_URL).mock(
            return_value=httpx.Response(
                200,
                json={"status": "error", "error": "storage unavailable"},
            )
        )

        with pytest.raises(OprimError, match="storage unavailable"):
            loki_query(
                loki_url="http://loki.test:3100",
                query='{service="helixa-prob"}',
                start_unix=0.0,
                end_unix=1.0,
            )
