"""Tests for oprim.prometheus_instant_query."""
from __future__ import annotations

import json

import httpx
import pytest

from oprim._exceptions import OprimError
from oprim.prometheus_instant_query import prometheus_instant_query

PROM_URL = "http://prom.test:9090/api/v1/query"


class TestPrometheusInstantQuery:
    def test_vector_query_normal(self, mock_prometheus, prometheus_fixtures_dir):
        """Vector query returns expected samples."""
        with open(prometheus_fixtures_dir / "latency_p99_normal.json") as f:
            payload = json.load(f)
        mock_prometheus.get(PROM_URL).mock(return_value=httpx.Response(200, json=payload))

        result = prometheus_instant_query(
            prom_url="http://prom.test:9090",
            query="histogram_quantile(0.99, helixa_signal_latency_seconds_bucket)",
        )

        assert result.result_type == "vector"
        assert result.sample_count == 1
        assert result.samples[0]["value"][1] == "0.85"

    def test_latency_spike_detected(self, mock_prometheus, prometheus_fixtures_dir):
        """Spike scenario returns high value."""
        with open(prometheus_fixtures_dir / "latency_p99_spike.json") as f:
            payload = json.load(f)
        mock_prometheus.get(PROM_URL).mock(return_value=httpx.Response(200, json=payload))

        result = prometheus_instant_query(
            prom_url="http://prom.test:9090",
            query="histogram_quantile(0.99, ...)",
        )

        assert float(result.samples[0]["value"][1]) > 4.0

    def test_empty_result(self, mock_prometheus, prometheus_fixtures_dir):
        """Empty result returns sample_count=0."""
        with open(prometheus_fixtures_dir / "empty_result.json") as f:
            payload = json.load(f)
        mock_prometheus.get(PROM_URL).mock(return_value=httpx.Response(200, json=payload))

        result = prometheus_instant_query(
            prom_url="http://prom.test:9090",
            query="nonexistent_metric",
        )

        assert result.sample_count == 0
        assert result.samples == []

    def test_promql_parse_error_400(self, mock_prometheus, prometheus_fixtures_dir):
        """400 response with parse error raises OprimError."""
        with open(prometheus_fixtures_dir / "promql_parse_error_400.json") as f:
            payload = json.load(f)
        mock_prometheus.get(PROM_URL).mock(return_value=httpx.Response(400, json=payload))

        with pytest.raises(OprimError, match="parse error"):
            prometheus_instant_query(
                prom_url="http://prom.test:9090",
                query="invalid | query",
            )

    def test_bearer_token_header(self, mock_prometheus, prometheus_fixtures_dir):
        """Bearer token is set in Authorization header."""
        with open(prometheus_fixtures_dir / "empty_result.json") as f:
            payload = json.load(f)
        route = mock_prometheus.get(PROM_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )

        prometheus_instant_query(
            prom_url="http://prom.test:9090",
            query="up",
            bearer_token="abc123",
        )

        assert route.calls[0].request.headers["Authorization"] == "Bearer abc123"

    def test_time_unix_parameter(self, mock_prometheus, prometheus_fixtures_dir):
        """time_unix parameter is included in query string."""
        with open(prometheus_fixtures_dir / "empty_result.json") as f:
            payload = json.load(f)
        route = mock_prometheus.get(PROM_URL).mock(
            return_value=httpx.Response(200, json=payload)
        )

        prometheus_instant_query(
            prom_url="http://prom.test:9090",
            query="up",
            time_unix=1234567890.5,
        )

        req_url = str(route.calls[0].request.url)
        assert "time=1234567890.5" in req_url

    def test_timeout(self, mock_prometheus):
        """TimeoutException raises OprimError with 'timeout'."""
        mock_prometheus.get(PROM_URL).mock(side_effect=httpx.TimeoutException)

        with pytest.raises(OprimError, match="timeout"):
            prometheus_instant_query(prom_url="http://prom.test:9090", query="up")

    def test_connect_error(self, mock_prometheus):
        """ConnectError raises OprimError with 'connect'."""
        mock_prometheus.get(PROM_URL).mock(side_effect=httpx.ConnectError)

        with pytest.raises(OprimError, match="Failed to connect"):
            prometheus_instant_query(prom_url="http://prom.test:9090", query="up")

    def test_non_json_error_body(self, mock_prometheus):
        """500 with plain-text body falls back to text in error message."""
        mock_prometheus.get(PROM_URL).mock(
            return_value=httpx.Response(500, text="plain text error")
        )

        with pytest.raises(OprimError, match="plain text error"):
            prometheus_instant_query(prom_url="http://prom.test:9090", query="up")

    def test_error_status_in_response_body(self, mock_prometheus):
        """200 with status=error in body raises OprimError."""
        mock_prometheus.get(PROM_URL).mock(
            return_value=httpx.Response(
                200,
                json={"status": "error", "error": "storage unavailable"},
            )
        )

        with pytest.raises(OprimError, match="storage unavailable"):
            prometheus_instant_query(prom_url="http://prom.test:9090", query="up")
