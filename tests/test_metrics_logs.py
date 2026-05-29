"""Tests for oprim metrics and logs operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from oprim import loki_log_query, prometheus_instant_query, prometheus_range_query, structlog_parse
from oprim._exceptions import OprimConnectionError, OprimValidationError
from oprim._metrics_logs import InstantResult, LogEntry, RangeResult


def _prom_resp(status_code=200, result_type="vector", results=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = {
        "data": {
            "resultType": result_type,
            "result": results or [],
        }
    }
    resp.text = ""
    return resp


# ---------------------------------------------------------------------------
# prometheus_instant_query
# ---------------------------------------------------------------------------

class TestPrometheusInstantQuery:
    def test_vector_result(self):
        results = [
            {"metric": {"instance": "localhost:9090"}, "value": [1716000000.0, "1.5"]}
        ]
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=results)):
            result = prometheus_instant_query(
                endpoint="http://localhost:9090",
                query="up",
            )
        assert isinstance(result, InstantResult)
        assert result.result_type == "vector"
        assert len(result.samples) == 1
        assert result.samples[0].value == pytest.approx(1.5)

    def test_empty_result(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=[])):
            result = prometheus_instant_query(
                endpoint="http://localhost:9090",
                query="nonexistent_metric",
            )
        assert result.samples == []

    def test_promql_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.is_success = False
        resp.json.return_value = {"error": "invalid expression"}
        resp.text = "invalid expression"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimValidationError):
                prometheus_instant_query(
                    endpoint="http://localhost:9090",
                    query="invalid[[[",
                )

    def test_connection_error(self):
        with patch("oprim._metrics_logs.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                prometheus_instant_query(
                    endpoint="http://nonexistent:9090",
                    query="up",
                )

    def test_multiple_samples(self):
        results = [
            {"metric": {"instance": "host1:9090"}, "value": [1716000000.0, "1.0"]},
            {"metric": {"instance": "host2:9090"}, "value": [1716000000.0, "0.0"]},
        ]
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=results)):
            result = prometheus_instant_query(
                endpoint="http://localhost:9090",
                query="up",
            )
        assert len(result.samples) == 2

    def test_timeout_error(self):
        with patch("oprim._metrics_logs.httpx.get", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OprimConnectionError):
                prometheus_instant_query(endpoint="http://localhost:9090", query="up")

    def test_server_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 503
        resp.is_success = False
        resp.text = "Service Unavailable"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimConnectionError):
                prometheus_instant_query(endpoint="http://localhost:9090", query="up")

    def test_with_time_parameter(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=[])) as mock_get:
            prometheus_instant_query(
                endpoint="http://localhost:9090",
                query="up",
                time="1716000000",
            )
        params_sent = mock_get.call_args.kwargs.get("params", {})
        assert "time" in params_sent

    def test_scalar_result(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.is_success = True
        resp.json.return_value = {"data": {"resultType": "scalar", "result": [1716000000.0, "42.0"]}}
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            result = prometheus_instant_query(endpoint="http://localhost:9090", query="scalar(1)")
        assert result.result_type == "scalar"
        assert len(result.samples) == 1
        assert result.samples[0].value == pytest.approx(42.0)

    def test_invalid_float_value_defaults_to_zero(self):
        results = [{"metric": {}, "value": [1716000000.0, "not-a-number"]}]
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=results)):
            result = prometheus_instant_query(endpoint="http://localhost:9090", query="up")
        assert result.samples[0].value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# prometheus_range_query
# ---------------------------------------------------------------------------

class TestPrometheusRangeQuery:
    def test_normal_range(self):
        results = [
            {
                "metric": {"instance": "localhost:9090"},
                "values": [[1716000000.0, "1.0"], [1716000060.0, "2.0"]],
            }
        ]
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=results)):
            result = prometheus_range_query(
                endpoint="http://localhost:9090",
                query="rate(up[5m])",
                start="2026-05-20T10:00:00Z",
                end="2026-05-20T11:00:00Z",
                step="1m",
            )
        assert isinstance(result, RangeResult)
        assert len(result.series) == 1
        assert len(result.series[0].values) == 2

    def test_empty_range(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=[])):
            result = prometheus_range_query(
                endpoint="http://localhost:9090",
                query="nonexistent",
                start="2026-05-20T10:00:00Z",
                end="2026-05-20T11:00:00Z",
                step="1m",
            )
        assert result.series == []

    def test_epoch_start_end(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=_prom_resp(results=[])):
            result = prometheus_range_query(
                endpoint="http://localhost:9090",
                query="up",
                start="1716000000",
                end="1716003600",
                step="60s",
            )
        assert result.series == []

    def test_promql_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.is_success = False
        resp.json.return_value = {"error": "bad syntax"}
        resp.text = "bad syntax"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimValidationError):
                prometheus_range_query(
                    endpoint="http://localhost:9090",
                    query="bad[[[",
                    start="2026-05-20T10:00:00Z",
                    end="2026-05-20T11:00:00Z",
                    step="1m",
                )

    def test_connection_error(self):
        with patch("oprim._metrics_logs.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                prometheus_range_query(
                    endpoint="http://nonexistent:9090",
                    query="up",
                    start="2026-05-20T10:00:00Z",
                    end="2026-05-20T11:00:00Z",
                    step="1m",
                )

    def test_range_server_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 503
        resp.is_success = False
        resp.text = "unavailable"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimConnectionError):
                prometheus_range_query(
                    endpoint="http://localhost:9090",
                    query="up",
                    start="2026-05-20T10:00:00Z",
                    end="2026-05-20T11:00:00Z",
                    step="1m",
                )


# ---------------------------------------------------------------------------
# loki_log_query
# ---------------------------------------------------------------------------

class TestLokiLogQuery:
    def _loki_resp(self, status_code=200, streams=None):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.is_success = 200 <= status_code < 300
        resp.json.return_value = {
            "data": {
                "result": streams or []
            }
        }
        resp.text = ""
        return resp

    def test_normal_logs(self):
        streams = [
            {
                "stream": {"app": "myapp"},
                "values": [
                    ["1716000000000000000", "first log line"],
                    ["1716000001000000000", "second log line"],
                ],
            }
        ]
        with patch("oprim._metrics_logs.httpx.get", return_value=self._loki_resp(streams=streams)):
            result = loki_log_query(endpoint="http://localhost:3100", logql='{app="myapp"}')
        assert len(result) == 2
        assert all(isinstance(e, LogEntry) for e in result)
        assert result[0].message == "first log line"
        assert result[0].labels == {"app": "myapp"}

    def test_no_logs(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=self._loki_resp(streams=[])):
            result = loki_log_query(endpoint="http://localhost:3100", logql='{app="nonexistent"}')
        assert result == []

    def test_custom_time_range(self):
        with patch("oprim._metrics_logs.httpx.get", return_value=self._loki_resp(streams=[])):
            result = loki_log_query(
                endpoint="http://localhost:3100",
                logql='{app="myapp"}',
                start="2026-05-20T10:00:00Z",
                end="2026-05-20T11:00:00Z",
            )
        assert result == []

    def test_logql_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 400
        resp.is_success = False
        resp.json.return_value = {"error": "parse error"}
        resp.text = "parse error"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimValidationError):
                loki_log_query(endpoint="http://localhost:3100", logql="invalid{{{")

    def test_connection_error(self):
        with patch("oprim._metrics_logs.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                loki_log_query(endpoint="http://nonexistent:3100", logql='{app="x"}')

    def test_loki_timeout_error(self):
        with patch("oprim._metrics_logs.httpx.get", side_effect=httpx.TimeoutException("timeout")):
            with pytest.raises(OprimConnectionError):
                loki_log_query(endpoint="http://localhost:3100", logql='{app="x"}')

    def test_loki_server_error(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 503
        resp.is_success = False
        resp.text = "unavailable"
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            with pytest.raises(OprimConnectionError):
                loki_log_query(endpoint="http://localhost:3100", logql='{app="x"}')


# ---------------------------------------------------------------------------
# structlog_parse
# ---------------------------------------------------------------------------

class TestStructlogParse:
    def test_json_lines(self):
        lines = [
            '{"level": "info", "event": "started", "timestamp": "2026-05-20T10:00:00Z"}',
            '{"level": "error", "event": "failed", "code": 500}',
        ]
        result = structlog_parse(raw_lines=lines)
        assert len(result) == 2
        assert result[0]["level"] == "info"
        assert result[1]["code"] == 500

    def test_empty_input(self):
        result = structlog_parse(raw_lines=[])
        assert result == []

    def test_parse_failure_returns_raw(self):
        lines = ["not-json-at-all", '{"valid": true}']
        result = structlog_parse(raw_lines=lines)
        assert len(result) == 2
        assert "_raw" in result[0]
        assert "_parse_error" in result[0]
        assert result[1]["valid"] is True

    def test_logfmt_simple(self):
        lines = ["level=info event=started ts=2026-05-20"]
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert len(result) == 1
        assert result[0]["level"] == "info"
        assert result[0]["event"] == "started"

    def test_logfmt_quoted_value(self):
        lines = ['level=info event="user logged in" user=alice']
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert result[0]["event"] == "user logged in"
        assert result[0]["user"] == "alice"

    def test_chinese_message_in_json(self):
        lines = ['{"event": "用户登录", "user": "张三"}']
        result = structlog_parse(raw_lines=lines)
        assert result[0]["event"] == "用户登录"

    def test_blank_lines_skipped(self):
        lines = ['{"level": "info"}', "", '{"level": "warn"}']
        result = structlog_parse(raw_lines=lines)
        assert len(result) == 2

    def test_logfmt_flag_key_no_equals(self):
        lines = ["debug key1=val1 verbose"]
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert result[0]["key1"] == "val1"
        assert result[0]["verbose"] is True

    def test_logfmt_key_equals_at_end(self):
        lines = ["key="]
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert result[0]["key"] == ""

    def test_logfmt_escape_in_quoted_value(self):
        lines = ['msg="line with \\"quotes\\" inside"']
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert '"quotes"' in result[0]["msg"]

    def test_logfmt_trailing_spaces(self):
        lines = ["key=val   "]
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert result[0]["key"] == "val"

    def test_logfmt_starts_with_equals(self):
        lines = ["=value key=ok"]
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert result[0]["key"] == "ok"

    def test_logfmt_unclosed_quote(self):
        lines = ['msg="no closing quote']
        result = structlog_parse(raw_lines=lines, fmt="logfmt")
        assert "msg" in result[0]
        assert result[0]["msg"] == "no closing quote"

    def test_prometheus_unknown_result_type(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.is_success = True
        resp.json.return_value = {"data": {"resultType": "matrix", "result": []}}
        with patch("oprim._metrics_logs.httpx.get", return_value=resp):
            result = prometheus_instant_query(endpoint="http://localhost:9090", query="up")
        assert result.result_type == "vector"  # falls back to vector for unknown types
