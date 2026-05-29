"""Tests for oprim RabbitMQ operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from oprim import (
    rabbitmq_connection_status,
    rabbitmq_consumer_status,
    rabbitmq_node_status,
    rabbitmq_queue_status,
)
from oprim._exceptions import OprimAuthError, OprimConnectionError, OprimNotFoundError
from oprim._rabbitmq import ConnectionsStatus, ConsumerStatus, NodeStatus, QueueStatus


def _mock_get(status_code: int = 200, json_data=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# ---------------------------------------------------------------------------
# rabbitmq_queue_status
# ---------------------------------------------------------------------------

class TestRabbitmqQueueStatus:
    def _queue_data(self, messages=5):
        return {
            "name": "my-queue",
            "vhost": "/",
            "messages": messages,
            "messages_ready": messages - 1,
            "messages_unacknowledged": 1,
            "consumers": 2,
            "state": "running",
            "memory": 65536,
            "disk_reads": 100,
            "messages_persistent": messages,
            "consumer_utilisation": 0.75,
        }

    def test_normal_queue(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._queue_data())):
            result = rabbitmq_queue_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="my-queue",
            )
        assert isinstance(result, QueueStatus)
        assert result.messages == 5
        assert result.state == "running"
        assert result.consumer_utilisation == pytest.approx(0.75)

    def test_queue_empty(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._queue_data(0))):
            result = rabbitmq_queue_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="my-queue",
            )
        assert result.messages == 0

    def test_queue_not_found(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(404)):
            with pytest.raises(OprimNotFoundError):
                rabbitmq_queue_status(
                    mgmt_url="http://guest:guest@localhost:15672/api/",
                    queue_name="nonexistent",
                )

    def test_auth_failure(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(401)):
            with pytest.raises(OprimAuthError):
                rabbitmq_queue_status(
                    mgmt_url="http://wrong:creds@localhost:15672/api/",
                    queue_name="q",
                )

    def test_connection_error(self):
        with patch("oprim._rabbitmq.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                rabbitmq_queue_status(
                    mgmt_url="http://guest:guest@nonexistent:15672/api/",
                    queue_name="q",
                )


# ---------------------------------------------------------------------------
# rabbitmq_connection_status
# ---------------------------------------------------------------------------

class TestRabbitmqConnectionStatus:
    def _conn_data(self):
        return [
            {"name": "conn1", "state": "running", "channels": 1, "recv_oct": 100, "send_oct": 200, "peer_host": "10.0.0.1", "user": "guest"},
            {"name": "conn2", "state": "blocked", "channels": 0, "recv_oct": 50, "send_oct": 50, "peer_host": "10.0.0.2", "user": "guest"},
            {"name": "conn3", "state": "running", "channels": 2, "recv_oct": 300, "send_oct": 400, "peer_host": "10.0.0.3", "user": "guest"},
        ]

    def test_normal_connections(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._conn_data())):
            result = rabbitmq_connection_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
            )
        assert isinstance(result, ConnectionsStatus)
        assert result.total == 3
        assert result.blocked == 1
        assert result.running == 2

    def test_no_connections(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, [])):
            result = rabbitmq_connection_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
            )
        assert result.total == 0
        assert result.blocked == 0

    def test_all_running(self):
        data = [
            {"name": f"conn{i}", "state": "running", "channels": 1, "recv_oct": 0, "send_oct": 0, "peer_host": "", "user": ""}
            for i in range(5)
        ]
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, data)):
            result = rabbitmq_connection_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
            )
        assert result.blocked == 0
        assert result.running == 5

    def test_auth_failure(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(401)):
            with pytest.raises(OprimAuthError):
                rabbitmq_connection_status(mgmt_url="http://bad:creds@localhost:15672/api/")

    def test_connection_error(self):
        with patch("oprim._rabbitmq.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                rabbitmq_connection_status(mgmt_url="http://guest:guest@nonexistent:15672/api/")


# ---------------------------------------------------------------------------
# rabbitmq_consumer_status
# ---------------------------------------------------------------------------

class TestRabbitmqConsumerStatus:
    def _consumer_data(self, queue_name="my-queue"):
        return [
            {
                "consumer_tag": "ctag1",
                "channel_details": {"name": "conn1.ch1"},
                "queue": {"name": queue_name},
                "prefetch_count": 10,
                "ack_required": True,
                "active": True,
            },
            {
                "consumer_tag": "ctag2",
                "channel_details": {"name": "conn2.ch1"},
                "queue": {"name": "other-queue"},
                "prefetch_count": 5,
                "ack_required": True,
                "active": True,
            },
        ]

    def test_with_consumers(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._consumer_data())):
            result = rabbitmq_consumer_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="my-queue",
            )
        assert isinstance(result, ConsumerStatus)
        assert result.consumer_count == 1
        assert result.consumers[0].consumer_tag == "ctag1"

    def test_no_consumers(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, [])):
            result = rabbitmq_consumer_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="empty-queue",
            )
        assert result.consumer_count == 0
        assert result.consumers == []

    def test_consumer_fields(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._consumer_data())):
            result = rabbitmq_consumer_status(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="my-queue",
            )
        c = result.consumers[0]
        assert c.prefetch_count == 10
        assert c.ack_required is True
        assert c.active is True

    def test_auth_failure(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(401)):
            with pytest.raises(OprimAuthError):
                rabbitmq_consumer_status(
                    mgmt_url="http://bad:creds@localhost:15672/api/",
                    queue_name="q",
                )

    def test_connection_error(self):
        with patch("oprim._rabbitmq.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                rabbitmq_consumer_status(
                    mgmt_url="http://guest:guest@nonexistent:15672/api/",
                    queue_name="q",
                )


# ---------------------------------------------------------------------------
# rabbitmq_node_status
# ---------------------------------------------------------------------------

class TestRabbitmqNodeStatus:
    def _node_data(self):
        return [
            {
                "name": "rabbit@node1",
                "type": "disc",
                "running": True,
                "mem_used": 500_000_000,
                "mem_limit": 2_000_000_000,
                "mem_alarm": False,
                "disk_free": 50_000_000_000,
                "disk_free_limit": 1_000_000_000,
                "disk_free_alarm": False,
                "fd_used": 100,
                "fd_total": 1024,
                "sockets_used": 50,
                "sockets_total": 512,
                "proc_used": 200,
                "proc_total": 1_000_000,
            }
        ]

    def test_single_node(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, self._node_data())):
            result = rabbitmq_node_status(mgmt_url="http://guest:guest@localhost:15672/api/")
        assert len(result) == 1
        assert isinstance(result[0], NodeStatus)
        assert result[0].name == "rabbit@node1"
        assert result[0].running is True
        assert result[0].mem_alarm is False

    def test_cluster_multiple_nodes(self):
        nodes = self._node_data() * 3
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, nodes)):
            result = rabbitmq_node_status(mgmt_url="http://guest:guest@localhost:15672/api/")
        assert len(result) == 3

    def test_node_with_alarm(self):
        data = self._node_data()
        data[0]["mem_alarm"] = True
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(200, data)):
            result = rabbitmq_node_status(mgmt_url="http://guest:guest@localhost:15672/api/")
        assert result[0].mem_alarm is True

    def test_auth_failure(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(401)):
            with pytest.raises(OprimAuthError):
                rabbitmq_node_status(mgmt_url="http://bad:creds@localhost:15672/api/")

    def test_connection_error(self):
        with patch("oprim._rabbitmq.httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(OprimConnectionError):
                rabbitmq_node_status(mgmt_url="http://guest:guest@nonexistent:15672/api/")

    def test_timeout_error(self):
        with patch("oprim._rabbitmq.httpx.get", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(OprimConnectionError):
                rabbitmq_node_status(mgmt_url="http://guest:guest@localhost:15672/api/")

    def test_server_error(self):
        with patch("oprim._rabbitmq.httpx.get", return_value=_mock_get(503)):
            with pytest.raises(OprimConnectionError):
                rabbitmq_node_status(mgmt_url="http://guest:guest@localhost:15672/api/")
