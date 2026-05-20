"""Tests for oprim.rabbitmq_queue_status."""
from __future__ import annotations

import json

import httpx
import pytest

from oprim._exceptions import OprimError
from oprim.rabbitmq_queue_status import rabbitmq_queue_status


class TestRabbitMQQueueStatus:
    def test_happy_path_normal_queue(self, mock_rabbitmq_mgmt, rabbitmq_fixtures_dir):
        """Healthy queue returns expected RabbitMQQueueStatus."""
        with open(rabbitmq_fixtures_dir / "queue_normal.json") as f:
            payload = json.load(f)
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(return_value=httpx.Response(200, json=payload))

        result = rabbitmq_queue_status(
            mgmt_url="http://rabbit.test:15672",
            queue_name="signals",
            username="guest",
            password="guest",
        )

        assert result.queue_name == "signals"
        assert result.consumers == 4
        assert result.state == "running"
        assert result.messages_ready == 12

    def test_queue_buildup_scenario(self, mock_rabbitmq_mgmt, rabbitmq_fixtures_dir):
        """Buildup scenario with high messages_ready and low consumer count."""
        with open(rabbitmq_fixtures_dir / "queue_buildup.json") as f:
            payload = json.load(f)
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(return_value=httpx.Response(200, json=payload))

        result = rabbitmq_queue_status(
            mgmt_url="http://rabbit.test:15672",
            queue_name="signals",
            username="guest",
            password="guest",
        )

        assert result.messages_ready == 45000
        assert result.consumers == 2

    def test_queue_not_found_404(self, mock_rabbitmq_mgmt, rabbitmq_fixtures_dir):
        """404 response raises OprimError with 'Queue not found'."""
        with open(rabbitmq_fixtures_dir / "queue_not_found.json") as f:
            payload = json.load(f)
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/nonexistent"
        ).mock(return_value=httpx.Response(404, json=payload))

        with pytest.raises(OprimError, match="Queue not found"):
            rabbitmq_queue_status(
                mgmt_url="http://rabbit.test:15672",
                queue_name="nonexistent",
                username="guest",
                password="guest",
            )

    def test_auth_failure_401(self, mock_rabbitmq_mgmt):
        """401 response raises OprimError with 'authentication failed'."""
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(return_value=httpx.Response(401, json={"error": "not_authorised"}))

        with pytest.raises(OprimError, match="authentication failed"):
            rabbitmq_queue_status(
                mgmt_url="http://rabbit.test:15672",
                queue_name="signals",
                username="bad",
                password="bad",
            )

    def test_management_api_500(self, mock_rabbitmq_mgmt):
        """5xx response raises OprimError with status code."""
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(return_value=httpx.Response(500, text="Internal Server Error"))

        with pytest.raises(OprimError, match="500"):
            rabbitmq_queue_status(
                mgmt_url="http://rabbit.test:15672",
                queue_name="signals",
                username="guest",
                password="guest",
            )

    def test_vhost_url_encoding(self, mock_rabbitmq_mgmt, rabbitmq_fixtures_dir):
        """Non-default vhost is URL-encoded correctly."""
        with open(rabbitmq_fixtures_dir / "queue_normal.json") as f:
            payload = json.load(f)
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/my%2Fvhost/signals"
        ).mock(return_value=httpx.Response(200, json=payload))

        result = rabbitmq_queue_status(
            mgmt_url="http://rabbit.test:15672",
            queue_name="signals",
            vhost="my/vhost",
            username="guest",
            password="guest",
        )
        assert result.queue_name == "signals"

    def test_no_consumers_scenario(self, mock_rabbitmq_mgmt, rabbitmq_fixtures_dir):
        """Queue with zero consumers."""
        with open(rabbitmq_fixtures_dir / "queue_no_consumers.json") as f:
            payload = json.load(f)
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(return_value=httpx.Response(200, json=payload))

        result = rabbitmq_queue_status(
            mgmt_url="http://rabbit.test:15672",
            queue_name="signals",
            username="guest",
            password="guest",
        )
        assert result.consumers == 0
        assert result.state == "idle"

    def test_timeout(self, mock_rabbitmq_mgmt):
        """TimeoutException raises OprimError with 'timeout'."""
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(side_effect=httpx.TimeoutException)

        with pytest.raises(OprimError, match="timeout"):
            rabbitmq_queue_status(
                mgmt_url="http://rabbit.test:15672",
                queue_name="signals",
                username="guest",
                password="guest",
            )

    def test_connect_error(self, mock_rabbitmq_mgmt):
        """ConnectError raises OprimError with 'connect'."""
        mock_rabbitmq_mgmt.get(
            "http://rabbit.test:15672/api/queues/%2F/signals"
        ).mock(side_effect=httpx.ConnectError)

        with pytest.raises(OprimError, match="Failed to connect"):
            rabbitmq_queue_status(
                mgmt_url="http://rabbit.test:15672",
                queue_name="signals",
                username="guest",
                password="guest",
            )
