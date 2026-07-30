"""B2 RabbitMQ focused wrapper tests — rabbitmq_queue_depth / rabbitmq_consumer_count."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from oprim import rabbitmq_queue_depth, rabbitmq_consumer_count
from oprim._rabbitmq import QueueStatus
from oprim._exceptions import OprimNotFoundError, OprimConnectionError


def _make_queue_status(messages_ready=5, messages_unacked=2, consumers=3):
    return QueueStatus(
        name="test-queue",
        vhost="/",
        messages=messages_ready + messages_unacked,
        messages_ready=messages_ready,
        messages_unacked=messages_unacked,
        consumers=consumers,
        state="running",
        memory_bytes=1024,
        disk_reads=0,
        messages_persistent=0,
        consumer_utilisation=0.8,
    )


def test_rabbitmq_queue_depth_returns_ready_plus_unacked():
    status = _make_queue_status(messages_ready=5, messages_unacked=2)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status):
        result = rabbitmq_queue_depth(
            mgmt_url="http://guest:guest@localhost:15672/api/",
            queue_name="test-queue",
        )
    assert result == 7


def test_rabbitmq_queue_depth_empty_queue():
    status = _make_queue_status(messages_ready=0, messages_unacked=0)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status):
        result = rabbitmq_queue_depth(
            mgmt_url="http://guest:guest@localhost:15672/api/",
            queue_name="empty-queue",
        )
    assert result == 0


def test_rabbitmq_consumer_count_returns_consumers():
    status = _make_queue_status(consumers=4)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status):
        result = rabbitmq_consumer_count(
            mgmt_url="http://guest:guest@localhost:15672/api/",
            queue_name="test-queue",
        )
    assert result == 4


def test_rabbitmq_consumer_count_zero():
    status = _make_queue_status(consumers=0)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status):
        result = rabbitmq_consumer_count(
            mgmt_url="http://guest:guest@localhost:15672/api/",
            queue_name="idle-queue",
        )
    assert result == 0


def test_rabbitmq_queue_depth_propagates_not_found():
    with patch(
        "oprim._rabbitmq.rabbitmq_queue_status",
        side_effect=OprimNotFoundError("Queue not found"),
    ):
        with pytest.raises(OprimNotFoundError):
            rabbitmq_queue_depth(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="missing",
            )


def test_rabbitmq_consumer_count_propagates_connection_error():
    with patch(
        "oprim._rabbitmq.rabbitmq_queue_status",
        side_effect=OprimConnectionError("Cannot connect"),
    ):
        with pytest.raises(OprimConnectionError):
            rabbitmq_consumer_count(
                mgmt_url="http://guest:guest@localhost:15672/api/",
                queue_name="test",
            )


def test_rabbitmq_queue_depth_passes_vhost():
    status = _make_queue_status(messages_ready=1)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status) as mock_fn:
        rabbitmq_queue_depth(
            mgmt_url="http://localhost:15672/api/",
            queue_name="q",
            vhost="/myvhost",
        )
    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["vhost"] == "/myvhost"


def test_rabbitmq_consumer_count_passes_timeout():
    status = _make_queue_status(consumers=1)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status) as mock_fn:
        rabbitmq_consumer_count(
            mgmt_url="http://localhost:15672/api/",
            queue_name="q",
            timeout_sec=15,
        )
    call_kwargs = mock_fn.call_args.kwargs
    assert call_kwargs["timeout_sec"] == 15


def test_rabbitmq_queue_depth_large_backlog():
    status = _make_queue_status(messages_ready=10000, messages_unacked=500)
    with patch("oprim._rabbitmq.rabbitmq_queue_status", return_value=status):
        result = rabbitmq_queue_depth(
            mgmt_url="http://localhost:15672/api/",
            queue_name="busy",
        )
    assert result == 10500
