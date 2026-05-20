"""Tests for oprim.docker_logs."""
from __future__ import annotations

from unittest import mock

import pytest
from docker.errors import APIError, DockerException, NotFound

from oprim._exceptions import OprimError
from oprim.docker_logs import docker_logs


class TestDockerLogs:
    def test_happy_path(self, mock_docker_client, docker_fixtures_dir):
        """Normal logs are returned as list of lines."""
        with open(docker_fixtures_dir / "logs_normal.txt") as f:
            fixture_text = f.read()

        container_mock = mock.MagicMock()
        container_mock.id = "abc123"
        container_mock.name = "helixa-prob"
        container_mock.logs.return_value = fixture_text.encode("utf-8")
        mock_docker_client.return_value.containers.get.return_value = container_mock

        result = docker_logs(container="helixa-prob")

        assert result.container_name == "helixa-prob"
        assert result.lines_fetched > 0
        assert any("prob-engine starting up" in line for line in result.log_lines)

    def test_container_not_found(self, mock_docker_client):
        """NotFound exception raises OprimError."""
        mock_docker_client.return_value.containers.get.side_effect = NotFound(
            "container missing"
        )
        with pytest.raises(OprimError, match="Container not found"):
            docker_logs(container="missing")

    def test_daemon_unreachable(self, mock_docker_client):
        """DockerException raises OprimError."""
        mock_docker_client.return_value.containers.get.side_effect = DockerException(
            "cannot connect to daemon"
        )
        with pytest.raises(OprimError, match="Failed to connect to Docker"):
            docker_logs(container="any")

    def test_oom_logs_scenario(self, mock_docker_client, docker_fixtures_dir):
        """OOM kill log scenario."""
        with open(docker_fixtures_dir / "logs_oom_kill.txt") as f:
            fixture_text = f.read()

        container_mock = mock.MagicMock()
        container_mock.id = "def456"
        container_mock.name = "helixa-prob"
        container_mock.logs.return_value = fixture_text.encode("utf-8")
        mock_docker_client.return_value.containers.get.return_value = container_mock

        result = docker_logs(container="helixa-prob")

        assert any("OOM detected" in line for line in result.log_lines)

    def test_empty_logs(self, mock_docker_client):
        """Empty logs return zero-length result."""
        container_mock = mock.MagicMock()
        container_mock.id = "empty1"
        container_mock.name = "empty"
        container_mock.logs.return_value = b""
        mock_docker_client.return_value.containers.get.return_value = container_mock

        result = docker_logs(container="empty")

        assert result.lines_fetched == 0
        assert result.log_lines == []

    def test_since_seconds_passed(self, mock_docker_client, docker_fixtures_dir):
        """since_seconds parameter is correctly passed to docker SDK."""
        with open(docker_fixtures_dir / "logs_normal.txt") as f:
            fixture_text = f.read()

        container_mock = mock.MagicMock()
        container_mock.id = "abc"
        container_mock.name = "test"
        container_mock.logs.return_value = fixture_text.encode("utf-8")
        mock_docker_client.return_value.containers.get.return_value = container_mock

        docker_logs(container="test", since_seconds=60)

        _, kwargs = container_mock.logs.call_args
        assert "since" in kwargs

    def test_api_error_during_logs_fetch(self, mock_docker_client):
        """APIError during logs() raises OprimError."""
        container_mock = mock.MagicMock()
        container_mock.logs.side_effect = APIError("API error")
        mock_docker_client.return_value.containers.get.return_value = container_mock

        with pytest.raises(OprimError, match="Docker API error"):
            docker_logs(container="test")
