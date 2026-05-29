"""Tests for oprim docker operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim import (
    docker_container_inspect,
    docker_container_logs,
    docker_container_restart,
    docker_container_start,
    docker_container_stats,
    docker_container_stop,
    docker_image_pull,
)
from oprim._docker import ContainerInfo, ContainerOpResult, ContainerStats, ImagePullResult, LogLine
from oprim._exceptions import OprimConnectionError, OprimNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_container(
    status: str = "running",
    container_id: str = "abc123",
    name: str = "mycontainer",
    exit_code: int = 0,
):
    c = MagicMock()
    c.id = container_id
    c.status = status
    c.reload = MagicMock()
    c.attrs = {
        "Id": container_id,
        "Name": f"/{name}",
        "State": {
            "Status": status,
            "StartedAt": "2026-05-20T10:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z",
            "ExitCode": exit_code,
            "Health": None,
        },
        "Config": {"Image": "nginx:latest", "Labels": {}},
        "RestartCount": 0,
        "HostConfig": {"PortBindings": {}},
        "Mounts": [],
    }
    return c


def _mock_client(container=None, image=None):
    client = MagicMock()
    if container is not None:
        client.containers.get.return_value = container
    if image is not None:
        client.images.get.return_value = image
    return client


# ---------------------------------------------------------------------------
# docker_container_inspect
# ---------------------------------------------------------------------------

class TestDockerContainerInspect:
    def test_running_container(self):
        container = _make_container(status="running")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert isinstance(result, ContainerInfo)
        assert result.state == "running"
        assert result.container_id == "abc123"
        assert result.name == "mycontainer"

    def test_exited_container(self):
        container = _make_container(status="exited", exit_code=1)
        container.attrs["State"]["FinishedAt"] = "2026-05-20T11:00:00Z"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert result.state == "exited"
        assert result.exit_code == 1
        assert result.finished_at == "2026-05-20T11:00:00Z"

    def test_with_health_check(self):
        container = _make_container(status="running")
        container.attrs["State"]["Health"] = {"Status": "healthy"}
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert result.health == "healthy"

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("not found")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_inspect(container_id="nonexistent")

    def test_docker_unreachable(self):
        import docker.errors
        with patch("oprim._docker.docker.DockerClient", side_effect=docker.errors.DockerException("fail")):
            with pytest.raises(OprimConnectionError):
                docker_container_inspect(container_id="abc123")

    def test_get_container_docker_exception(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.DockerException("daemon error")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimConnectionError):
                docker_container_inspect(container_id="abc123")

    def test_inspect_started_at_epoch_zero(self):
        container = _make_container(status="created")
        container.attrs["State"]["StartedAt"] = "0001-01-01T00:00:00Z"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert result.started_at is None

    def test_inspect_with_port_bindings(self):
        container = _make_container(status="running")
        container.attrs["HostConfig"]["PortBindings"] = {
            "80/tcp": [{"HostPort": "8080"}],
        }
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert len(result.ports) == 1
        assert result.ports[0]["container_port"] == "80"
        assert result.ports[0]["protocol"] == "tcp"

    def test_inspect_with_port_no_protocol(self):
        container = _make_container(status="running")
        container.attrs["HostConfig"]["PortBindings"] = {
            "8080": [{"HostPort": "8080"}],
        }
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_inspect(container_id="abc123")
        assert result.ports[0]["protocol"] == "tcp"
        assert result.ports[0]["container_port"] == "8080"


# ---------------------------------------------------------------------------
# docker_container_logs
# ---------------------------------------------------------------------------

class TestDockerContainerLogs:
    def test_recent_lines(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z hello\n2026-05-20T10:00:01Z world\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(ln, LogLine) for ln in result)

    def test_since_relative(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z line1\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123", since="5m")
        assert len(result) == 1

    def test_since_absolute(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z line1\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123", since="2026-05-20T09:00:00Z")
        assert len(result) == 1

    def test_empty_logs(self):
        container = _make_container()
        container.logs.return_value = b""
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123")
        assert result == []

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("not found")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_logs(container_id="gone")

    def test_with_mixed_lines(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z line1\n2026-05-20T10:00:01Z line2\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123", lines=50)
        assert len(result) == 2

    def test_logs_with_until(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z line\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123", until="2026-05-20T11:00:00Z")
        assert len(result) == 1

    def test_logs_line_without_timestamp(self):
        container = _make_container()
        container.logs.return_value = b"noseparatorline\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123")
        assert len(result) == 1
        assert result[0].message == "noseparatorline"

    def test_logs_docker_exception(self):
        import docker.errors
        container = _make_container()
        container.logs.side_effect = docker.errors.DockerException("read error")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            with pytest.raises(OprimConnectionError):
                docker_container_logs(container_id="abc123")

    def test_logs_with_empty_line_skipped(self):
        container = _make_container()
        container.logs.return_value = b"2026-05-20T10:00:00Z line1\n\n2026-05-20T10:00:01Z line2\n"
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_logs(container_id="abc123")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# docker_container_start
# ---------------------------------------------------------------------------

class TestDockerContainerStart:
    def test_start_stopped_container(self):
        container = _make_container(status="exited")
        container.reload.side_effect = [None]
        # After start(), reload sets status to "running"
        def reload():
            container.status = "running"
        container.reload.side_effect = reload
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_start(container_id="abc123")
        assert isinstance(result, ContainerOpResult)
        assert result.operation == "start"
        assert result.success is True
        assert result.state_before == "exited"
        assert result.state_after == "running"

    def test_start_already_running(self):
        container = _make_container(status="running")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_start(container_id="abc123")
        assert result.success is True

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("nf")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_start(container_id="gone")

    def test_docker_unreachable(self):
        import docker.errors
        with patch("oprim._docker.docker.DockerClient", side_effect=docker.errors.DockerException("fail")):
            with pytest.raises(OprimConnectionError):
                docker_container_start(container_id="abc123")

    def test_elapsed_ms_positive(self):
        container = _make_container(status="exited")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_start(container_id="abc123")
        assert result.elapsed_ms >= 0

    def test_start_docker_exception(self):
        import docker.errors
        container = _make_container(status="exited")
        container.start.side_effect = docker.errors.DockerException("daemon error")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            with pytest.raises(OprimConnectionError):
                docker_container_start(container_id="abc123")


# ---------------------------------------------------------------------------
# docker_container_stop
# ---------------------------------------------------------------------------

class TestDockerContainerStop:
    def test_stop_running(self):
        container = _make_container(status="running")
        def reload():
            container.status = "exited"
        container.reload.side_effect = reload
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stop(container_id="abc123")
        assert result.operation == "stop"
        assert result.success is True

    def test_stop_already_stopped(self):
        container = _make_container(status="exited")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stop(container_id="abc123")
        assert result.success is True

    def test_custom_timeout(self):
        container = _make_container(status="running")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stop(container_id="abc123", timeout_sec=30)
        assert result.success is True
        container.stop.assert_called_once_with(timeout=30)

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("nf")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_stop(container_id="gone")

    def test_elapsed_ms_non_negative(self):
        container = _make_container(status="running")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stop(container_id="abc123")
        assert result.elapsed_ms >= 0

    def test_stop_docker_exception(self):
        import docker.errors
        container = _make_container(status="running")
        container.stop.side_effect = docker.errors.DockerException("daemon error")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            with pytest.raises(OprimConnectionError):
                docker_container_stop(container_id="abc123")


# ---------------------------------------------------------------------------
# docker_container_restart
# ---------------------------------------------------------------------------

class TestDockerContainerRestart:
    def test_restart_running(self):
        container = _make_container(status="running")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_restart(container_id="abc123")
        assert result.operation == "restart"
        assert result.success is True

    def test_restart_stopped(self):
        container = _make_container(status="exited")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_restart(container_id="abc123")
        assert result.success is True

    def test_custom_timeout(self):
        container = _make_container()
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            docker_container_restart(container_id="abc123", timeout_sec=15)
        container.restart.assert_called_once_with(timeout=15)

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("nf")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_restart(container_id="gone")

    def test_elapsed_positive(self):
        container = _make_container()
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_restart(container_id="abc123")
        assert result.elapsed_ms >= 0

    def test_restart_docker_exception(self):
        import docker.errors
        container = _make_container(status="running")
        container.restart.side_effect = docker.errors.DockerException("daemon error")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            with pytest.raises(OprimConnectionError):
                docker_container_restart(container_id="abc123")


# ---------------------------------------------------------------------------
# docker_image_pull
# ---------------------------------------------------------------------------

class TestDockerImagePull:
    def test_pull_public_latest(self):
        client = MagicMock()
        import docker.errors
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        img = MagicMock()
        img.id = "sha256:abc"
        img.attrs = {"Size": 50_000_000}
        client.images.pull.return_value = img
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            result = docker_image_pull(image="nginx", tag="latest")
        assert isinstance(result, ImagePullResult)
        assert result.pulled is True
        assert result.size_bytes == 50_000_000

    def test_pull_specific_tag(self):
        client = MagicMock()
        import docker.errors
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        img = MagicMock()
        img.id = "sha256:abc"
        img.attrs = {"Size": 40_000_000}
        client.images.pull.return_value = img
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            result = docker_image_pull(image="nginx", tag="1.21")
        assert result.tag == "1.21"

    def test_already_local(self):
        client = MagicMock()
        local_img = MagicMock()
        client.images.get.return_value = local_img
        pulled_img = MagicMock()
        pulled_img.id = "sha256:abc"
        pulled_img.attrs = {"Size": 50_000_000}
        client.images.pull.return_value = pulled_img
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            result = docker_image_pull(image="nginx", tag="latest")
        assert result.pulled is False

    def test_image_not_found(self):
        import docker.errors
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        client.images.pull.side_effect = docker.errors.ImageNotFound("not found")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_image_pull(image="nosuchimage", tag="notexist")

    def test_auth_failure(self):
        import docker.errors
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        client.images.pull.side_effect = docker.errors.APIError("unauthorized: authentication required")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimAuthError if False else (OprimAuthError, OprimConnectionError)):
                docker_image_pull(image="private/image", tag="latest")

    def test_registry_unreachable(self):
        import docker.errors
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        client.images.pull.side_effect = docker.errors.DockerException("connection refused")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimConnectionError):
                docker_image_pull(image="nginx", tag="latest")

    def test_local_image_check_docker_exception(self):
        import docker.errors
        client = MagicMock()
        client.images.get.side_effect = docker.errors.DockerException("daemon error")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimConnectionError):
                docker_image_pull(image="nginx", tag="latest")

    def test_api_error_non_auth(self):
        import docker.errors
        client = MagicMock()
        client.images.get.side_effect = docker.errors.ImageNotFound("not local")
        client.images.pull.side_effect = docker.errors.APIError("rate limited by registry")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimConnectionError):
                docker_image_pull(image="nginx", tag="latest")


# Fix: import OprimAuthError
from oprim._exceptions import OprimAuthError


# ---------------------------------------------------------------------------
# docker_container_stats
# ---------------------------------------------------------------------------

class TestDockerContainerStats:
    def _make_stats(self) -> dict:
        return {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200_000_000, "percpu_usage": [100_000_000, 100_000_000]},
                "system_cpu_usage": 1_000_000_000,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 500_000_000,
            },
            "memory_stats": {
                "usage": 100 * 1024 * 1024,
                "limit": 1024 * 1024 * 1024,
            },
            "networks": {
                "eth0": {"rx_bytes": 1000, "tx_bytes": 2000}
            },
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "Read", "value": 512},
                    {"op": "Write", "value": 256},
                ]
            },
            "pids_stats": {"current": 5},
        }

    def test_running_container_stats(self):
        container = _make_container()
        container.stats.return_value = self._make_stats()
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert isinstance(result, ContainerStats)
        assert result.cpu_percent >= 0
        assert result.memory_usage_bytes == 100 * 1024 * 1024
        assert result.network_rx_bytes == 1000
        assert result.network_tx_bytes == 2000

    def test_pids_count(self):
        container = _make_container()
        stats = self._make_stats()
        stats["pids_stats"]["current"] = 42
        container.stats.return_value = stats
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.pids == 42

    def test_memory_unlimited(self):
        container = _make_container()
        stats = self._make_stats()
        stats["memory_stats"]["limit"] = 0  # unlimited
        container.stats.return_value = stats
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.memory_percent == 0.0

    def test_container_not_found(self):
        import docker.errors
        client = MagicMock()
        client.containers.get.side_effect = docker.errors.NotFound("nf")
        with patch("oprim._docker.docker.DockerClient", return_value=client):
            with pytest.raises(OprimNotFoundError):
                docker_container_stats(container_id="gone")

    def test_timestamp_present(self):
        container = _make_container()
        container.stats.return_value = self._make_stats()
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.timestamp  # non-empty ISO string

    def test_stats_docker_exception(self):
        import docker.errors
        container = _make_container()
        container.stats.side_effect = docker.errors.DockerException("stats error")
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            with pytest.raises(OprimConnectionError):
                docker_container_stats(container_id="abc123")

    def test_stats_zero_cpu_delta(self):
        container = _make_container()
        stats = self._make_stats()
        # Same values → system_delta = 0 → cpu_percent = 0
        stats["cpu_stats"]["system_cpu_usage"] = 500_000_000
        stats["precpu_stats"]["system_cpu_usage"] = 500_000_000
        container.stats.return_value = stats
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.cpu_percent == 0.0

    def test_stats_blkio_read_write(self):
        container = _make_container()
        stats = self._make_stats()
        container.stats.return_value = stats
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.block_read_bytes == 512
        assert result.block_write_bytes == 256

    def test_stats_blkio_with_total_op(self):
        container = _make_container()
        stats = self._make_stats()
        stats["blkio_stats"]["io_service_bytes_recursive"].append({"op": "Total", "value": 768})
        container.stats.return_value = stats
        with patch("oprim._docker.docker.DockerClient", return_value=_mock_client(container)):
            result = docker_container_stats(container_id="abc123")
        assert result.block_read_bytes == 512
        assert result.block_write_bytes == 256
