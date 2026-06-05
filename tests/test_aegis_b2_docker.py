"""B2 Docker new elements tests — docker_logs/ps/restart/stats/inspect/compose_*/docker_compose_pull."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from oprim import (
    docker_logs,
    docker_ps,
    docker_restart,
    docker_stats,
    docker_inspect,
    docker_compose_up,
    docker_compose_down,
    docker_compose_pull,
)
from oprim._docker import (
    docker_container_logs,
    docker_container_list,
    docker_container_restart,
    docker_container_stats,
    docker_container_inspect,
    compose_up,
    compose_down,
)
from oprim._exceptions import OprimConnectionError, OprimNotFoundError


# ===== Alias identity tests =====


def test_docker_logs_is_alias():
    assert docker_logs is docker_container_logs


def test_docker_ps_is_alias():
    assert docker_ps is docker_container_list


def test_docker_restart_is_alias():
    assert docker_restart is docker_container_restart


def test_docker_stats_is_alias():
    assert docker_stats is docker_container_stats


def test_docker_inspect_is_alias():
    assert docker_inspect is docker_container_inspect


def test_docker_compose_up_is_alias():
    assert docker_compose_up is compose_up


def test_docker_compose_down_is_alias():
    assert docker_compose_down is compose_down


# ===== docker_compose_pull unit tests =====


def test_docker_compose_pull_file_not_found(tmp_path):
    """Missing compose file raises OprimNotFoundError."""
    with pytest.raises(OprimNotFoundError, match="not found"):
        docker_compose_pull(compose_file=str(tmp_path / "nonexistent.yml"))


def test_docker_compose_pull_success(tmp_path):
    """docker_compose_pull calls subprocess and returns stdout/stderr."""
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("version: '3'\nservices:\n  app:\n    image: nginx\n")

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = "Pulling nginx..."
    mock_proc.returncode = 0

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        result = docker_compose_pull(compose_file=str(compose_file))

    assert "stdout" in result
    assert "stderr" in result
    cmd = mock_run.call_args[0][0]
    assert "pull" in cmd
    assert str(compose_file) in cmd


def test_docker_compose_pull_with_project_name(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("version: '3'\n")

    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        docker_compose_pull(compose_file=str(compose_file), project_name="myproj")

    cmd = mock_run.call_args[0][0]
    assert "-p" in cmd
    assert "myproj" in cmd


def test_docker_compose_pull_subprocess_error(tmp_path):
    """CalledProcessError converts to OprimConnectionError."""
    import subprocess

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("version: '3'\n")

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "docker", stderr="pull failed"),
    ):
        with pytest.raises(OprimConnectionError, match="pull failed"):
            docker_compose_pull(compose_file=str(compose_file))


def test_docker_compose_pull_custom_docker_host(tmp_path):
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("version: '3'\n")
    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        docker_compose_pull(
            compose_file=str(compose_file),
            docker_host="tcp://remote:2375",
        )

    env = mock_run.call_args[1]["env"]
    assert env["DOCKER_HOST"] == "tcp://remote:2375"
