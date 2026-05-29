import pytest
from unittest.mock import MagicMock, patch
import subprocess
from oprim import compose_up, compose_down
from oprim._exceptions import OprimNotFoundError, OprimConnectionError

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_up(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_proc = MagicMock()
    mock_proc.stdout = "up done"
    mock_proc.stderr = "Container c1 Created\nContainer c1 Started"
    mock_run.return_value = mock_proc

    res = compose_up(compose_file="docker-compose.yml")
    assert "c1" in res["started_services"]
    assert res["stdout"] == "up done"

@patch("os.path.exists")
def test_compose_up_not_found(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(OprimNotFoundError):
        compose_up(compose_file="missing.yml")

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_up_fail(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")
    with pytest.raises(OprimConnectionError):
        compose_up(compose_file="fail.yml")

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_down(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_proc = MagicMock()
    mock_proc.stdout = "down done"
    mock_run.return_value = mock_proc

    res = compose_down(compose_file="docker-compose.yml")
    assert res["stdout"] == "down done"

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_up_parsing(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_proc = MagicMock()
    mock_proc.stderr = " Container svc1 Created\n Container svc2 Started\n Container svc1 Started"
    mock_run.return_value = mock_proc
    res = compose_up(compose_file="docker-compose.yml")
    assert "svc1" in res["started_services"]
    assert "svc2" in res["started_services"]

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_up_generic_error(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.side_effect = Exception("generic error")
    with pytest.raises(OprimConnectionError, match="Failed to execute docker compose"):
        compose_up(compose_file="fail.yml")

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_down_volumes(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.return_value = MagicMock()
    compose_down(compose_file="docker-compose.yml", volumes=True)
    args, kwargs = mock_run.call_args
    assert "-v" in args[0]

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_down_error(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")
    with pytest.raises(OprimConnectionError):
        compose_down(compose_file="fail.yml")

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_down_generic_error(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.side_effect = Exception("error")
    with pytest.raises(OprimConnectionError):
        compose_down(compose_file="fail.yml")

@patch("os.path.exists")
@patch("subprocess.run")
def test_compose_down_project(mock_run, mock_exists):
    mock_exists.return_value = True
    mock_run.return_value = MagicMock()
    compose_down(compose_file="docker-compose.yml", project_name="myproj")
    args, kwargs = mock_run.call_args
    assert "-p" in args[0]
    assert "myproj" in args[0]
