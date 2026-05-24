import pytest
from unittest.mock import MagicMock, patch
import docker.errors
from oprim import (
    docker_image_list,
    docker_image_delete,
    docker_volume_list,
    docker_volume_delete,
    docker_network_list,
    docker_container_list,
)
from oprim._exceptions import OprimNotFoundError, OprimConnectionError

@patch("docker.DockerClient")
def test_docker_image_list(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_image = MagicMock()
    mock_image.id = "sha256:123"
    mock_image.tags = ["nginx:latest"]
    mock_image.attrs = {"Size": 1000, "Created": "2024-01-01"}
    mock_client.images.list.return_value = [mock_image]

    res = docker_image_list()
    assert len(res) == 1
    assert res[0]["id"] == "sha256:123"
    assert res[0]["tags"] == ["nginx:latest"]
    assert res[0]["size_bytes"] == 1000

@patch("docker.DockerClient")
def test_docker_image_list_error(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.images.list.side_effect = docker.errors.DockerException("error")
    with pytest.raises(OprimConnectionError):
        docker_image_list()

@patch("docker.DockerClient")
def test_docker_image_delete(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.images.remove.return_value = "deleted"
    res = docker_image_delete(image="nginx:latest")
    assert res["result"] == "deleted"

@patch("docker.DockerClient")
def test_docker_image_delete_not_found(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.images.remove.side_effect = docker.errors.ImageNotFound("not found")
    with pytest.raises(OprimNotFoundError):
        docker_image_delete(image="ghost")

@patch("docker.DockerClient")
def test_docker_volume_list(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_vol = MagicMock()
    mock_vol.name = "myvol"
    mock_vol.attrs = {"Driver": "local", "Mountpoint": "/var/lib/docker/volumes/myvol/_data", "CreatedAt": "2024-01-01", "Labels": {}}
    mock_client.volumes.list.return_value = [mock_vol]

    res = docker_volume_list()
    assert len(res) == 1
    assert res[0]["name"] == "myvol"

@patch("docker.DockerClient")
def test_docker_volume_delete(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_vol = MagicMock()
    mock_client.volumes.get.return_value = mock_vol
    res = docker_volume_delete(name="myvol")
    assert res["deleted"] == "myvol"
    mock_vol.remove.assert_called_once()

@patch("docker.DockerClient")
def test_docker_volume_list_error(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.volumes.list.side_effect = docker.errors.DockerException("error")
    with pytest.raises(OprimConnectionError):
        docker_volume_list()

@patch("docker.DockerClient")
def test_docker_volume_delete_not_found(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.volumes.get.side_effect = docker.errors.NotFound("not found")
    with pytest.raises(OprimNotFoundError):
        docker_volume_delete(name="ghost")

@patch("docker.DockerClient")
def test_docker_volume_delete_error(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_vol = MagicMock()
    mock_client.volumes.get.return_value = mock_vol
    mock_vol.remove.side_effect = docker.errors.DockerException("error")
    with pytest.raises(OprimConnectionError):
        docker_volume_delete(name="myvol")

@patch("docker.DockerClient")
def test_docker_network_list(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_net = MagicMock()
    mock_net.id = "net123"
    mock_net.name = "bridge"
    mock_net.attrs = {"Driver": "bridge", "Scope": "local", "Internal": False}
    mock_client.networks.list.return_value = [mock_net]

    res = docker_network_list()
    assert len(res) == 1
    assert res[0]["name"] == "bridge"

@patch("docker.DockerClient")
def test_docker_image_list_empty(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.images.list.return_value = []
    res = docker_image_list()
    assert res == []

@patch("docker.DockerClient")
def test_docker_image_list_host(mock_client_class):
    docker_image_list(docker_host="tcp://1.2.3.4:2375")
    mock_client_class.assert_called_with(base_url="tcp://1.2.3.4:2375")

@patch("docker.DockerClient")
def test_docker_image_list_multiple(mock_client_class):
    mock_client = mock_client_class.return_value
    img1 = MagicMock(id="1", tags=["t1"], attrs={"Size": 1, "Created": "2024"})
    img2 = MagicMock(id="2", tags=["t2"], attrs={"Size": 2, "Created": "2024"})
    mock_client.images.list.return_value = [img1, img2]
    res = docker_image_list()
    assert len(res) == 2

@patch("docker.DockerClient")
def test_docker_image_delete_force(mock_client_class):
    mock_client = mock_client_class.return_value
    docker_image_delete(image="img", force=True)
    mock_client.images.remove.assert_called_with("img", force=True)

@patch("docker.DockerClient")
def test_docker_image_delete_host(mock_client_class):
    docker_image_delete(image="img", docker_host="unix:///tmp/docker.sock")
    mock_client_class.assert_called_with(base_url="unix:///tmp/docker.sock")

@patch("docker.DockerClient")
def test_docker_volume_list_empty(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.volumes.list.return_value = []
    res = docker_volume_list()
    assert res == []

@patch("docker.DockerClient")
def test_docker_volume_list_labels(mock_client_class):
    mock_client = mock_client_class.return_value
    vol = MagicMock(name="v")
    vol.attrs = {"Driver": "l", "Mountpoint": "m", "CreatedAt": "c", "Labels": {"foo": "bar"}}
    mock_client.volumes.list.return_value = [vol]
    res = docker_volume_list()
    assert res[0]["labels"] == {"foo": "bar"}

@patch("docker.DockerClient")
def test_docker_volume_delete_force(mock_client_class):
    mock_client = mock_client_class.return_value
    vol = MagicMock()
    mock_client.volumes.get.return_value = vol
    docker_volume_delete(name="v", force=True)
    vol.remove.assert_called_with(force=True)

@patch("docker.DockerClient")
def test_docker_network_list_empty(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.networks.list.return_value = []
    res = docker_network_list()
    assert res == []

@patch("docker.DockerClient")
def test_docker_network_list_host(mock_client_class):
    docker_network_list(docker_host="tcp://remote:2375")
    mock_client_class.assert_called_with(base_url="tcp://remote:2375")

@patch("docker.DockerClient")
def test_docker_network_list_error(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_client.networks.list.side_effect = docker.errors.DockerException("error")
    with pytest.raises(OprimConnectionError):
        docker_network_list()

@patch("docker.DockerClient")
def test_docker_container_list(mock_client_class):
    mock_client = mock_client_class.return_value
    mock_c = MagicMock()
    mock_c.id = "c1"
    mock_c.name = "con1"
    mock_c.image.tags = ["img:1"]
    mock_c.status = "running"
    mock_c.attrs = {"State": {"Status": "running"}, "Config": {"Labels": {}}}
    mock_client.containers.list.return_value = [mock_c]
    
    res = docker_container_list()
    assert len(res) == 1
    assert res[0].container_id == "c1"
