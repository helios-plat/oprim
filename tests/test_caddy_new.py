import pytest
from unittest.mock import patch, MagicMock
from oprim import caddy_admin_post
from oprim._exceptions import OprimValidationError, OprimConnectionError

@patch("httpx.post")
def test_caddy_admin_post_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_success = True
    mock_resp.json.return_value = {"status": "ok"}
    mock_post.return_value = mock_resp

    res = caddy_admin_post(admin_url="http://localhost:2019", path="/config/apps", body={"foo": "bar"})
    assert res == {"status": "ok"}
    mock_post.assert_called_once()

@patch("httpx.request")
def test_caddy_admin_patch_success(mock_request):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_success = True
    mock_resp.json.return_value = {"patched": True}
    mock_request.return_value = mock_resp

    res = caddy_admin_post(admin_url="http://localhost:2019", path="/config/...", method="PATCH")
    assert res == {"patched": True}

@patch("httpx.post")
def test_caddy_admin_post_validation_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.is_success = False
    mock_resp.text = "invalid json"
    mock_post.return_value = mock_resp

    with pytest.raises(OprimValidationError, match="HTTP 400"):
        caddy_admin_post(admin_url="http://localhost:2019", path="/config", body={})

@patch("httpx.post")
def test_caddy_admin_post_connection_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.is_success = False
    mock_resp.text = "server error"
    mock_post.return_value = mock_resp

    with pytest.raises(OprimConnectionError, match="HTTP 500"):
        caddy_admin_post(admin_url="http://localhost:2019", path="/config", body={})

@patch("httpx.post")
def test_caddy_admin_post_no_json(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_resp.is_success = True
    mock_resp.json.side_effect = Exception("not json")
    mock_post.return_value = mock_resp

    res = caddy_admin_post(admin_url="http://localhost:2019", path="/config")
    assert res["status"] == "ok"
    assert res["http_code"] == 204
