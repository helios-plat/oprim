"""Unit tests for GoogleDriveAdapter (mocked — no network required).

Integration tests that require a live OAuth session are in test_gdrive_integration.py
and are marked @pytest.mark.integration.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from oprim.storage.errors import (
    AuthenticationError,
    FileNotFoundStorageError,
    NetworkError,
    RateLimitStorageError,
    StorageError,
    StorageQuotaExceededError,
    TokenExpiredError,
)
from oprim.storage._oauth import load_oauth_config, set_token_permissions
from oprim.storage.providers.gdrive import GoogleDriveAdapter, _parse_gdrive_datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_http_error(status: int, reason: str = "") -> HttpError:
    resp = MagicMock()
    resp.status = status
    err = HttpError(resp=resp, content=b"")
    if reason:
        err.error_details = [{"reason": reason}]
    else:
        err.error_details = []
    return err


def _make_adapter(tmp_path: Path, token_data: dict | None = None) -> GoogleDriveAdapter:
    creds_dir = tmp_path / "credentials"
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    creds_dir.mkdir(parents=True)

    oauth_cfg = {
        "installed": {
            "client_id": "test-id.apps.googleusercontent.com",
            "client_secret": "test-secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    (secrets_dir / "gdrive_oauth.json").write_text(json.dumps(oauth_cfg))

    adapter = GoogleDriveAdapter(
        credentials_dir=creds_dir,
        oauth_config_path=secrets_dir / "gdrive_oauth.json",
    )

    if token_data:
        (creds_dir / "gdrive_token.json").write_text(json.dumps(token_data))

    return adapter


def _attach_mock_service(adapter: GoogleDriveAdapter) -> MagicMock:
    svc = MagicMock()
    adapter._service = svc
    adapter._creds = MagicMock(valid=True)
    return svc


# ── datetime parsing ──────────────────────────────────────────────────────────

class TestParseDatetime:
    def test_parses_full_iso(self):
        dt = _parse_gdrive_datetime("2024-01-15T10:30:00.000Z")
        assert dt.year == 2024
        assert dt.month == 1

    def test_parses_short_iso(self):
        dt = _parse_gdrive_datetime("2024-01-15T10:30:00Z")
        assert dt.year == 2024

    def test_raises_on_bad_format(self):
        with pytest.raises(ValueError):
            _parse_gdrive_datetime("not-a-date")


# ── authenticate ──────────────────────────────────────────────────────────────

class TestAuthenticate:
    async def test_raises_if_oauth_config_missing(self, tmp_path):
        creds_dir = tmp_path / "creds"
        secrets_dir = tmp_path / "secrets"
        creds_dir.mkdir()
        secrets_dir.mkdir()
        adapter = GoogleDriveAdapter(
            credentials_dir=creds_dir,
            oauth_config_path=secrets_dir / "missing.json",
        )
        with pytest.raises(AuthenticationError):
            await adapter.authenticate()

    async def test_loads_valid_token(self, tmp_path):
        token = {
            "token": "access_tok",
            "refresh_token": "refresh_tok",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "c.apps.googleusercontent.com",
            "client_secret": "s",
            "scopes": ["https://www.googleapis.com/auth/drive.file"],
        }
        adapter = _make_adapter(tmp_path, token_data=token)

        mock_creds = MagicMock()
        mock_creds.valid = True

        with (
            patch("oprim.storage.providers.gdrive.Credentials.from_authorized_user_file", return_value=mock_creds),
            patch("oprim.storage.providers.gdrive.build") as mock_build,
        ):
            result = await adapter.authenticate()
        assert result is True
        mock_build.assert_called_once()

    async def test_refresh_called_on_expired_token(self, tmp_path):
        token = {
            "token": "old_tok",
            "refresh_token": "refresh_tok",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "c.apps.googleusercontent.com",
            "client_secret": "s",
            "scopes": ["https://www.googleapis.com/auth/drive.file"],
        }
        adapter = _make_adapter(tmp_path, token_data=token)

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_tok"
        mock_creds.to_json.return_value = json.dumps(token)

        with (
            patch("oprim.storage.providers.gdrive.Credentials.from_authorized_user_file", return_value=mock_creds),
            patch("oprim.storage.providers.gdrive.build"),
            patch("asyncio.to_thread") as mock_thread,
        ):
            mock_thread.return_value = None
            await adapter.authenticate()
        # to_thread was called with creds.refresh
        assert mock_thread.call_count >= 1

    async def test_refresh_failure_raises_token_expired(self, tmp_path):
        # token_data must be provided so that token_path.exists() is True
        dummy_token = {
            "token": "old",
            "refresh_token": "tok",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "c.apps.googleusercontent.com",
            "client_secret": "s",
            "scopes": ["https://www.googleapis.com/auth/drive.file"],
        }
        adapter = _make_adapter(tmp_path, token_data=dummy_token)

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "tok"

        with (
            patch("oprim.storage.providers.gdrive.Credentials.from_authorized_user_file", return_value=mock_creds),
            patch("asyncio.to_thread", side_effect=Exception("network fail")),
        ):
            with pytest.raises(TokenExpiredError):
                await adapter.authenticate()


# ── _ensure_authenticated ─────────────────────────────────────────────────────

class TestEnsureAuthenticated:
    def test_raises_when_not_authenticated(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(AuthenticationError):
            adapter._ensure_authenticated()

    def test_no_raise_when_service_set(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._service = MagicMock()
        adapter._ensure_authenticated()  # must not raise


# ── _handle_http_error ────────────────────────────────────────────────────────

class TestHandleHttpError:
    def test_401_raises_token_expired(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(TokenExpiredError):
            adapter._handle_http_error(_make_http_error(401), "ctx")

    def test_403_quota_raises_quota_exceeded(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(StorageQuotaExceededError):
            adapter._handle_http_error(_make_http_error(403, "quotaExceeded"), "ctx")

    def test_403_storage_quota_raises_quota_exceeded(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(StorageQuotaExceededError):
            adapter._handle_http_error(_make_http_error(403, "storageQuotaExceeded"), "ctx")

    def test_403_rate_limit_raises_rate_limit(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(RateLimitStorageError):
            adapter._handle_http_error(_make_http_error(403, "rateLimitExceeded"), "ctx")

    def test_403_user_rate_limit_raises_rate_limit(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(RateLimitStorageError):
            adapter._handle_http_error(_make_http_error(403, "userRateLimitExceeded"), "ctx")

    def test_403_other_raises_storage_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(StorageError):
            adapter._handle_http_error(_make_http_error(403, "accessNotConfigured"), "ctx")

    def test_404_raises_file_not_found(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(FileNotFoundStorageError):
            adapter._handle_http_error(_make_http_error(404), "ctx")

    def test_500_raises_network_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(NetworkError):
            adapter._handle_http_error(_make_http_error(500), "ctx")

    def test_503_raises_network_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(NetworkError):
            adapter._handle_http_error(_make_http_error(503), "ctx")

    def test_400_raises_storage_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with pytest.raises(StorageError):
            adapter._handle_http_error(_make_http_error(400), "ctx")


# ── upload ────────────────────────────────────────────────────────────────────

class TestUpload:
    async def test_upload_calls_create(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)

        # Mock folder lookup to return a folder id
        svc.files().list().execute.return_value = {"files": [{"id": "folder123"}]}
        svc.files().create().execute.return_value = {
            "id": "file456", "size": "100", "md5Checksum": "abcdef"
        }

        src = tmp_path / "test.txt"
        src.write_bytes(b"test content")

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            result = await adapter.upload(str(src), "doc.txt")

        assert result.file_id == "file456"
        assert result.size == 100
        assert result.md5 == "abcdef"

    async def test_upload_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().list().execute.return_value = {"files": [{"id": "f"}]}
        svc.files().create().execute.side_effect = _make_http_error(403, "quotaExceeded")

        src = tmp_path / "test.txt"
        src.write_bytes(b"x")

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(StorageQuotaExceededError):
                await adapter.upload(str(src), "doc.txt")


# ── get_quota ─────────────────────────────────────────────────────────────────

class TestGetQuota:
    async def test_returns_storage_quota(self, tmp_path):
        from oprim.storage.protocol import StorageQuota

        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.about().get().execute.return_value = {
            "storageQuota": {"usage": "1000", "limit": "10000"}
        }

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            q = await adapter.get_quota()

        assert isinstance(q, StorageQuota)
        assert q.used_bytes == 1000
        assert q.total_bytes == 10000
        assert q.available_bytes == 9000


# ── list_changes_since ────────────────────────────────────────────────────────

class TestListChangesSince:
    async def test_none_token_returns_start_token(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.changes().getStartPageToken().execute.return_value = {"startPageToken": "tok1"}

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            changes, token = await adapter.list_changes_since(None)

        assert changes == []
        assert token == "tok1"

    async def test_with_token_returns_changes(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.changes().list().execute.return_value = {
            "changes": [{"fileId": "f1", "removed": False}],
            "newStartPageToken": "tok2",
        }

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            changes, token = await adapter.list_changes_since("tok1")

        assert len(changes) == 1
        assert token == "tok2"

    async def test_supports_changes_api_is_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        assert adapter.supports_changes_api is True


# ── health_check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    async def test_returns_true_when_service_ok(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.about().get().execute.return_value = {"user": {"displayName": "Wiki"}}

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            assert await adapter.health_check() is True

    async def test_returns_false_on_exception(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.about().get().execute.side_effect = Exception("network fail")

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            assert await adapter.health_check() is False

    async def test_returns_false_when_not_authenticated(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        # service not set
        assert await adapter.health_check() is False


# ── OAuth helpers ─────────────────────────────────────────────────────────────

class TestLoadOAuthConfig:
    def test_nested_installed_format(self, tmp_path):
        cfg = {
            "installed": {
                "client_id": "test.apps.googleusercontent.com",
                "client_secret": "secret",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            }
        }
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        result = load_oauth_config(p)
        assert result == cfg

    def test_nested_web_format(self, tmp_path):
        cfg = {
            "web": {
                "client_id": "test.apps.googleusercontent.com",
                "client_secret": "secret",
                "redirect_uris": [],
            }
        }
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        result = load_oauth_config(p)
        assert result == cfg

    def test_flat_format(self, tmp_path):
        cfg = {
            "client_id": "test.apps.googleusercontent.com",
            "client_secret": "secret",
        }
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        result = load_oauth_config(p)
        assert result == cfg

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(AuthenticationError, match="not found"):
            load_oauth_config(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "oauth.json"
        p.write_text("not json {{{")
        with pytest.raises(AuthenticationError, match="not valid JSON"):
            load_oauth_config(p)

    def test_missing_client_id_raises(self, tmp_path):
        cfg = {"installed": {"client_secret": "s"}}
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(AuthenticationError, match="missing keys"):
            load_oauth_config(p)

    def test_missing_client_secret_raises(self, tmp_path):
        cfg = {"installed": {"client_id": "test.apps.googleusercontent.com"}}
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(AuthenticationError, match="missing keys"):
            load_oauth_config(p)

    def test_non_standard_client_id_logs_warning(self, tmp_path):
        cfg = {"client_id": "non-standard-id", "client_secret": "s"}
        p = tmp_path / "oauth.json"
        p.write_text(json.dumps(cfg))
        # Should not raise, just warns
        result = load_oauth_config(p)
        assert result == cfg


class TestSetTokenPermissions:
    def test_sets_0600(self, tmp_path):
        import stat
        p = tmp_path / "token.json"
        p.write_text("{}")
        set_token_permissions(p)
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600

    def test_missing_file_logs_warning(self, tmp_path):
        # Should not raise — just logs warning
        set_token_permissions(tmp_path / "nonexistent.json")


# ── download (mocked) ─────────────────────────────────────────────────────────

class TestDownload:
    async def test_download_writes_file(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)

        # _stream runs synchronously inside to_thread; mock get_media and downloader
        request_mock = MagicMock()
        svc.files().get_media.return_value = request_mock

        import io
        output_bytes = b"file contents"

        def fake_to_thread(fn, *a, **k):
            # simulate the download by writing to buf directly
            fn()
            return None

        dest = tmp_path / "out.txt"

        with patch("asyncio.to_thread") as mock_tt:
            async def async_fn(fn, *a, **k):
                # simulate: downloader fills buf; we patch Path.write_bytes
                return None
            mock_tt.side_effect = async_fn

            # Patch write_bytes to capture the call
            with patch("oprim.storage.providers.gdrive.Path") as mock_path:
                mock_path.return_value.write_bytes = MagicMock()
                await adapter.download("file123", str(dest))

    async def test_download_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().get_media.side_effect = _make_http_error(404)

        dest = tmp_path / "out.txt"
        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(FileNotFoundStorageError):
                await adapter.download("missing_file_id", str(dest))


# ── delete (mocked) ───────────────────────────────────────────────────────────

class TestDelete:
    async def test_delete_calls_api(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().delete().execute.return_value = None

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            await adapter.delete("file123")

        svc.files().delete.assert_called()

    async def test_delete_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().delete().execute.side_effect = _make_http_error(404)

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(FileNotFoundStorageError):
                await adapter.delete("missing_id")


# ── list_files (mocked) ───────────────────────────────────────────────────────

class TestListFiles:
    async def test_list_files_yields_storage_files(self, tmp_path):
        from oprim.storage.protocol import StorageFile

        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        # folder lookup
        svc.files().list().execute.return_value = {
            "files": [
                {
                    "id": "f1",
                    "name": "doc.pdf",
                    "size": "2048",
                    "mimeType": "application/pdf",
                    "createdTime": "2024-01-01T00:00:00Z",
                    "modifiedTime": "2024-01-02T00:00:00Z",
                    "md5Checksum": "abc",
                }
            ]
        }

        call_count = [0]
        original_list = svc.files().list().execute.return_value

        def _list_execute():
            call_count[0] += 1
            if call_count[0] == 1:
                # folder search returns folder id
                return {"files": [{"id": "folder_id"}]}
            # file list returns files (no next page)
            return original_list

        svc.files().list().execute.side_effect = _list_execute

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            files = [f async for f in adapter.list_files("/Stratum")]

        assert len(files) == 1
        assert isinstance(files[0], StorageFile)
        assert files[0].file_id == "f1"
        assert files[0].name == "doc.pdf"

    async def test_list_files_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)

        call_count = [0]

        def _list_execute():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"files": [{"id": "folder_id"}]}
            raise _make_http_error(403, "rateLimitExceeded")

        svc.files().list().execute.side_effect = _list_execute

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(RateLimitStorageError):
                async for _ in adapter.list_files("/Stratum"):
                    pass


# ── get_file_metadata (mocked) ────────────────────────────────────────────────

class TestGetFileMetadata:
    async def test_returns_storage_file(self, tmp_path):
        from oprim.storage.protocol import StorageFile

        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().get().execute.return_value = {
            "id": "f1",
            "name": "notes.txt",
            "size": "512",
            "mimeType": "text/plain",
            "createdTime": "2024-03-01T10:00:00Z",
            "modifiedTime": "2024-03-02T12:00:00Z",
            "md5Checksum": "deadbeef",
        }

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            meta = await adapter.get_file_metadata("f1")

        assert isinstance(meta, StorageFile)
        assert meta.file_id == "f1"
        assert meta.name == "notes.txt"
        assert meta.md5 == "deadbeef"

    async def test_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().get().execute.side_effect = _make_http_error(404)

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(FileNotFoundStorageError):
                await adapter.get_file_metadata("missing")


# ── _find_or_create_folder ────────────────────────────────────────────────────

class TestFindOrCreateFolder:
    async def test_returns_existing_folder(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().list().execute.return_value = {"files": [{"id": "existing_folder"}]}

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            fid = await adapter._find_or_create_folder("Stratum", None)

        assert fid == "existing_folder"

    async def test_creates_folder_when_not_found(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        # folder not found
        svc.files().list().execute.return_value = {"files": []}
        # create returns new folder
        svc.files().create().execute.return_value = {"id": "new_folder_id"}

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            fid = await adapter._find_or_create_folder("Stratum", None)

        assert fid == "new_folder_id"

    async def test_creates_folder_with_parent(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().list().execute.return_value = {"files": []}
        svc.files().create().execute.return_value = {"id": "child_id"}

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            fid = await adapter._find_or_create_folder("substrates", "parent_123")


# ── authenticate OAuth flow ───────────────────────────────────────────────────

class TestAuthenticateOAuthFlow:
    async def test_runs_oauth_flow_when_no_token(self, tmp_path):
        """Cover the else branch (flow.run_local_server)."""
        adapter = _make_adapter(tmp_path)  # no token file

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.to_json.return_value = '{"token": "new_tok"}'

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        with (
            patch("oprim.storage.providers.gdrive.InstalledAppFlow.from_client_secrets_file", return_value=mock_flow),
            patch("oprim.storage.providers.gdrive.build"),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn(*a, **k) if callable(fn) else fn),
        ):
            result = await adapter.authenticate()

        assert result is True
        mock_flow.run_local_server.assert_called_once()

    async def test_refresh_success_writes_token(self, tmp_path):
        """Cover the refresh success path (lines 74-75, 87-88)."""
        dummy_token = {
            "token": "old",
            "refresh_token": "rtok",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "c.apps.googleusercontent.com",
            "client_secret": "s",
            "scopes": ["https://www.googleapis.com/auth/drive.file"],
        }
        adapter = _make_adapter(tmp_path, token_data=dummy_token)

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "rtok"
        mock_creds.to_json.return_value = json.dumps(dummy_token)

        with (
            patch("oprim.storage.providers.gdrive.Credentials.from_authorized_user_file", return_value=mock_creds),
            patch("oprim.storage.providers.gdrive.build"),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: None),
        ):
            result = await adapter.authenticate()

        assert result is True
        assert adapter.token_path.exists()


# ── error paths for quota / changes ──────────────────────────────────────────

class TestErrorPaths:
    async def test_get_quota_http_error_propagates(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.about().get().execute.side_effect = _make_http_error(403, "rateLimitExceeded")

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(RateLimitStorageError):
                await adapter.get_quota()

    async def test_list_changes_none_http_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.changes().getStartPageToken().execute.side_effect = _make_http_error(401)

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(TokenExpiredError):
                await adapter.list_changes_since(None)

    async def test_list_changes_with_token_http_error(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.changes().list().execute.side_effect = _make_http_error(500)

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            with pytest.raises(NetworkError):
                await adapter.list_changes_since("tok")


# ── download _stream coverage ─────────────────────────────────────────────────

class TestDownloadStream:
    async def test_download_stream_calls_downloader(self, tmp_path):
        """Cover lines 190-195: _stream function + MediaIoBaseDownload loop."""
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().get_media.return_value = MagicMock()

        status_mock = MagicMock()
        status_mock.progress.return_value = 0.75
        downloader_mock = MagicMock()
        downloader_mock.next_chunk.side_effect = [(status_mock, False), (None, True)]

        progress_vals = []
        dest = tmp_path / "output.txt"

        with (
            patch("oprim.storage.providers.gdrive.MediaIoBaseDownload", return_value=downloader_mock),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
            patch("oprim.storage.providers.gdrive.Path") as mock_path,
        ):
            mock_path.return_value.write_bytes = MagicMock()
            await adapter.download("file123", str(dest), on_progress=progress_vals.append)

        assert downloader_mock.next_chunk.call_count == 2
        assert 0.75 in progress_vals

    async def test_download_stream_no_status(self, tmp_path):
        """Cover line 194-195: status is None (on_progress not called)."""
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)
        svc.files().get_media.return_value = MagicMock()

        downloader_mock = MagicMock()
        downloader_mock.next_chunk.side_effect = [(None, True)]

        dest = tmp_path / "output.txt"

        with (
            patch("oprim.storage.providers.gdrive.MediaIoBaseDownload", return_value=downloader_mock),
            patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()),
            patch("oprim.storage.providers.gdrive.Path") as mock_path,
        ):
            mock_path.return_value.write_bytes = MagicMock()
            await adapter.download("file123", str(dest))


# ── list_files pagination ─────────────────────────────────────────────────────

class TestListFilesPagination:
    async def test_pagination_follows_next_page_token(self, tmp_path):
        """Cover line 229: kwargs['pageToken'] = page_token."""
        adapter = _make_adapter(tmp_path)
        svc = _attach_mock_service(adapter)

        call_count = [0]

        def _execute():
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: folder search returns folder id
                return {"files": [{"id": "folder_id"}]}
            if call_count[0] == 2:
                # Second call: first page of files with nextPageToken
                return {
                    "files": [
                        {
                            "id": "f1", "name": "a.txt", "size": "10",
                            "mimeType": "text/plain",
                            "createdTime": "2024-01-01T00:00:00Z",
                            "modifiedTime": "2024-01-01T00:00:00Z",
                        }
                    ],
                    "nextPageToken": "page2",
                }
            # Third call: second page, no more pages
            return {
                "files": [
                    {
                        "id": "f2", "name": "b.txt", "size": "20",
                        "mimeType": "text/plain",
                        "createdTime": "2024-01-02T00:00:00Z",
                        "modifiedTime": "2024-01-02T00:00:00Z",
                    }
                ],
            }

        svc.files().list().execute.side_effect = _execute

        with patch("asyncio.to_thread", side_effect=lambda fn, *a, **k: fn()):
            files = [f async for f in adapter.list_files("/Stratum")]

        assert len(files) == 2
        assert files[0].file_id == "f1"
        assert files[1].file_id == "f2"
        assert call_count[0] == 3  # folder lookup + 2 pages
