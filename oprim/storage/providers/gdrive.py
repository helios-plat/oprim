"""Google Drive storage provider."""
from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from oprim._logging import log
from oprim.storage._oauth import load_oauth_config, set_token_permissions
from oprim.storage.errors import (
    AuthenticationError,
    FileNotFoundStorageError,
    NetworkError,
    RateLimitStorageError,
    StorageError,
    StorageQuotaExceededError,
    TokenExpiredError,
)
from oprim.storage.protocol import StorageFile, StorageQuota, UploadResult

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_ISO_FMT_SHORT = "%Y-%m-%dT%H:%M:%SZ"


def _parse_gdrive_datetime(s: str) -> datetime:
    for fmt in (_ISO_FMT, _ISO_FMT_SHORT):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse GDrive datetime: {s!r}")


class GoogleDriveAdapter:
    """Google Drive storage adapter (OAuth 2.0, drive.file scope)."""

    name = "gdrive"
    supports_changes_api = True
    max_file_size = 5 * 1024**4  # 5 TB

    def __init__(
        self,
        credentials_dir: Path = Path.home() / ".stratum" / "credentials",
        oauth_config_path: Path = Path.home() / ".stratum" / "secrets" / "gdrive_oauth.json",
    ) -> None:
        self.credentials_dir = Path(credentials_dir)
        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        self.oauth_config_path = Path(oauth_config_path)
        self.token_path = self.credentials_dir / "gdrive_token.json"
        self._service = None
        self._creds: Credentials | None = None

    # ── Auth ──────────────────────────────────────────────────────────────

    async def authenticate(self) -> bool:
        """Run OAuth flow on first use; auto-refresh on subsequent calls."""
        if self.token_path.exists():
            self._creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not self._creds or not self._creds.valid:
            if self._creds and self._creds.expired and self._creds.refresh_token:
                try:
                    await asyncio.to_thread(self._creds.refresh, Request())
                    log.info("gdrive_token_refreshed")
                except Exception as exc:
                    log.error("gdrive_token_refresh_failed", error=str(exc))
                    raise TokenExpiredError(f"Token refresh failed: {exc}") from exc
            else:
                load_oauth_config(self.oauth_config_path)  # validate before opening browser
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.oauth_config_path), SCOPES
                )
                self._creds = await asyncio.to_thread(flow.run_local_server, port=0)
                log.info("gdrive_oauth_completed")

            self.token_path.write_text(self._creds.to_json())
            set_token_permissions(self.token_path)

        self._service = build("drive", "v3", credentials=self._creds, cache_discovery=False)
        log.info("gdrive_authenticated")
        return True

    def _ensure_authenticated(self) -> None:
        if self._service is None:
            raise AuthenticationError("Not authenticated — call authenticate() first.")

    # ── Folder helpers ────────────────────────────────────────────────────

    async def _find_or_create_folder(self, name: str, parent_id: str | None) -> str:
        """Return the folder id for *name* under *parent_id*, creating it if absent."""
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"

        def _search():
            return (
                self._service.files()
                .list(q=q, fields="files(id)", pageSize=1)
                .execute()
            )

        resp = await asyncio.to_thread(_search)
        files = resp.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        def _create():
            return self._service.files().create(body=metadata, fields="id").execute()

        folder = await asyncio.to_thread(_create)
        log.info("gdrive_folder_created", name=name, parent=parent_id)
        return folder["id"]

    async def _ensure_folder(self, folder_path: str) -> str:
        """Ensure a nested folder path exists; return the leaf folder id."""
        parts = [p for p in folder_path.strip("/").split("/") if p]
        parent_id: str | None = None
        for part in parts:
            parent_id = await self._find_or_create_folder(part, parent_id)
        return parent_id  # type: ignore[return-value]

    # ── CRUD ──────────────────────────────────────────────────────────────

    async def upload(
        self,
        local_path: str,
        remote_path: str,
        mime_type: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> UploadResult:
        self._ensure_authenticated()
        parts = remote_path.lstrip("/").split("/")
        file_name = parts[-1]
        folder_parts = parts[:-1]
        folder_path = "/Stratum/" + "/".join(folder_parts) if folder_parts else "/Stratum"
        folder_id = await self._ensure_folder(folder_path)

        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
        metadata = {"name": file_name, "parents": [folder_id]}

        def _execute():
            return (
                self._service.files()
                .create(body=metadata, media_body=media, fields="id,size,md5Checksum")
                .execute()
            )

        try:
            file = await asyncio.to_thread(_execute)
        except HttpError as exc:
            self._handle_http_error(exc, context=f"upload {remote_path}")

        log.info("gdrive_upload_ok", remote=remote_path, file_id=file["id"])
        return UploadResult(
            file_id=file["id"],
            size=int(file.get("size", 0)),
            md5=file.get("md5Checksum", ""),
        )

    async def download(
        self,
        file_id: str,
        local_path: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self._ensure_authenticated()
        try:
            request = self._service.files().get_media(fileId=file_id)
            buf = io.BytesIO()

            def _stream():
                downloader = MediaIoBaseDownload(buf, request, chunksize=10 * 1024 * 1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if on_progress and status:
                        on_progress(status.progress())

            await asyncio.to_thread(_stream)
            Path(local_path).write_bytes(buf.getvalue())
        except HttpError as exc:
            self._handle_http_error(exc, context=f"download {file_id}")

        log.info("gdrive_download_ok", file_id=file_id, local=local_path)

    async def delete(self, file_id: str) -> None:
        self._ensure_authenticated()
        try:
            await asyncio.to_thread(
                self._service.files().delete(fileId=file_id).execute
            )
        except HttpError as exc:
            self._handle_http_error(exc, context=f"delete {file_id}")
        log.info("gdrive_delete_ok", file_id=file_id)

    async def list_files(
        self,
        folder: str = "/Stratum",
        recursive: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[StorageFile]:
        self._ensure_authenticated()
        folder_id = await self._ensure_folder(folder)
        page_token: str | None = None
        _fields = "nextPageToken, files(id,name,size,mimeType,createdTime,modifiedTime,md5Checksum)"

        while True:
            q = f"'{folder_id}' in parents and trashed=false"
            kwargs: dict = dict(q=q, pageSize=page_size, fields=_fields)
            if page_token:
                kwargs["pageToken"] = page_token

            def _list(kw=kwargs):
                return self._service.files().list(**kw).execute()

            try:
                resp = await asyncio.to_thread(_list)
            except HttpError as exc:
                self._handle_http_error(exc, context=f"list_files {folder}")

            for f in resp.get("files", []):
                yield StorageFile(
                    file_id=f["id"],
                    name=f["name"],
                    size=int(f.get("size", 0)),
                    mime_type=f.get("mimeType", "application/octet-stream"),
                    created_at=_parse_gdrive_datetime(f["createdTime"]),
                    modified_at=_parse_gdrive_datetime(f["modifiedTime"]),
                    md5=f.get("md5Checksum"),
                    metadata={},
                )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    async def get_file_metadata(self, file_id: str) -> StorageFile:
        self._ensure_authenticated()
        _fields = "id,name,size,mimeType,createdTime,modifiedTime,md5Checksum"

        def _get():
            return self._service.files().get(fileId=file_id, fields=_fields).execute()

        try:
            f = await asyncio.to_thread(_get)
        except HttpError as exc:
            self._handle_http_error(exc, context=f"get_file_metadata {file_id}")

        return StorageFile(
            file_id=f["id"],
            name=f["name"],
            size=int(f.get("size", 0)),
            mime_type=f.get("mimeType", "application/octet-stream"),
            created_at=_parse_gdrive_datetime(f["createdTime"]),
            modified_at=_parse_gdrive_datetime(f["modifiedTime"]),
            md5=f.get("md5Checksum"),
            metadata={},
        )

    async def get_quota(self) -> StorageQuota:
        self._ensure_authenticated()

        def _about():
            return self._service.about().get(fields="storageQuota").execute()

        try:
            about = await asyncio.to_thread(_about)
        except HttpError as exc:
            self._handle_http_error(exc, context="get_quota")

        q = about["storageQuota"]
        used = int(q.get("usage", 0))
        total = int(q.get("limit", 0))
        return StorageQuota(used_bytes=used, total_bytes=total, available_bytes=total - used)

    async def list_changes_since(
        self, page_token: str | None
    ) -> tuple[list[dict], str]:
        self._ensure_authenticated()

        if page_token is None:
            def _start():
                return self._service.changes().getStartPageToken().execute()

            try:
                resp = await asyncio.to_thread(_start)
            except HttpError as exc:
                self._handle_http_error(exc, context="list_changes_since (start token)")
            return ([], resp["startPageToken"])

        def _list():
            return (
                self._service.changes()
                .list(
                    pageToken=page_token,
                    fields=(
                        "nextPageToken, newStartPageToken, "
                        "changes(fileId, removed, file(id,name,modifiedTime,md5Checksum))"
                    ),
                )
                .execute()
            )

        try:
            resp = await asyncio.to_thread(_list)
        except HttpError as exc:
            self._handle_http_error(exc, context="list_changes_since")

        changes = resp.get("changes", [])
        next_token = resp.get("nextPageToken") or resp.get("newStartPageToken", "")
        log.info("gdrive_changes_fetched", count=len(changes), next_token=next_token[:20])
        return (changes, next_token)

    async def health_check(self) -> bool:
        try:
            self._ensure_authenticated()

            def _check():
                return self._service.about().get(fields="user").execute()

            await asyncio.to_thread(_check)
            return True
        except Exception as exc:
            log.warning("gdrive_health_check_failed", error=str(exc))
            return False

    # ── Error mapping ─────────────────────────────────────────────────────

    def _handle_http_error(self, exc: HttpError, context: str) -> None:
        status = exc.resp.status
        log.error("gdrive_http_error", status=status, context=context)
        if status == 401:
            raise TokenExpiredError(f"Auth failed at {context}") from exc
        if status == 403:
            reason = ""
            if exc.error_details:
                reason = exc.error_details[0].get("reason", "")
            if "quotaExceeded" in reason or "storageQuotaExceeded" in reason:
                raise StorageQuotaExceededError(f"Quota exceeded at {context}") from exc
            if "rateLimitExceeded" in reason or "userRateLimitExceeded" in reason:
                raise RateLimitStorageError(f"Rate limit at {context}") from exc
            raise StorageError(f"Forbidden at {context}: {reason}") from exc
        if status == 404:
            raise FileNotFoundStorageError(f"Not found at {context}") from exc
        if status >= 500:
            raise NetworkError(f"Server error {status} at {context}") from exc
        raise StorageError(f"HTTP {status} at {context}: {exc}") from exc
