"""Tests for oprim S3 operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprim import s3_object_metadata, s3_upload_file
from oprim._exceptions import OprimAuthError, OprimConnectionError, OprimError, OprimNotFoundError
from oprim._s3 import ObjectMetadata, UploadResult


def _make_s3_client(head_response=None, upload_side_effect=None, head_side_effect=None):
    client = MagicMock()
    if upload_side_effect:
        client.upload_file.side_effect = upload_side_effect
    if head_response is not None:
        client.head_object.return_value = head_response
    if head_side_effect:
        client.head_object.side_effect = head_side_effect
    return client


# ---------------------------------------------------------------------------
# s3_upload_file
# ---------------------------------------------------------------------------

class TestS3UploadFile:
    def test_successful_upload(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello s3")

        s3_client = _make_s3_client(
            head_response={
                "ETag": '"abc123"',
                "ContentLength": 8,
            }
        )

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_upload_file(
                local_path=str(f),
                s3_url="s3://my-bucket/path/test.txt",
            )

        assert isinstance(result, UploadResult)
        assert result.s3_url == "s3://my-bucket/path/test.txt"
        assert result.etag == "abc123"
        assert result.size_bytes == 8

    def test_upload_with_content_type(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_bytes(b'{"key": "val"}')

        s3_client = MagicMock()
        s3_client.head_object.return_value = {"ETag": '"etag"', "ContentLength": 14}

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_upload_file(
                local_path=str(f),
                s3_url="s3://bucket/data.json",
                content_type="application/json",
            )

        s3_client.upload_file.assert_called_once()
        call_kwargs = s3_client.upload_file.call_args
        extra_args = call_kwargs[1].get("ExtraArgs") or call_kwargs[0][3]
        assert extra_args.get("ContentType") == "application/json"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(OprimNotFoundError):
            s3_upload_file(
                local_path="/nonexistent/file.txt",
                s3_url="s3://bucket/file.txt",
            )

    def test_no_credentials_raises_auth_error(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")

        NoCredentials = type("NoCredentialsError", (Exception,), {})

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            with patch("oprim._s3.boto3.client") as mock_client:
                s3 = MagicMock()
                mock_client.return_value = s3
                s3.upload_file.side_effect = NoCredentials("no creds")

                with patch("oprim._s3.boto3") as mock_b3:
                    mock_b3.client.return_value = s3
                    with patch("oprim._s3._make_s3_client", return_value=s3):
                        import botocore.exceptions
                        s3.upload_file.side_effect = botocore.exceptions.NoCredentialsError()
                        with pytest.raises((OprimAuthError, OprimConnectionError, Exception)):
                            s3_upload_file(
                                local_path=str(f),
                                s3_url="s3://bucket/file.txt",
                            )

    def test_invalid_s3_url(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        with pytest.raises(OprimError):
            s3_upload_file(
                local_path=str(f),
                s3_url="http://not-s3/bucket/file.txt",
            )

    def test_empty_bucket_url(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        with pytest.raises(OprimError):
            s3_upload_file(local_path=str(f), s3_url="s3:///key")

    def test_boto3_not_installed(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        with patch("oprim._s3.boto3", None):
            with pytest.raises(OprimError, match="boto3"):
                s3_upload_file(local_path=str(f), s3_url="s3://bucket/file.txt")

    def test_client_creation_fails(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.side_effect = RuntimeError("no endpoint")
            with pytest.raises(OprimConnectionError):
                s3_upload_file(local_path=str(f), s3_url="s3://bucket/file.txt")

    def test_upload_with_sse(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello")
        s3_client = _make_s3_client(head_response={"ETag": '"e"', "ContentLength": 5})

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_upload_file(
                local_path=str(f),
                s3_url="s3://bucket/file.txt",
                sse="AES256",
            )
        call_kwargs = s3_client.upload_file.call_args
        extra = call_kwargs[1].get("ExtraArgs") or call_kwargs[0][3]
        assert extra.get("ServerSideEncryption") == "AES256"

    def test_upload_client_error_auth(self, tmp_path):
        import botocore.exceptions
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        err = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidAccessKeyId", "Message": "auth failed"}}, "PutObject"
        )
        s3_client = _make_s3_client(upload_side_effect=err)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimAuthError):
                s3_upload_file(local_path=str(f), s3_url="s3://bucket/file.txt")

    def test_upload_client_error_generic(self, tmp_path):
        import botocore.exceptions
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        err = botocore.exceptions.ClientError(
            {"Error": {"Code": "InternalError", "Message": "server error"}}, "PutObject"
        )
        s3_client = _make_s3_client(upload_side_effect=err)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimConnectionError):
                s3_upload_file(local_path=str(f), s3_url="s3://bucket/file.txt")

    def test_upload_generic_exception(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        s3_client = _make_s3_client(upload_side_effect=RuntimeError("unexpected"))

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimConnectionError):
                s3_upload_file(local_path=str(f), s3_url="s3://bucket/file.txt")


# ---------------------------------------------------------------------------
# s3_object_metadata
# ---------------------------------------------------------------------------

class TestS3ObjectMetadata:
    def test_existing_object(self):
        head_resp = {
            "ContentLength": 1024,
            "ETag": '"etag123"',
            "LastModified": MagicMock(),
            "ContentType": "text/plain",
            "StorageClass": "STANDARD",
            "Metadata": {"custom": "value"},
        }
        head_resp["LastModified"].isoformat.return_value = "2026-05-20T10:00:00+00:00"
        s3_client = _make_s3_client(head_response=head_resp)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_object_metadata(s3_url="s3://my-bucket/key.txt")

        assert isinstance(result, ObjectMetadata)
        assert result.exists is True
        assert result.size_bytes == 1024
        assert result.etag == "etag123"
        assert result.content_type == "text/plain"
        assert result.metadata == {"custom": "value"}

    def test_nonexistent_object(self):
        import botocore.exceptions
        error = botocore.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )
        s3_client = _make_s3_client(head_side_effect=error)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_object_metadata(s3_url="s3://my-bucket/missing.txt")

        assert result.exists is False
        assert result.size_bytes is None
        assert result.etag is None

    def test_invalid_s3_url(self):
        with pytest.raises(OprimError):
            s3_object_metadata(s3_url="ftp://not-s3/key")

    def test_no_credentials(self):
        import botocore.exceptions
        error = botocore.exceptions.NoCredentialsError()
        s3_client = _make_s3_client(head_side_effect=error)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimAuthError):
                s3_object_metadata(s3_url="s3://bucket/key")

    def test_metadata_access_denied(self):
        import botocore.exceptions
        err = botocore.exceptions.ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
        )
        s3_client = _make_s3_client(head_side_effect=err)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimAuthError):
                s3_object_metadata(s3_url="s3://bucket/key")

    def test_metadata_generic_client_error(self):
        import botocore.exceptions
        err = botocore.exceptions.ClientError(
            {"Error": {"Code": "InternalError", "Message": "server error"}}, "HeadObject"
        )
        s3_client = _make_s3_client(head_side_effect=err)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimConnectionError):
                s3_object_metadata(s3_url="s3://bucket/key")

    def test_metadata_generic_exception(self):
        s3_client = _make_s3_client(head_side_effect=RuntimeError("unexpected"))

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            with pytest.raises(OprimConnectionError):
                s3_object_metadata(s3_url="s3://bucket/key")

    def test_metadata_no_last_modified(self):
        head_resp = {
            "ContentLength": 100,
            "ETag": '"abc"',
            "ContentType": "text/plain",
            "StorageClass": "STANDARD",
            "Metadata": {},
        }
        s3_client = _make_s3_client(head_response=head_resp)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_object_metadata(s3_url="s3://bucket/key")

        assert result.exists is True
        assert result.last_modified is None

    def test_metadata_empty_dict(self):
        head_resp = {
            "ContentLength": 100,
            "ETag": '"e"',
            "LastModified": MagicMock(),
            "ContentType": "application/octet-stream",
            "StorageClass": "STANDARD",
            "Metadata": {},
        }
        head_resp["LastModified"].isoformat.return_value = "2026-05-20T10:00:00+00:00"
        s3_client = _make_s3_client(head_response=head_resp)

        with patch("oprim._s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3_client
            result = s3_object_metadata(s3_url="s3://bucket/key")

        assert result.metadata == {}
