"""oprim.storage — pluggable storage adapters for Stratum (Phase 2)."""
from oprim.storage.errors import (
    AuthenticationError,
    ConflictError,
    FileNotFoundStorageError,
    NetworkError,
    RateLimitStorageError,
    StorageError,
    StorageQuotaExceededError,
    TokenExpiredError,
)
from oprim.storage.protocol import StorageAdapter, StorageFile, StorageQuota, UploadResult
from oprim.storage.providers.gdrive import GoogleDriveAdapter
from oprim.storage.providers.local import LocalStorageAdapter

__all__ = [
    # protocol
    "StorageAdapter",
    "StorageFile",
    "StorageQuota",
    "UploadResult",
    # providers
    "GoogleDriveAdapter",
    "LocalStorageAdapter",
    # errors
    "StorageError",
    "AuthenticationError",
    "StorageQuotaExceededError",
    "FileNotFoundStorageError",
    "RateLimitStorageError",
    "NetworkError",
    "ConflictError",
    "TokenExpiredError",
]
