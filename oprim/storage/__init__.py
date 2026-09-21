"""oprim.storage — storage adapter protocol surface (Phase 2 Wave 1).

本包仅提供 storage 的原子 protocol/类型契约:
- StorageAdapter Protocol
- StorageFile / StorageQuota / UploadResult 数据契约
- storage 错误层次

重型 provider/backend 实现 (GoogleDrive OAuth / LocalFS 等) 不在本包内 —
按 3O SPEC 归属 obase; oprim 只持有跨 provider 的原子契约, 零业务状态。
"""

from __future__ import annotations

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

__all__ = [
    # protocol + data contracts
    "StorageAdapter",
    "StorageFile",
    "StorageQuota",
    "UploadResult",
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
