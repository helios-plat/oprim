"""Storage-domain error hierarchy."""
from __future__ import annotations

from oprim.errors import StratumError


class StorageError(StratumError):
    """Base class for all storage errors."""


class AuthenticationError(StorageError):
    """OAuth authentication failed or credentials not configured."""


class StorageQuotaExceededError(StorageError):
    """Remote storage quota exceeded."""


class FileNotFoundStorageError(StorageError):
    """File not found in remote/local storage."""


class RateLimitStorageError(StorageError):
    """Provider rate limit hit — caller should back off and retry."""


class NetworkError(StorageError):
    """Transient network / server error."""


class ConflictError(StorageError):
    """File conflict — already exists or md5 mismatch."""


class TokenExpiredError(AuthenticationError):
    """OAuth token expired and refresh failed."""
