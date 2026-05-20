"""Shared OAuth helper for GDrive provider."""
from __future__ import annotations

import json
import os
from pathlib import Path

from oprim._logging import log
from oprim.storage.errors import AuthenticationError


def load_oauth_config(path: Path) -> dict:
    """Load and validate OAuth client config from *path*.

    Supports both the flat format and the nested ``installed``/``web`` format
    that the Google Cloud Console produces.
    """
    if not path.exists():
        raise AuthenticationError(f"OAuth config not found: {path}")

    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AuthenticationError(f"OAuth config is not valid JSON: {path}") from exc

    inner = cfg.get("installed") or cfg.get("web") or cfg
    required = ("client_id", "client_secret")
    missing = [k for k in required if not inner.get(k)]
    if missing:
        raise AuthenticationError(f"OAuth config missing keys: {missing} in {path}")

    client_id = inner["client_id"]
    if not client_id.endswith(".apps.googleusercontent.com"):
        log.warning(
            "gdrive_oauth_client_id_format",
            warning="client_id does not end with .apps.googleusercontent.com",
            client_id_prefix=client_id[:30],
        )

    return cfg


def set_token_permissions(token_path: Path) -> None:
    """Restrict token file to owner read/write only (0o600)."""
    try:
        os.chmod(token_path, 0o600)
    except OSError as exc:
        log.warning("gdrive_token_chmod_failed", path=str(token_path), error=str(exc))
