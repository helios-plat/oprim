"""oprim.fernet_vault — Fernet encryption primitives for the zero-trust secrets vault.

3O layer: oprim (single atomic crypto computation, pure logic, no I/O policy —
the CALLER decides where keys/secrets live).

Consumed by oskill.zero_trust_vault: real secrets are encrypted at rest with
Fernet (symmetric AES-128-CBC + HMAC); the LLM never touches plaintext.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


def fernet_load_or_create_key(key_path: str | Path, env_key: str | None = None) -> Fernet:
    """Load a Fernet key (env override first); generate + persist if missing.

    Args:
        key_path: Where the key file lives (caller ensures parent dir exists).
        env_key: Optional key string from environment (highest priority).

    Returns:
        A ready-to-use Fernet instance. Key file written with 0o600 perms.
    """
    if env_key:
        return Fernet(env_key.encode())
    path = Path(key_path)
    if path.exists():
        return Fernet(path.read_text(encoding="utf-8").strip().encode())
    key = Fernet.generate_key()
    path.write_text(key.decode(), encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)  # non-POSIX platform — best effort
    return Fernet(key)


def fernet_encrypt_dict(fernet: Fernet, secrets: dict[str, str]) -> dict[str, str]:
    """Encrypt a {vault_id: plaintext} dict for JSON persistence."""
    return {k: fernet.encrypt(v.encode()).decode() for k, v in secrets.items()}


def fernet_decrypt_dict(fernet: Fernet, encrypted: dict[str, str]) -> dict[str, str]:
    """Decrypt a persisted {vault_id: ciphertext} dict back to plaintext.

    Raises: cryptography.fernet.InvalidToken on tampering / wrong key.
    """
    return {k: fernet.decrypt(v.encode()).decode() for k, v in encrypted.items()}


def fernet_dump(fernet: Fernet, secrets: dict[str, str], path: str | Path) -> None:
    """Encrypt + atomically persist secrets to a JSON file (tmp + rename)."""
    payload = fernet_encrypt_dict(fernet, secrets)
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, Path(path))
    with contextlib.suppress(OSError):
        os.chmod(Path(path), 0o600)


def fernet_load(fernet: Fernet, path: str | Path) -> dict[str, str]:
    """Load + decrypt a persisted secrets JSON file. Empty dict if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        encrypted: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return fernet_decrypt_dict(fernet, encrypted)
    except (json.JSONDecodeError, OSError):
        return {}
