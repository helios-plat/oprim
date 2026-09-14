"""Compatibility re-export for the canonical obase Fernet substrate."""

from obase._fernet_vault import (
    fernet_decrypt_dict,
    fernet_dump,
    fernet_encrypt_dict,
    fernet_load,
    fernet_load_or_create_key,
)

__all__ = [
    "fernet_decrypt_dict",
    "fernet_dump",
    "fernet_encrypt_dict",
    "fernet_load",
    "fernet_load_or_create_key",
]
