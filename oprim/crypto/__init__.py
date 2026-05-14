"""Cryptographic primitives submodule."""

from oprim.crypto.ed25519 import ed25519_keypair_generate, ed25519_sign, ed25519_verify
from oprim.crypto.hashing import hmac_sha256, sha256_hash
from oprim.crypto.merkle import rfc6962_inclusion_proof, rfc6962_merkle_root

__all__ = [
    "sha256_hash",
    "hmac_sha256",
    "rfc6962_merkle_root",
    "rfc6962_inclusion_proof",
    "ed25519_keypair_generate",
    "ed25519_sign",
    "ed25519_verify",
]
