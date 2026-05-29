"""Contract tests for oprim.crypto.hashing.sha256_hash (Bug backfill §Step 4).

These tests pin the observable API contract of sha256_hash so that any
regression in type, length, determinism, or known-value behaviour is caught
immediately.
"""
from __future__ import annotations

import pytest

from oprim.crypto.hashing import sha256_hash


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_sha256_returns_str() -> None:
    """sha256_hash must always return a str."""
    result = sha256_hash(b"hello")
    assert isinstance(result, str), f"Expected str, got {type(result).__name__}"


# ---------------------------------------------------------------------------
# 2. Output length
# ---------------------------------------------------------------------------


def test_sha256_length_64() -> None:
    """SHA-256 hex digest must be exactly 64 characters."""
    result = sha256_hash(b"hello")
    assert len(result) == 64, f"Expected length 64, got {len(result)}"


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------


def test_sha256_deterministic() -> None:
    """Same input must always produce the same output."""
    data = b"deterministic test input"
    assert sha256_hash(data) == sha256_hash(data), "sha256_hash is not deterministic"


# ---------------------------------------------------------------------------
# 4. Bytes input
# ---------------------------------------------------------------------------


def test_sha256_bytes_input() -> None:
    """sha256_hash must accept bytes and return a valid 64-char hex string."""
    result = sha256_hash(b"\x00\xff\xab\xcd")
    assert isinstance(result, str)
    assert len(result) == 64
    # Ensure only lowercase hex characters
    assert all(c in "0123456789abcdef" for c in result), "Not a valid lowercase hex string"


# ---------------------------------------------------------------------------
# 5. String input
# ---------------------------------------------------------------------------


def test_sha256_str_input() -> None:
    """sha256_hash must accept str (UTF-8 encoded) and return a valid 64-char hex string."""
    result = sha256_hash("hello world")
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result), "Not a valid lowercase hex string"


# ---------------------------------------------------------------------------
# 6. Known NIST test vector
# ---------------------------------------------------------------------------


def test_sha256_known_value_empty_string() -> None:
    """sha256('') == NIST FIPS 180-4 reference value."""
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_hash("") == expected, (
        f"sha256('') mismatch.\n  got: {sha256_hash('')}\n  expected: {expected}"
    )


def test_sha256_known_value_hello() -> None:
    """sha256('hello') == well-known reference value."""
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert sha256_hash("hello") == expected, (
        f"sha256('hello') mismatch.\n  got: {sha256_hash('hello')}\n  expected: {expected}"
    )


# ---------------------------------------------------------------------------
# 7. Different inputs produce different hashes
# ---------------------------------------------------------------------------


def test_sha256_different_inputs_differ() -> None:
    """Two distinct inputs must produce distinct hashes (collision resistance spot-check)."""
    h1 = sha256_hash(b"input_alpha")
    h2 = sha256_hash(b"input_beta")
    assert h1 != h2, "Distinct inputs must not produce the same hash"


# ---------------------------------------------------------------------------
# 8. TypeError for invalid input
# ---------------------------------------------------------------------------


def test_sha256_raises_type_error_on_int() -> None:
    """sha256_hash must raise TypeError for non-bytes/str input."""
    with pytest.raises(TypeError):
        sha256_hash(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. str vs bytes equivalence (UTF-8 round-trip)
# ---------------------------------------------------------------------------


def test_sha256_str_bytes_utf8_equivalence() -> None:
    """sha256_hash('abc') must equal sha256_hash(b'abc') since str is UTF-8 encoded."""
    assert sha256_hash("abc") == sha256_hash(b"abc"), (
        "sha256_hash str and bytes inputs must agree for ASCII strings"
    )
