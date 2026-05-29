"""Tests for oprim.crypto.hmac_sha256."""

import pytest

from oprim.crypto.hashing import hmac_sha256


def test_hmac_sha256_empty_key_empty_data():
    result = hmac_sha256(b"", b"")
    assert len(result) == 64
    assert result == result.lower()


def test_hmac_sha256_str_data():
    result_str = hmac_sha256(b"key", "hello")
    result_bytes = hmac_sha256(b"key", b"hello")
    assert result_str == result_bytes


def test_hmac_sha256_bytes_data():
    result = hmac_sha256(b"secret", b"message")
    assert len(result) == 64


def test_hmac_sha256_str_key_raises_typeerror():
    with pytest.raises(TypeError):
        hmac_sha256("not bytes", b"data")


def test_hmac_sha256_invalid_key_type():
    with pytest.raises(TypeError):
        hmac_sha256(None, b"data")
    with pytest.raises(TypeError):
        hmac_sha256(123, b"data")


def test_hmac_sha256_invalid_data_type():
    with pytest.raises(TypeError):
        hmac_sha256(b"key", 42)
    with pytest.raises(TypeError):
        hmac_sha256(b"key", None)


def test_hmac_sha256_deterministic():
    assert hmac_sha256(b"k", b"d") == hmac_sha256(b"k", b"d")


def test_hmac_sha256_different_keys_differ():
    assert hmac_sha256(b"key1", b"data") != hmac_sha256(b"key2", b"data")


@pytest.mark.academic_reference
def test_hmac_sha256_rfc4231_test_vectors():
    """RFC 4231 Section 4 official test vectors for HMAC-SHA-256."""
    # Test Case 1: key=20 bytes 0x0b, data="Hi There"
    key1 = bytes.fromhex("0b" * 20)
    assert hmac_sha256(key1, b"Hi There") == \
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"

    # Test Case 2: key="Jefe", data="what do ya want for nothing?"
    assert hmac_sha256(b"Jefe", b"what do ya want for nothing?") == \
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"

    # Test Case 4: key=25 bytes, data=50 bytes 0xcd
    key4 = bytes.fromhex("0102030405060708090a0b0c0d0e0f10111213141516171819")
    data4 = bytes.fromhex("cd" * 50)
    assert hmac_sha256(key4, data4) == \
        "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"
