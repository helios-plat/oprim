"""Tests for oprim.crypto.sha256_hash."""

import pytest

from oprim.crypto.hashing import sha256_hash


def test_sha256_empty_string():
    assert sha256_hash("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_abc():
    assert sha256_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_bytes_input():
    result = sha256_hash(b"abc")
    assert result == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_str_input_utf8_encoded():
    assert sha256_hash("abc") == sha256_hash(b"abc")


def test_sha256_invalid_type_raises():
    with pytest.raises(TypeError):
        sha256_hash(123)
    with pytest.raises(TypeError):
        sha256_hash(None)
    with pytest.raises(TypeError):
        sha256_hash([1, 2, 3])


def test_sha256_returns_64_char_hex():
    result = sha256_hash("hello")
    assert len(result) == 64
    assert result == result.lower()
    assert all(c in "0123456789abcdef" for c in result)


def test_sha256_deterministic():
    assert sha256_hash("test") == sha256_hash("test")


def test_sha256_different_inputs_differ():
    assert sha256_hash("a") != sha256_hash("b")


def test_sha256_unicode_input():
    result = sha256_hash("日本語")
    assert len(result) == 64


@pytest.mark.academic_reference
def test_sha256_nist_fips_180_4_test_vectors():
    """NIST FIPS 180-4 Appendix B official test vectors."""
    # B.1: empty
    assert sha256_hash(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    # B.1: "abc"
    assert sha256_hash(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    # B.1: "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    assert sha256_hash(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") == \
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    # B.3: 1 million 'a'
    assert sha256_hash(b"a" * 1_000_000) == \
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
