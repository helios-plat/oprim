"""Tests for oprim.crypto.merkle (RFC 6962 — arbitrary-bytes leaves)."""

import hashlib

import pytest

from oprim.crypto.merkle import (
    _verify_inclusion,
    rfc6962_inclusion_proof,
    rfc6962_merkle_root,
)


# ---------------------------------------------------------------------------
# rfc6962_merkle_root
# ---------------------------------------------------------------------------


def test_merkle_root_empty_list():
    assert rfc6962_merkle_root([]) == hashlib.sha256(b"").digest()


def test_merkle_root_single_leaf():
    leaf = b"hello"
    expected = hashlib.sha256(b"\x00" + leaf).digest()
    assert rfc6962_merkle_root([leaf]) == expected


def test_merkle_root_two_leaves():
    leaf0, leaf1 = b"a", b"b"
    h0 = hashlib.sha256(b"\x00" + leaf0).digest()
    h1 = hashlib.sha256(b"\x00" + leaf1).digest()
    expected = hashlib.sha256(b"\x01" + h0 + h1).digest()
    assert rfc6962_merkle_root([leaf0, leaf1]) == expected


def test_merkle_root_three_leaves():
    # k=2: left = MTH([0,1]), right = MTH([2])
    leaves = [b"x", b"y", b"z"]
    h0 = hashlib.sha256(b"\x00" + b"x").digest()
    h1 = hashlib.sha256(b"\x00" + b"y").digest()
    h2 = hashlib.sha256(b"\x00" + b"z").digest()
    left = hashlib.sha256(b"\x01" + h0 + h1).digest()
    expected = hashlib.sha256(b"\x01" + left + h2).digest()
    assert rfc6962_merkle_root(leaves) == expected


def test_merkle_root_seven_leaves():
    leaves = [bytes([i]) for i in range(7)]
    root = rfc6962_merkle_root(leaves)
    assert len(root) == 32


def test_merkle_root_eight_leaves():
    leaves = [bytes([i]) for i in range(8)]
    root = rfc6962_merkle_root(leaves)
    assert len(root) == 32


def test_merkle_root_non_list_raises():
    with pytest.raises(TypeError):
        rfc6962_merkle_root((b"x",))


def test_merkle_root_non_bytes_leaf_raises():
    with pytest.raises(TypeError):
        rfc6962_merkle_root(["not bytes"])


# ---------------------------------------------------------------------------
# rfc6962_inclusion_proof
# ---------------------------------------------------------------------------


def test_inclusion_proof_single_leaf():
    proof = rfc6962_inclusion_proof([b"only"], 0)
    assert proof == []


def test_inclusion_proof_two_leaves():
    leaves = [b"a", b"b"]
    proof0 = rfc6962_inclusion_proof(leaves, 0)
    assert len(proof0) == 1
    expected_sibling = hashlib.sha256(b"\x00" + b"b").digest()
    assert proof0[0] == expected_sibling


def test_inclusion_proof_eight_leaves():
    leaves = [bytes([i]) for i in range(8)]
    proof = rfc6962_inclusion_proof(leaves, 0)
    assert len(proof) == 3


def test_inclusion_proof_invalid_index_raises():
    leaves = [b"a", b"b", b"c"]
    with pytest.raises(ValueError):
        rfc6962_inclusion_proof(leaves, 3)
    with pytest.raises(ValueError):
        rfc6962_inclusion_proof(leaves, 5)


def test_inclusion_proof_negative_index_raises():
    with pytest.raises(ValueError):
        rfc6962_inclusion_proof([b"a", b"b"], -1)


def test_inclusion_proof_empty_leaves_raises():
    with pytest.raises(ValueError):
        rfc6962_inclusion_proof([], 0)


def test_inclusion_proof_verify_roundtrip():
    leaves = [bytes([i]) for i in range(8)]
    root = rfc6962_merkle_root(leaves)
    for i in range(8):
        proof = rfc6962_inclusion_proof(leaves, i)
        assert _verify_inclusion(leaves[i], i, 8, proof, root), f"failed for i={i}"


@pytest.mark.academic_reference
def test_inclusion_proof_rfc6962_test_vectors():
    """RFC 6962 §2.1 / §2.1.1 roundtrip for various tree sizes."""
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 16]:
        leaves = [bytes([i % 256]) for i in range(n)]
        root = rfc6962_merkle_root(leaves)
        for i in range(n):
            proof = rfc6962_inclusion_proof(leaves, i)
            assert _verify_inclusion(leaves[i], i, n, proof, root), \
                f"verify failed: n={n}, i={i}"


def test_inclusion_proof_non_list_raises():
    with pytest.raises(TypeError):
        rfc6962_inclusion_proof((b"a", b"b"), 0)


def test_inclusion_proof_non_bytes_leaf_raises():
    with pytest.raises(TypeError):
        rfc6962_inclusion_proof(["not bytes", "also not bytes"], 0)


def test_verify_inclusion_single_leaf():
    """_verify_inclusion with 1-leaf tree (proof=[])."""
    leaf = b"solo"
    root = rfc6962_merkle_root([leaf])
    assert _verify_inclusion(leaf, 0, 1, [], root)
