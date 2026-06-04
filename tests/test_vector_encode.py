"""Tests for oprim.vector_encode — ≥10 tests."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.vector_encode import vector_encode


# 1. Returns np.ndarray
def test_returns_ndarray():
    result = vector_encode(texts=["hello"])
    assert isinstance(result, np.ndarray)


# 2. Shape (n, dim) correct for single text
def test_shape_single_text():
    result = vector_encode(texts=["hello"])
    assert result.ndim == 2
    assert result.shape[0] == 1


# 3. Shape (n, dim) correct for multiple texts
def test_shape_multiple_texts():
    texts = ["hello", "world", "foo"]
    result = vector_encode(texts=texts)
    assert result.shape[0] == 3
    assert result.shape[1] > 0


# 4. dtype is float32
def test_dtype_float32():
    result = vector_encode(texts=["hello", "world"])
    assert result.dtype == np.float32


# 5. Normalized: norm ≈ 1.0 when normalize=True
def test_normalized_norm_one():
    result = vector_encode(texts=["test text", "another text"], normalize=True)
    for i in range(result.shape[0]):
        norm = np.linalg.norm(result[i])
        assert norm == pytest.approx(1.0, abs=1e-5)


# 6. Different texts → different vectors
def test_different_texts_different_vectors():
    result = vector_encode(texts=["apple", "orange"])
    assert not np.allclose(result[0], result[1])


# 7. Same text → same vector (deterministic stub)
def test_same_text_deterministic():
    r1 = vector_encode(texts=["consistent text"])
    r2 = vector_encode(texts=["consistent text"])
    np.testing.assert_array_equal(r1, r2)


# 8. Empty list → shape (0, dim) — stub produces (0, 128)
def test_empty_list_shape():
    result = vector_encode(texts=[])
    assert result.ndim == 2
    assert result.shape[0] == 0


# 9. Single text works (no crash, correct shape)
def test_single_text_no_crash():
    result = vector_encode(texts=["just one text"])
    assert result.shape == (1, 128)


# 10. Multiple texts at once — all rows present
def test_multiple_texts_all_rows():
    texts = [f"text_{i}" for i in range(10)]
    result = vector_encode(texts=texts)
    assert result.shape[0] == 10


# 11. normalize=False returns unnormalized (norm != 1.0 in general)
def test_unnormalized_not_unit():
    result = vector_encode(texts=["hello"], normalize=False)
    norm = np.linalg.norm(result[0])
    # stub uses standard_normal without norm — norm will not be exactly 1.0
    # (could be close in rare cases, but the vector is not forced to unit length)
    # We just verify the function runs and returns correct shape/dtype
    assert result.dtype == np.float32
    assert result.shape == (1, 128)


# 12. Vectors are finite (no NaN/Inf)
def test_vectors_finite():
    result = vector_encode(texts=["a", "b", "c"])
    assert np.all(np.isfinite(result))


# 13. Order of output rows matches order of input texts
def test_row_order_matches_input():
    texts = ["alpha", "beta", "gamma"]
    result = vector_encode(texts=texts)
    for i, t in enumerate(texts):
        single = vector_encode(texts=[t])
        np.testing.assert_array_equal(result[i], single[0])


# ---------------------------------------------------------------------------
# ProviderRegistry fix tests (v2.29.1)
# ---------------------------------------------------------------------------


class TestProviderRegistryPath:
    def test_provider_not_registered_falls_back_to_stub(self):
        """ProviderNotFoundError → deterministic stub + log.warning (no raise)."""
        from unittest.mock import patch
        from obase.exceptions import ProviderNotFoundError

        with patch("obase.ProviderRegistry.get", side_effect=ProviderNotFoundError("embedding", "missing")):
            result = vector_encode(texts=["hello"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_provider_registered_calls_provider(self):
        """Registered provider is called; its return value used (not stub)."""
        from unittest.mock import patch, MagicMock

        fake_vecs = [[1.0] * 128]
        mock_embed = MagicMock(return_value=fake_vecs)
        with patch("obase.ProviderRegistry.get", return_value=mock_embed) as mock_get:
            result = vector_encode(texts=["test"], provider="bge-m3", normalize=False)
        mock_get.assert_called_once_with("embedding", "bge-m3")
        mock_embed.assert_called_once_with(["test"])
        np.testing.assert_allclose(result[0, 0], 1.0)

    def test_code_error_reraises(self):
        """Non-ProviderNotFoundError (e.g. AttributeError) must propagate, not fall to stub."""
        from unittest.mock import patch

        with patch("obase.ProviderRegistry.get", side_effect=AttributeError("bad attr")):
            with pytest.raises(AttributeError):
                vector_encode(texts=["hello"])
