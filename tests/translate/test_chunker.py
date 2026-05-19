"""Tests for TranslationChunker."""
import pytest
from oprim.translate.chunker import TextChunk, TranslationChunker


def test_split_plain_text():
    chunker = TranslationChunker(max_chars=100)
    text = "Hello world.\n\nThis is a second paragraph."
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    assert all(c.translatable for c in chunks)


def test_code_fence_not_translatable():
    chunker = TranslationChunker(max_chars=2000)
    text = "Intro.\n\n```python\nprint('hello')\n```\n\nOutro."
    chunks = chunker.split(text)
    code_chunks = [c for c in chunks if not c.translatable]
    prose_chunks = [c for c in chunks if c.translatable]
    assert len(code_chunks) == 1
    assert "print" in code_chunks[0].text
    assert len(prose_chunks) >= 1


def test_long_prose_splits():
    chunker = TranslationChunker(max_chars=50)
    para = "A" * 30
    text = "\n\n".join([para] * 5)
    chunks = chunker.split(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.translatable


def test_join_preserves_order():
    chunker = TranslationChunker(max_chars=2000)
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunker.split(text)
    rejoined = chunker.join(chunks)
    assert "Para one" in rejoined
    assert "Para three" in rejoined


def test_indices_are_sequential():
    chunker = TranslationChunker(max_chars=20)
    text = "A" * 10 + "\n\n" + "B" * 10 + "\n\n" + "C" * 10
    chunks = chunker.split(text)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_empty_text():
    chunker = TranslationChunker()
    chunks = chunker.split("")
    assert chunks == []


def test_only_code_fence():
    chunker = TranslationChunker()
    text = "```\ncode only\n```"
    chunks = chunker.split(text)
    assert len(chunks) == 1
    assert not chunks[0].translatable
