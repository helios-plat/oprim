"""Tests for TranslationCheckpoint."""
import json
import pytest
from pathlib import Path
from oprim.translate.checkpoint import TranslationCheckpoint


def test_save_and_retrieve(tmp_path):
    cp_path = tmp_path / "checkpoint.json"
    cp = TranslationCheckpoint(cp_path)
    cp.save_chunk(0, "translated text")
    assert cp.is_done(0)
    assert cp.get_chunk(0) == "translated text"
    assert not cp.is_done(1)


def test_persists_across_instances(tmp_path):
    cp_path = tmp_path / "checkpoint.json"
    cp1 = TranslationCheckpoint(cp_path)
    cp1.save_chunk(5, "hello")
    cp2 = TranslationCheckpoint(cp_path)
    assert cp2.is_done(5)
    assert cp2.get_chunk(5) == "hello"


def test_clear_removes_file(tmp_path):
    cp_path = tmp_path / "checkpoint.json"
    cp = TranslationCheckpoint(cp_path)
    cp.save_chunk(0, "x")
    assert cp_path.exists()
    cp.clear()
    assert not cp_path.exists()
    assert not cp.is_done(0)


def test_completed_indices(tmp_path):
    cp = TranslationCheckpoint(tmp_path / "cp.json")
    cp.save_chunk(0, "a")
    cp.save_chunk(3, "b")
    assert cp.completed_indices() == {0, 3}


def test_missing_file_starts_empty(tmp_path):
    cp = TranslationCheckpoint(tmp_path / "nonexistent.json")
    assert cp.completed_indices() == set()
    assert cp.get_chunk(0) is None
