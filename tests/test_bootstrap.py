"""Tests for oprim.bootstrap."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oprim.bootstrap import bootstrap


class TestBootstrap:
    def test_creates_stratum_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        bootstrap(log_level="WARNING")
        stratum_dir = tmp_path / ".stratum"
        for subdir in ["inbox", "data/substrate", "_archive", "index/lance", "index/tantivy"]:
            assert (stratum_dir / subdir).is_dir(), f"Missing: {subdir}"

    def test_bootstrap_with_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        config_file = tmp_path / "config.yaml"
        config_file.write_text("STRATUM_CUSTOM: testvalue\n")
        bootstrap(config_path=config_file, log_level="WARNING")
        from oprim._config import cfg
        assert cfg.get("STRATUM_CUSTOM") == "testvalue"

    def test_bootstrap_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        bootstrap(log_level="WARNING")
        bootstrap(log_level="WARNING")  # second call must not raise
        stratum_dir = tmp_path / ".stratum"
        assert (stratum_dir / "inbox").is_dir()

    def test_bootstrap_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        bootstrap()  # should not raise with defaults
