"""TranslationCheckpoint — persist chunk-by-chunk progress for long documents."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TranslationCheckpoint:
    """File-backed checkpoint for resumable multi-chunk translations.

    Each chunk is keyed by its integer index. Completed chunks are written
    to disk immediately so progress survives process restarts.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    # ── read ──────────────────────────────────────────────────────────────────

    def is_done(self, chunk_index: int) -> bool:
        return str(chunk_index) in self._data.get("chunks", {})

    def get_chunk(self, chunk_index: int) -> str | None:
        return self._data.get("chunks", {}).get(str(chunk_index))

    def completed_indices(self) -> set[int]:
        return {int(k) for k in self._data.get("chunks", {}).keys()}

    # ── write ─────────────────────────────────────────────────────────────────

    def save_chunk(self, chunk_index: int, translated_text: str) -> None:
        self._data.setdefault("chunks", {})[str(chunk_index)] = translated_text
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def clear(self) -> None:
        self._data = {}
        if self.path.exists():
            self.path.unlink()

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
