"""oprim._asr_runtime — reusable local ASR model handles (infra, not a capability).

`transcribe_audio` used to construct a `WhisperModel` on *every* call, so an
N-segment transcription paid N model loads. Model lifetime is a runtime
concern, not an OPrim concern, so the cache lives here — behind an explicit,
clearable handle — rather than as module-global state inside the element.

Two properties matter and are tested:

* **Reuse**: the same (path, device, compute_type) triple loads once.
* **No hidden global**: `clear_asr_cache()` empties it, and a caller that needs
  isolation can pass its own `AsrModelCache` instead of the shared default.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

__oprim_layer__ = "infra"  # runtime base for the transcribe element

#: Key components are hashed into one string so the cache key is a single value.
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


class AsrModelCache:
    """Thread-safe, explicitly clearable cache of loaded local ASR models."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()
        self.load_count = 0

    def get(self, *, model_path: Path, device: str, compute_type: str) -> Any:
        """Return the cached model, loading it on first request for this key."""
        key = (str(model_path), device, compute_type)
        with self._lock:
            model = self._models.get(key)
            if model is not None:
                return model
        # Load outside the lock: model construction is slow and must not block
        # other keys. A concurrent duplicate load is harmless (idempotent).
        model = self._load(model_path, device, compute_type)
        with self._lock:
            self.load_count += 1
            return self._models.setdefault(key, model)

    @staticmethod
    def _load(model_path: Path, device: str, compute_type: str) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install with: pip install faster-whisper"
            ) from exc
        return WhisperModel(str(model_path), device=device, compute_type=compute_type)

    def clear(self) -> None:
        with self._lock:
            self._models.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


#: Process-wide default. Explicitly clearable; never written to by elements.
_DEFAULT_CACHE = AsrModelCache()


def default_asr_cache() -> AsrModelCache:
    """The shared process-wide ASR model cache."""
    return _DEFAULT_CACHE


def clear_asr_cache() -> None:
    """Drop every cached local ASR model (frees memory / forces reload)."""
    _DEFAULT_CACHE.clear()
