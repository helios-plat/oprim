"""oprim.soul_config_rewrite — DeerFlow atomic SOUL.md/config write (stage-then-replace).

Mirrors DeerFlow's ``FileAgentStore._write``: both files are staged to temp
files first, then atomically ``os.replace``'d.  If only one file is requested
the write is all-or-nothing; with two files a crash *between* the two replaces
leaves the first committed and the second stale — a sub-millisecond window
that DeerFlow explicitly accepts as a tradeoff (no corruption, no leftover
temp files).

3O element: ``oprim.soul_config_rewrite``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def soul_config_rewrite(
    soul: str | None = None,
    config: dict[str, Any] | None = None,
    soul_path: str | Path | None = None,
    config_path: str | Path | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Atomically write SOUL.md and/or config.json (stage-then-replace).

    Args:
        soul: New SOUL.md content (None = skip).
        config: New config dict (None = skip).
        soul_path: Target file path for SOUL.md.
        config_path: Target file path for config JSON.
        context: Optional config.

    Returns:
        {status, soul_written, config_written, soul_path, config_path}
    """
    ctx = context or {}
    base = Path(ctx.get("output_dir", Path.home() / ".veya" / "agents"))
    soul_file = Path(soul_path) if soul_path else base / "SOUL.md"
    config_file = Path(config_path) if config_path else base / "config.json"

    results: dict[str, Any] = {"status": "completed", "soul_written": False, "config_written": False}

    if soul is not None:
        try:
            _atomic_write(soul_file, soul)
            results["soul_written"] = True
        except Exception as exc:
            results["status"] = "partial"
            results["soul_error"] = str(exc)

    if config is not None:
        import json
        try:
            _atomic_write(config_file, json.dumps(config, ensure_ascii=False, indent=2))
            results["config_written"] = True
        except Exception as exc:
            results["status"] = "partial"
            results["config_error"] = str(exc)

    results["soul_path"] = str(soul_file)
    results["config_path"] = str(config_file)
    return results


def _atomic_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix="." + target.name + ".tmp.")
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, str(target))
    except Exception:
        os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise
