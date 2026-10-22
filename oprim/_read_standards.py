"""oprim.read_standards — load project STANDARDS.md or builtin baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_ENG = Path(".veya-project") / "engineering"
_STANDARDS_REL = _ENG / "STANDARDS.md"

BUILTIN_STANDARDS = """# Engineering Standards (builtin baseline)

## Correctness
- Changes must match the stated intent; no silent behavior change.
- Handle error paths; do not swallow exceptions without a reason.
- Prefer the smallest change that satisfies the request.

## Security
- No unsanitized shell interpolation, eval/exec on untrusted input, or secret leakage.
- Do not weaken authz, path jail, or SSRF checks.
- Do not write credentials, tokens, or private keys into the tree.

## Lifecycle
- New behavior has a test, or an explicit reason it cannot.
- Do not leave TODOs that hide incomplete delivery.
- Prefer deleting unused code over commenting it out.
- Do not introduce a second task queue, dispatcher, or parallel Coordinator router.
"""

DEFAULT_CHECK_MAP: dict[str, Any] = {
    "rules": [
        {
            "glob": "**/*.py",
            "checks": ["pytest_related", "ruff_or_flake8"],
        },
        {
            "glob": "**/*.{js,jsx,ts,tsx,mjs,cjs}",
            "checks": ["npm_test_related"],
        },
        {
            "glob": "package.json",
            "checks": ["npm_test_related"],
        },
        {
            "glob": "**/*.{md,rst,txt,adoc}",
            "checks": [],
        },
    ]
}


def read_standards(*, project_root: str | Path) -> dict[str, Any]:
    """Read ``.veya-project/engineering/STANDARDS.md`` or the builtin baseline.

    Returns:
        ``{ok, text, path, standards_source: project|builtin, check_map, check_map_source}``
    """
    root = Path(project_root).resolve()
    std_path = root / _STANDARDS_REL
    if std_path.is_file():
        text = std_path.read_text(encoding="utf-8")
        source = "project"
        path = str(std_path)
    else:
        text = BUILTIN_STANDARDS
        source = "builtin"
        path = ""

    check_map, map_source = _load_check_map(root)
    return {
        "ok": True,
        "text": text,
        "path": path,
        "standards_source": source,
        "check_map": check_map,
        "check_map_source": map_source,
    }


def _load_check_map(root: Path) -> tuple[dict[str, Any], str]:
    candidate = root / _ENG / "check-map.yml"
    if not candidate.is_file():
        alt = root / _ENG / "check-map.yaml"
        candidate = alt if alt.is_file() else candidate
    if not candidate.is_file():
        return dict(DEFAULT_CHECK_MAP), "builtin"
    raw = candidate.read_text(encoding="utf-8")
    parsed = _parse_check_map(raw)
    if parsed is None:
        return dict(DEFAULT_CHECK_MAP), "builtin"
    return parsed, "project"


def _parse_check_map(raw: str) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        data = yaml.safe_load(raw)
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data
        return None
    # Minimal fallback: accept JSON-shaped files even with a .yml suffix.
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        return data
    return None
