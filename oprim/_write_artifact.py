"""oprim.write_artifact / write_proposal — jail writes to engineering/."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENG = Path(".veya-project") / "engineering"


def _eng_root(project_root: Path) -> Path:
    return (project_root / _ENG).resolve()


def _jail(project_root: Path, relpath: str | Path) -> tuple[Path | None, str]:
    eng = _eng_root(project_root)
    target = (eng / relpath).resolve()
    try:
        target.relative_to(eng)
    except ValueError:
        return None, f"artifact path escapes {_ENG}: {relpath}"
    return target, ""


def write_artifact(
    *,
    project_root: str | Path,
    relpath: str,
    content: str,
    kind: str = "artifact",
) -> dict[str, Any]:
    """Write a text artifact under ``.veya-project/engineering/`` only.

    Args:
        project_root: Project root.
        relpath: Path relative to ``engineering/`` (not the repo root).
        content: File body.
        kind: Free-form label stored in the result.

    Returns:
        ``{ok, path, kind, error?}``
    """
    root = Path(project_root).resolve()
    target, err = _jail(root, relpath)
    if target is None:
        return {"ok": False, "path": "", "kind": kind, "error": err}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(target), "kind": kind}


def write_proposal(
    *,
    project_root: str | Path,
    title: str,
    body: str,
    slug: str | None = None,
) -> dict[str, Any]:
    """Write a formal proposal under ``engineering/proposals/``. Never writes business source."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = _slug(slug or title)
    relpath = Path("proposals") / f"{stamp}-{safe}.md"
    header = f"# {title}\n\n"
    content = body if body.lstrip().startswith("#") else header + body
    result = write_artifact(
        project_root=project_root,
        relpath=str(relpath),
        content=content,
        kind="proposal",
    )
    result["title"] = title
    return result


def _slug(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", ".", "/"}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "proposal")[:60]
