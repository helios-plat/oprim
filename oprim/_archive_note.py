"""oprim.archive_note — promote or suppress an agent note by future value."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

_ENG = Path(".veya-project") / "engineering"


def archive_note(
    *,
    project_root: str | Path,
    title: str,
    body: str,
    decision: Literal["promote", "suppress"],
    reason: str = "",
    source_path: str = "",
) -> dict[str, Any]:
    """Write a note into ``engineering/archive/`` or ``engineering/archive/suppressed/``.

    Does not write business source. Caller decides promote vs suppress.
    """
    if decision not in {"promote", "suppress"}:
        return {
            "ok": False,
            "path": "",
            "decision": decision,
            "error": f"decision must be promote|suppress, got {decision!r}",
        }
    root = Path(project_root).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = _slug(title)
    if decision == "promote":
        rel = Path("archive") / f"{stamp}-{safe}.md"
    else:
        rel = Path("archive") / "suppressed" / f"{stamp}-{safe}.md"
    eng = (root / _ENG).resolve()
    target = (eng / rel).resolve()
    try:
        target.relative_to(eng)
    except ValueError:
        return {"ok": False, "path": "", "decision": decision, "error": "path jail failed"}
    header = [
        f"# {title}",
        "",
        f"- decision: {decision}",
        f"- reason: {reason or '(none)'}",
    ]
    if source_path:
        header.append(f"- source: {source_path}")
    header.append("")
    content = "\n".join(header) + "\n" + body.rstrip() + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    if source_path:
        src = Path(source_path)
        if src.is_file():
            try:
                src.relative_to((root / _ENG / "notes-inbox").resolve())
            except ValueError:
                src = None
            if src is not None and src.exists():
                src.unlink()
    return {
        "ok": True,
        "path": str(target),
        "decision": decision,
        "reason": reason,
        "title": title,
    }


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
    return (slug or "note")[:60]
