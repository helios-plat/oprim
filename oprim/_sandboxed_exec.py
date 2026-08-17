"""oprim.sandboxed_exec — one argv exec inside ProcessJail. No shell."""

from __future__ import annotations

from typing import Any

from obase.sandbox.process_jail import ProcessJail


def sandboxed_exec(
    jail: ProcessJail,
    *,
    argv: list[str],
    timeout_s: int | None = None,
    cwd: str = ".",
) -> dict[str, Any]:
    """Run argv in the jail. Returns structured stdout/stderr. Never raises Timeout."""
    code, out, err = jail.run(argv, timeout=timeout_s, cwd=cwd)
    return {
        "ok": code == 0,
        "exit_code": code,
        "stdout": out,
        "stderr": err,
        "timed_out": code == 124 and err == "timed out",
        "error": err if code != 0 else "",
    }
