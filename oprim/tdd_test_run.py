"""oprim.tdd_test_run — real TDD test execution (subprocess pytest / pnpm test).

Spawns an isolated test run via ``subprocess.run``, parses the exit code and
stdout/stderr, and reports pass/fail + stacktrace excerpts.

3O element: ``oprim.tdd_test_run`` (``_tdd_test_run`` legacy name).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def tdd_test_run(
    project_dir: str,
    test_pattern: str | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run unit tests in an isolated directory and report results.

    Args:
        project_dir: Path to the project (must contain a valid test runner config).
        test_pattern: Optional pytest expression / glob (``-k`` flag).
        context: Optional config override dict.

    Returns:
        ``{status, total, passed, failed, errors, stdout, stderr, exit_code}``
    """
    ctx = context or {}
    runner = ctx.get("test_runner", "pytest")  # "pytest" | "pnpm test" | "cargo test"

    project = Path(project_dir).resolve()
    if not project.is_dir():
        return {"status": "failed", "error": f"not a directory: {project_dir}", "exit_code": -1}

    if runner == "pytest":
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if test_pattern:
            cmd.extend(["-k", test_pattern])
    elif runner.startswith("pnpm"):
        cmd = ["pnpm", "test"]
        parts = runner.split()
        if len(parts) > 1:
            cmd = parts
    elif runner.startswith("cargo"):
        cmd = ["cargo", "test"]
    else:
        cmd = runner.split()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project),
            capture_output=True,
            text=True,
            timeout=int(ctx.get("timeout", 120)),
        )
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "test run timed out", "exit_code": -1}
    except FileNotFoundError:
        return {"status": "failed", "error": f"runner not found: {cmd[0]}", "exit_code": -1}

    # Parse pytest-style summary: "X passed, Y failed" or fall back to exit code
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    passed = 0
    failed = 0
    errors = 0
    if " passed" in stdout or " failed" in stdout:
        import re
        m = re.search(r"(\d+) passed", stdout)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", stdout)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+) error", stdout)
        if m: errors = int(m.group(1))

    # Extract last few lines of stderr as a focused stacktrace excerpt
    stderr_lines = stderr.splitlines()
    excerpt = "\n".join(stderr_lines[-12:]) if len(stderr_lines) > 12 else stderr

    return {
        "status": "completed" if proc.returncode == 0 else "failed",
        "total": passed + failed + errors or 1,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "exit_code": proc.returncode,
        "stdout": stdout[-2000:],
        "stderr_excerpt": excerpt[-800:],
    }
