"""oprim.run_targeted_checks — change-scoped checks; never full pytest by default."""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oprim._read_standards import DEFAULT_CHECK_MAP

Runner = Callable[..., dict[str, Any]]


def run_targeted_checks(
    *,
    project_root: str | Path,
    files: list[str] | None = None,
    force_full: bool = False,
    check_map: dict[str, Any] | None = None,
    runner: Runner | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    """Run the smallest check set implied by ``files``.

    Full ``pytest`` (no path args) is forbidden unless ``force_full=True``.
    Markdown-only / empty change sets skip heavy runners.

    ``runner(cmd, cwd=..., timeout=...)`` is injectable; default is subprocess.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        return {
            "ok": False,
            "skipped": False,
            "reason": f"not a directory: {project_root}",
            "commands": [],
            "reports": [],
        }

    changed = [f.replace("\\", "/") for f in (files or []) if f and f != "/dev/null"]
    cmap = check_map or DEFAULT_CHECK_MAP
    commands: list[dict[str, Any]] = []

    if not changed and not force_full:
        return {
            "ok": True,
            "skipped": True,
            "reason": "no changes",
            "commands": [],
            "reports": [],
        }

    wanted = _wanted_checks(changed, cmap) if changed else set()
    if force_full:
        wanted = wanted | {"pytest_related", "ruff_or_flake8"}
        if (root / "package.json").is_file():
            wanted.add("npm_test_related")

    if not wanted and not force_full:
        return {
            "ok": True,
            "skipped": True,
            "reason": "md-only or no mapped checks",
            "commands": [],
            "reports": [],
            "wanted": [],
        }

    ok = True
    if "ruff_or_flake8" in wanted:
        rec = _run_lint(root, changed, runner, timeout)
        commands.append(rec)
        if rec.get("ran") and rec.get("code", 0) != 0:
            ok = False
    if "pytest_related" in wanted or force_full:
        rec = _run_pytest(root, changed, force_full, runner, timeout)
        commands.append(rec)
        if rec.get("ran") and rec.get("code", 0) != 0:
            ok = False
        if rec.get("forbidden"):
            ok = False
    if "npm_test_related" in wanted:
        rec = _run_npm(root, changed, runner, timeout)
        commands.append(rec)
        if rec.get("ran") and rec.get("code", 0) != 0:
            ok = False

    return {
        "ok": ok,
        "skipped": not any(c.get("ran") for c in commands),
        "reason": "" if any(c.get("ran") for c in commands) else "mapped checks had nothing to run",
        "commands": commands,
        "reports": [c for c in commands if c.get("ran")],
        "wanted": sorted(wanted),
    }


def _wanted_checks(files: list[str], check_map: dict[str, Any]) -> set[str]:
    wanted: set[str] = set()
    rules = check_map.get("rules") if isinstance(check_map, dict) else None
    if not isinstance(rules, list):
        rules = DEFAULT_CHECK_MAP["rules"]
    for path in files:
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            glob = str(rule.get("glob") or "")
            if glob and _match_glob(path, glob):
                for check in rule.get("checks") or []:
                    wanted.add(str(check))
    return wanted


def _match_glob(path: str, glob: str) -> bool:
    patterns = _expand_braces(glob)
    expanded: list[str] = []
    for pat in patterns:
        expanded.append(pat)
        if pat.startswith("**/"):
            expanded.append(pat[3:])
    name = Path(path).name
    return any(fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(name, pat) for pat in expanded)


def _expand_braces(glob: str) -> list[str]:
    if "{" not in glob or "}" not in glob:
        return [glob]
    start = glob.index("{")
    end = glob.index("}")
    inner = glob[start + 1 : end]
    prefix, suffix = glob[:start], glob[end + 1 :]
    return [prefix + alt + suffix for alt in inner.split(",")]


def _exec(cmd: list[str], *, cwd: Path, timeout: int, runner: Runner | None) -> dict[str, Any]:
    if runner is not None:
        rec = runner(cmd, cwd=str(cwd), timeout=timeout)
        rec.setdefault("cmd", cmd)
        rec.setdefault("ran", True)
        return rec
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "ran": True,
            "code": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    except FileNotFoundError:
        return {
            "cmd": cmd,
            "ran": False,
            "code": 127,
            "stdout": "",
            "stderr": f"{cmd[0]} not found",
        }
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "ran": True, "code": 124, "stdout": "", "stderr": "timed out"}


def _run_lint(root: Path, files: list[str], runner: Runner | None, timeout: int) -> dict[str, Any]:
    py_files = [f for f in files if f.endswith(".py")]
    if not py_files:
        return {"name": "ruff_or_flake8", "ran": False, "reason": "no python files"}
    ruff = shutil.which("ruff")
    if ruff or (root / "ruff.toml").is_file() or _pyproject_has(root, "ruff"):
        cmd = [ruff or "ruff", "check", *py_files]
        rec = _exec(cmd, cwd=root, timeout=timeout, runner=runner)
        rec["name"] = "ruff"
        return rec
    flake = shutil.which("flake8")
    if flake:
        rec = _exec([flake, *py_files], cwd=root, timeout=timeout, runner=runner)
        rec["name"] = "flake8"
        return rec
    return {"name": "ruff_or_flake8", "ran": False, "reason": "ruff/flake8 not present"}


def _run_pytest(
    root: Path,
    files: list[str],
    force_full: bool,
    runner: Runner | None,
    timeout: int,
) -> dict[str, Any]:
    if force_full:
        rec = _exec(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            timeout=timeout,
            runner=runner,
        )
        rec["name"] = "pytest_full"
        rec["force_full"] = True
        return rec
    targets = _related_pytest_targets(root, files)
    if not targets:
        return {"name": "pytest_related", "ran": False, "reason": "no related tests"}
    rec = _exec(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=root,
        timeout=timeout,
        runner=runner,
    )
    rec["name"] = "pytest_related"
    rec["targets"] = targets
    # Guard: a runner that dropped path args is treated as a contract break.
    cmd = rec.get("cmd") or []
    if isinstance(cmd, list) and cmd[-2:] == ["-m", "pytest"] or (
        isinstance(cmd, list) and len(cmd) >= 3 and cmd[-1] == "-q" and "pytest" in cmd
    ):
        rec["forbidden"] = True
        rec["reason"] = "full pytest without path args is forbidden unless force_full=true"
    return rec


def _related_pytest_targets(root: Path, files: list[str]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = Path(rel)
        posix = rel.replace("\\", "/")
        if path.name.startswith("test_") or path.name.endswith("_test.py") or "/tests/" in posix:
            if rel not in seen:
                seen.add(rel)
                targets.append(rel)
            continue
        candidates = [
            f"tests/test_{path.stem}.py",
            str(path.parent / f"test_{path.stem}.py"),
            f"tests/{path.parent}/test_{path.stem}.py" if str(path.parent) != "." else "",
        ]
        for cand in candidates:
            if not cand:
                continue
            cand_n = cand.replace("\\", "/")
            if cand_n in seen:
                continue
            if (root / cand_n).is_file():
                seen.add(cand_n)
                targets.append(cand_n)
    return targets


def _run_npm(root: Path, files: list[str], runner: Runner | None, timeout: int) -> dict[str, Any]:
    pkg = root / "package.json"
    if not pkg.is_file():
        return {"name": "npm_test_related", "ran": False, "reason": "no package.json"}
    scripts: dict[str, Any] = {}
    try:
        scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts") or {}
    except json.JSONDecodeError:
        scripts = {}
    npm = shutil.which("npm") or "npm"
    js_ext = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
    related = [f for f in files if f.endswith(js_ext) or f == "package.json"]
    if "test:related" in scripts:
        rec = _exec(
            [npm, "run", "test:related", "--", *related],
            cwd=root,
            timeout=timeout,
            runner=runner,
        )
        rec["name"] = "npm_test_related"
        return rec
    rec = _exec(
        [npm, "test", "--", "--related", *related],
        cwd=root,
        timeout=timeout,
        runner=runner,
    )
    rec["name"] = "npm_test_related"
    return rec


def _pyproject_has(root: Path, tool: str) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return f"[tool.{tool}" in text
