"""3O 范式内化 — 测试输出结构化解析 (oprim 确定性原语, 无 LLM).

将编译器/测试框架输出提取为结构化结果。
"""

from __future__ import annotations

import re
from typing import Any

_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_ERRORS_RE = re.compile(r"(\d+)\s+errors?", re.IGNORECASE)
_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)
_DURATION_RE = re.compile(r"in\s+([\d.]+)\s*s", re.IGNORECASE)
_ERROR_LINE_RE = re.compile(r"^(?:File\s+\"([^\"]+)\",\s+line\s+(\d+)|(.+?):(\d+):)")


def parse_test_output(output: str) -> dict[str, Any]:
    """单次原子解析: 提取通过/失败/跳过/耗时/错误行。

    Returns:
        {tests_passed, tests_failed, tests_skipped, errors, execution_time_ms}
    """
    passed = _count(_PASSED_RE, output)
    failed = _count(_FAILED_RE, output)
    errors = _count(_ERRORS_RE, output)
    skipped = _count(_SKIPPED_RE, output)
    duration = _DURATION_RE.search(output)

    error_lines = [
        line.strip()
        for line in output.splitlines()
        if re.search(r"(error|Error|failed|FAIL|AssertionError)", line)
        and not line.strip().startswith(("ok ", "PASS"))
    ][:20]

    return {
        "tests_passed": passed,
        "tests_failed": failed + errors,
        "tests_skipped": skipped,
        "errors": error_lines,
        "execution_time_ms": int(float(duration.group(1)) * 1000) if duration else 0,
    }


def parse_compile_error(output: str) -> list[dict[str, str]]:
    """单次原子解析: 提取编译器报错位置 (file/line/message)。"""
    results: list[dict[str, str]] = []
    for line in output.splitlines():
        m = _ERROR_LINE_RE.search(line)
        if m:
            file, line_no, alt_file, alt_line = m.groups()
            results.append(
                {
                    "file": file or alt_file or "",
                    "line": line_no or alt_line or "",
                    "message": line.strip()[:200],
                }
            )
    return results


def _count(pattern: re.Pattern, output: str) -> int:
    matches = pattern.findall(output)
    return max(int(m) for m in matches) if matches else 0
