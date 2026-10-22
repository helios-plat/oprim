"""3O 范式内化 — oprim 确定性原语测试."""

from __future__ import annotations

from oprim import (
    crux_hash,
    extract_crux,
    parse_compile_error,
    parse_symbols,
    parse_test_output,
    retrieve_rules,
    retrieve_similar,
)


def test_parse_symbols_python():
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "class Service:\n"
        "    def handle(self, req):\n"
        "        pass\n"
    )
    symbols = parse_symbols(code, language="python")
    kinds = {s["kind"] for s in symbols}
    names = [s["name"] for s in symbols]
    assert kinds == {"import", "class", "function"}
    assert "os" in names
    assert "pathlib.Path" in names
    assert "Service" in names
    assert "handle" in names
    svc = next(s for s in symbols if s["name"] == "Service")
    assert svc["line"] == 4


def test_parse_symbols_ts_fallback_no_crash():
    # tree-sitter 未安装时非 Python 语言返回空集 (不炸)
    symbols = parse_symbols("function foo() {}", language="typescript")
    assert isinstance(symbols, list)


def test_extract_crux_stable_across_line_shifts():
    code_v1 = "def f(a):\n    # comment\n    return a + 1\n\n"
    code_v2 = "# header\n\n\ndef f(a):\n    return a + 1\n"  # 行号/注释变动
    assert extract_crux(code_v1) == extract_crux(code_v2)
    assert crux_hash(code_v1) == crux_hash(code_v2)


def test_extract_crux_non_python_normalized():
    js = "function f() {\n  // comment\n  return 1;\n}"
    crux = extract_crux(js)
    assert "comment" not in crux
    assert "return 1;" in crux


def test_parse_test_output_pytest():
    out = "collected 13 items\n......F....F..\n=== 11 passed, 2 failed in 1.42s ===\n"
    parsed = parse_test_output(out)
    assert parsed["tests_passed"] == 11
    assert parsed["tests_failed"] == 2
    assert parsed["execution_time_ms"] == 1420


def test_parse_compile_error():
    out = 'File "src/main.py", line 12\n    return  x + \nSyntaxError: invalid syntax\n'
    errors = parse_compile_error(out)
    assert errors and errors[0]["file"] == "src/main.py"
    assert errors[0]["line"] == "12"


def test_retrieve_similar_ranks():
    corpus = [
        "function to refresh expired JWT tokens",
        "JWT token refresh middleware setup",
        "CSS button styling for dark mode",
    ]
    hits = retrieve_similar("refresh token expired jwt", corpus=corpus, top_k=2)
    assert hits[0]["index"] == 0
    assert hits[0]["score"] > hits[1]["score"]
    assert "button" not in [h["text"] for h in hits]


def test_retrieve_rules_keyword():
    rules = [
        "never call HTTP inside a DB transaction",
        "always close file handles after reading",
    ]
    hits = retrieve_rules("http call transaction deadlock", rules=rules, top_k=1)
    assert hits[0]["index"] == 0
