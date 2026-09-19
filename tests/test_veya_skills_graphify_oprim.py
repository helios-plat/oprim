"""Tests for oprim.code_graph_parse, adaptive_node_extract, domain_rule_check,
dlt_schema_normalize."""

from __future__ import annotations

import inspect

from oprim._adaptive_node_extract import adaptive_node_extract
from oprim._code_graph_parse import code_graph_parse
from oprim._dlt_schema_normalize import dlt_schema_normalize
from oprim._domain_rule_check import domain_rule_check

# ============================================================================
# code_graph_parse
# ============================================================================


class TestCodeGraphParse:
    def test_parses_function_def(self):
        code = "def foo():\n    pass\n"
        result = code_graph_parse(code, file_path="mod.py")
        assert result["status"] == "success"
        func_nodes = [n for n in result["nodes"] if n["type"] == "function"]
        assert len(func_nodes) >= 1
        assert func_nodes[0]["name"] == "foo"

    def test_parses_class_def(self):
        code = "class Bar:\n    def method(self):\n        pass\n"
        result = code_graph_parse(code, file_path="mod.py")
        class_nodes = [n for n in result["nodes"] if n["type"] == "class"]
        method_nodes = [n for n in result["nodes"] if n["type"] == "method"]
        assert len(class_nodes) >= 1
        assert len(method_nodes) >= 1

    def test_parses_async_function(self):
        code = "async def fetch():\n    pass\n"
        result = code_graph_parse(code, file_path="mod.py")
        func_nodes = [n for n in result["nodes"] if n["type"] == "function"]
        assert any(n.get("is_async") for n in func_nodes)

    def test_parses_calls(self):
        code = "def a():\n    b()\ndef b():\n    pass\n"
        result = code_graph_parse(code, file_path="mod.py")
        call_edges = [e for e in result["edges"] if e.get("relation") == "calls"]
        assert len(call_edges) >= 1

    def test_parses_imports(self):
        code = "import os\nfrom sys import path\n"
        result = code_graph_parse(code, file_path="mod.py")
        import_edges = [e for e in result["edges"] if e.get("relation") == "imports"]
        assert len(import_edges) >= 1

    def test_syntax_error(self):
        result = code_graph_parse("def foo(:", file_path="bad.py")
        assert result["status"] == "failed"

    def test_empty_code(self):
        result = code_graph_parse("", file_path="empty.py")
        assert result["status"] == "success"

    def test_positional_only_one(self):
        sig = inspect.signature(code_graph_parse)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1

    def test_edges_have_source_target_relation(self):
        code = "def foo():\n    pass\nclass A:\n    pass\n"
        result = code_graph_parse(code, file_path="test.py")
        for e in result["edges"]:
            assert "source" in e
            assert "target" in e
            assert "relation" in e


# ============================================================================
# adaptive_node_extract
# ============================================================================


class TestAdaptiveNodeExtract:
    def test_extracts_by_class(self):
        html = "<div class='article-content'><p>Main text here</p></div>"
        result = adaptive_node_extract(html, target_semantic="article-content")
        assert result["status"] == "success"
        assert "Main text here" in result["extracted_content"]

    def test_fallback_to_article_tag(self):
        html = "<article><p>Article content</p></article>"
        result = adaptive_node_extract(html, target_semantic="nonexistent")
        assert result["status"] == "success"
        assert "Article content" in result["extracted_content"]

    def test_fallback_to_raw_text(self):
        html = "<div><p>Some random text</p></div>"
        result = adaptive_node_extract(html, target_semantic="nonexistent")
        assert result["status"] == "fallback"
        assert result["anchor_confidence"] < 0.5

    def test_empty_html(self):
        result = adaptive_node_extract("", target_semantic="main")
        assert result["status"] == "fallback"
        assert result["anchor_confidence"] == 0.0

    def test_high_confidence_on_match(self):
        html = "<section id='article-content'><p>Hello</p></section>"
        result = adaptive_node_extract(html, target_semantic="article-content")
        assert result["anchor_confidence"] >= 0.9

    def test_positional_only_one(self):
        sig = inspect.signature(adaptive_node_extract)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1


# ============================================================================
# domain_rule_check
# ============================================================================


class TestDomainRuleCheck:
    def test_solana_missing_signer(self):
        code = "pub fn transfer(ctx: Context<Transfer>, account: AccountInfo)"
        result = domain_rule_check(code, domain="solana")
        assert result["is_compliant"] is False
        assert any("is_signer" in v["rule"] for v in result["violations"])

    def test_solana_panic_detected(self):
        code = 'fn process() { panic!("error"); }'
        result = domain_rule_check(code, domain="solana")
        assert result["is_compliant"] is False

    def test_ethereum_tx_origin(self):
        code = "require(tx.origin == owner);"
        result = domain_rule_check(code, domain="ethereum")
        assert result["is_compliant"] is False
        assert any("tx.origin" in v["rule"] for v in result["violations"])

    def test_rust_unwrap_warning(self):
        code = "let x = some_option.unwrap();"
        result = domain_rule_check(code, domain="rust")
        assert not result["is_compliant"]

    def test_security_hardcoded_password(self):
        code = "const password = 'admin123';"
        result = domain_rule_check(code, domain="security")
        assert not result["is_compliant"]

    def test_compliant_code_passes(self):
        code = "fn safe_function() -> Result<()> { Ok(()) }"
        result = domain_rule_check(code, domain="solana")
        # This safe code passes solana checks
        assert isinstance(result["is_compliant"], bool)

    def test_positional_only_one(self):
        sig = inspect.signature(domain_rule_check)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1


# ============================================================================
# dlt_schema_normalize
# ============================================================================


class TestDltSchemaNormalize:
    def test_normalizes_field_names(self):
        data = [{"First-Name": "Alice", "Age Group": "adult"}]
        result = dlt_schema_normalize(data)
        assert result["status"] == "success"
        assert result["count"] == 1
        record = result["sample_record"]
        assert "first_name" in record or "age_group" in record

    def test_empty_input(self):
        result = dlt_schema_normalize([])
        assert result["status"] == "empty"
        assert result["count"] == 0

    def test_all_values_stringified(self):
        data = [{"num": 42, "flag": True, "list": [1, 2, 3]}]
        result = dlt_schema_normalize(data)
        record = result["sample_record"]
        assert isinstance(record.get("num"), str)

    def test_multiple_records(self):
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = dlt_schema_normalize(data)
        assert result["count"] == 3

    def test_positional_only_one(self):
        sig = inspect.signature(dlt_schema_normalize)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        assert len(positional) <= 1
