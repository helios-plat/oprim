"""Tests for validate_html."""
from __future__ import annotations

import pytest

from oprim._validate_html import validate_html


class TestValidateHtml:

    def test_clean_html_passes(self):
        html = "<div><p>Hello world</p><span class='note'>safe</span></div>"
        result = validate_html(html=html)
        assert result.is_safe is True
        assert result.violations == []
        assert result.sanitized is None

    def test_external_script_src_detected(self):
        html = '<html><head><script src="https://evil.com/payload.js"></script></head></html>'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "external_script_src" in result.violations

    def test_inline_event_handler_detected(self):
        html = '<img src="x.png" onerror="fetch(document.cookie)" />'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "inline_event_handler" in result.violations

    def test_eval_detected(self):
        html = '<script>eval(atob("YWxlcnQoMSk="))</script>'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "eval_usage" in result.violations

    def test_empty_input_is_safe(self):
        assert validate_html(html="").is_safe is True
        assert validate_html(html="   ").is_safe is True

    def test_javascript_uri_detected(self):
        html = '<a href="javascript:alert(document.cookie)">click</a>'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "javascript_uri" in result.violations

    def test_fetch_detected(self):
        html = "<script>fetch('/api/steal?c=' + document.cookie)</script>"
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "fetch_usage" in result.violations

    def test_xmlhttprequest_detected(self):
        html = "<script>var x = new XMLHttpRequest(); x.open('GET','/secret')</script>"
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "xmlhttprequest_usage" in result.violations

    def test_external_src_blocked_by_default(self):
        html = '<img src="https://tracker.example.com/pixel.png" />'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "external_src" in result.violations

    def test_external_src_allowed_when_flag_set(self):
        html = '<img src="https://cdn.example.com/logo.png" />'
        result = validate_html(html=html, allow_external_src=True)
        assert result.is_safe is True
        assert "external_src" not in result.violations

    def test_external_iframe_detected(self):
        html = '<iframe src="https://evil.com/phish"></iframe>'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert "external_iframe" in result.violations

    def test_multiple_violations_all_reported(self):
        html = '<div onclick="eval(fetch(\'x\'))">click</div>'
        result = validate_html(html=html)
        assert result.is_safe is False
        assert len(result.violations) >= 2
        assert "inline_event_handler" in result.violations
        assert "eval_usage" in result.violations
        assert "fetch_usage" in result.violations

    def test_sanitized_populated_on_violation(self):
        html = '<img onerror="steal()" src="local.png" />'
        result = validate_html(html=html)
        assert result.sanitized is not None
        # The dangerous attribute should be neutralised
        assert "onerror=" not in result.sanitized or "data-blocked" in result.sanitized

    def test_sanitized_none_when_safe(self):
        html = "<p>Safe paragraph</p>"
        result = validate_html(html=html)
        assert result.sanitized is None

    def test_onload_inline_event_detected(self):
        html = '<body onload="malicious()">'
        result = validate_html(html=html)
        assert "inline_event_handler" in result.violations
