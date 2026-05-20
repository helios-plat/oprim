"""Tests for push protocol dataclasses."""
from __future__ import annotations

from oprim.push.protocol import PushResult


class TestPushResult:
    def test_success_fields(self):
        r = PushResult(channel="web", success=True, recipient="user@example.com")
        assert r.channel == "web"
        assert r.success is True
        assert r.error_message is None
        assert r.sent_at is not None

    def test_failure_fields(self):
        r = PushResult(channel="email", success=False, recipient="x", error_message="timeout")
        assert r.success is False
        assert r.error_message == "timeout"
