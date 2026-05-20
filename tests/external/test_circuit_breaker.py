"""Tests for oprim.external.circuit_breaker."""
from __future__ import annotations

import time

import pytest

from oprim.external.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerTransitions:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.call_allowed()

    def test_closed_to_open_after_threshold(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # still closed
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.call_allowed()

    def test_open_blocks_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_sec=9999)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.call_allowed()

    def test_open_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_sec=0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.01)
        assert cb.call_allowed()
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_failure_returns_to_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_sec=0)
        cb.record_failure()  # → OPEN
        time.sleep(0.01)
        cb.call_allowed()    # → HALF_OPEN
        cb.record_failure()  # → OPEN again
        assert cb.state == CircuitState.OPEN

    def test_half_open_success_closes_after_threshold(self):
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout_sec=0,
            success_threshold_to_close=2,
        )
        cb.record_failure()  # → OPEN
        time.sleep(0.01)
        cb.call_allowed()    # → HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # need 2 successes
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_success_in_closed_resets_failure_counter(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets counter
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # needs 1 more

    def test_full_cycle(self):
        cb = CircuitBreaker(
            name="full_cycle",
            failure_threshold=2,
            recovery_timeout_sec=0,
            success_threshold_to_close=1,
        )
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.01)
        assert cb.call_allowed()
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
