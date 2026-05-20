"""Three-state circuit breaker for external tool clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from oprim._logging import log


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout_sec: int = 60
    success_threshold_to_close: int = 2

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _consecutive_successes_half_open: int = field(default=0, init=False, repr=False)
    _opened_at: datetime | None = field(default=None, init=False, repr=False)

    def call_allowed(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self._opened_at and datetime.utcnow() - self._opened_at > timedelta(
                seconds=self.recovery_timeout_sec
            ):
                self._transition_to_half_open()
                return True
            return False
        # HALF_OPEN
        return True

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self._consecutive_successes_half_open += 1
            if self._consecutive_successes_half_open >= self.success_threshold_to_close:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        if self.state == CircuitState.CLOSED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        log.warning(
            "circuit_breaker_open",
            name=self.name,
            failures=self._consecutive_failures,
        )
        self.state = CircuitState.OPEN
        self._opened_at = datetime.utcnow()
        self._consecutive_failures = 0
        self._consecutive_successes_half_open = 0

    def _transition_to_half_open(self) -> None:
        log.info("circuit_breaker_half_open", name=self.name)
        self.state = CircuitState.HALF_OPEN
        self._consecutive_successes_half_open = 0

    def _transition_to_closed(self) -> None:
        log.info("circuit_breaker_closed", name=self.name)
        self.state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes_half_open = 0
        self._opened_at = None
