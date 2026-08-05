"""oprim.task_state — transactional task state machine (pure logic).

3O layer: oprim (single atomic state transition, pure logic, no I/O).
Step-based task lifecycle: PENDING -> READY -> SUCCESS / FAILED / PAUSED,
with Time-Travel rollback slices. Persistence is the caller's concern
(obase.task_store); this module only computes state transitions.
"""

from __future__ import annotations

# Task lifecycle states
PENDING = "PENDING"
READY = "READY"
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
FAILED = "FAILED"
PAUSED = "PAUSED"

VALID_STATES = (PENDING, READY, RUNNING, SUCCESS, FAILED, PAUSED)

# Allowed transitions (source -> targets)
TRANSITIONS: dict[str, set[str]] = {
    PENDING: {READY, RUNNING, FAILED},
    READY: {RUNNING, FAILED},
    RUNNING: {SUCCESS, FAILED, PAUSED},
    PAUSED: {RUNNING, FAILED},
    SUCCESS: set(),
    FAILED: set(),
}


def validate_state(state: str) -> str:
    """Normalize + validate a state string; raises ValueError on unknown."""
    s = str(state).upper()
    if s not in VALID_STATES:
        raise ValueError(f"unknown task state: {state!r}")
    return s


def can_transition(current: str, target: str) -> bool:
    """Whether current -> target is a legal state-machine transition."""
    return target in TRANSITIONS.get(validate_state(current), set())


def build_steps(total_steps: int, initial_payload: dict | None = None) -> list[dict]:
    """Build the initial steps array: first step READY with initial payload."""
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    steps = [
        {"step": i, "status": PENDING, "payload": {}}
        for i in range(total_steps)
    ]
    steps[0]["status"] = READY
    steps[0]["payload"] = dict(initial_payload or {})
    return steps


def advance_step(
    steps: list[dict],
    step_index: int,
    step_payload: dict,
    total_steps: int,
) -> tuple[list[dict], int, bool]:
    """Mark a step SUCCESS, arm the next one; returns (steps, next_index, done)."""
    if not (0 <= step_index < len(steps)):
        raise IndexError(f"step_index {step_index} out of range")
    steps = [dict(s) for s in steps]
    steps[step_index]["status"] = SUCCESS
    steps[step_index]["payload"] = dict(step_payload)

    next_index = step_index + 1
    done = next_index >= total_steps
    if not done:
        steps[next_index]["status"] = READY
    return steps, next_index, done


def rollback_to(steps: list[dict], target_step: int) -> list[dict]:
    """Time-Travel: reset all steps AFTER target_step back to PENDING.

    The target step itself keeps its payload (resume point) and is marked
    READY so execution can be replayed from there.
    """
    if not (0 <= target_step < len(steps)):
        raise IndexError(f"target_step {target_step} out of range")
    rolled = [dict(s) for s in steps]
    for s in rolled[target_step + 1 :]:
        s["status"] = PENDING
        s["payload"] = {}
    rolled[target_step]["status"] = READY
    return rolled


def get_checkpoint_chain(steps: list[dict]) -> list[dict]:
    """Payload snapshot chain of completed steps (Time-Travel history)."""
    return [dict(s) for s in steps if s["status"] == SUCCESS]


def summary(steps: list[dict]) -> dict:
    """Compact step summary (for logs / UI)."""
    return {
        "total": len(steps),
        "completed": sum(1 for s in steps if s["status"] == SUCCESS),
        "current": next((s["step"] for s in steps if s["status"] == READY), None),
    }
