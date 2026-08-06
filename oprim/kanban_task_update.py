"""oprim.kanban_task_update — ClawTeam auto-unblocking kanban (dependency chain wake-up).

When a task is marked ``completed``, this automatically scans for tasks that
were ``blocked_by`` it and transitions them to ``pending`` (wake-up).  Also
supports priority-based task re-ordering for the ready queue.

3O element: ``oprim.kanban_task_update``.
"""

from __future__ import annotations

from typing import Any


def kanban_task_update(
    task_id: str,
    new_status: str,
    tasks: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Update a task status and auto-unblock dependent tasks.

    Args:
        task_id: The task being updated.
        new_status: New status (``completed`` / ``in_progress`` / ``blocked`` / ``pending``).
        tasks: Full task list (mutated in-place for the response).
        context: Optional config.

    Returns:
        {status, updated_task, unblocked: [{task_id, subject}], ready_queue: [...]}
    """
    ctx = context or {}
    updated: dict[str, Any] | None = None
    for t in tasks:
        if t.get("id") == task_id:
            t["status"] = new_status
            updated = t
            break
    if updated is None:
        return {"status": "failed", "error": f"task {task_id} not found", "unblocked": []}

    # auto-unblock: any task that had this one in blocked_by
    unblocked: list[dict[str, Any]] = []
    if new_status == "completed":
        for t in tasks:
            blocked = list(t.get("blocked_by", []))
            if task_id in blocked:
                blocked.remove(task_id)
                t["blocked_by"] = blocked
                if not blocked and t.get("status") == "blocked":
                    t["status"] = "pending"
                    unblocked.append({"task_id": t.get("id"), "subject": t.get("subject", "")})

    # ready queue: pending tasks sorted by priority
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    ready = sorted(
        [t for t in tasks if t.get("status") == "pending" and not t.get("blocked_by") and not t.get("locked_by")],
        key=lambda t: priority_order.get(str(t.get("priority", "medium")), 2),
    )

    return {"status": "completed", "updated_task": updated, "unblocked": unblocked, "ready_queue": [r.get("id") for r in ready[:10]], "ready_count": len(ready)}
