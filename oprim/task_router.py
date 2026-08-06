"""oprim.task_router — ClawTeam-style priority task router.

Routes team messages to dispatch decisions based on priority, due scheduling,
and the receiving agent's current workload.  Deterministic — no LLM involved.

3O element: ``oprim.task_router`` (``route_tasks`` / ``dispatch_decision``).
"""

from __future__ import annotations

from typing import Any


def route_tasks(
    tasks: list[dict[str, Any]],
    members: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Assign pending/unassigned tasks to available team members.

    Priority order: urgent → high → medium → low.  Tasks with ``blocked_by``
    are skipped (waiting on dependencies).  The least-busy idle member gets
    each task.

    Returns:
        [{task_id, assigned_to, task, decision}, ...]
    """
    ctx = context or {}
    priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}

    # filter routable tasks
    routable = [
        t for t in tasks
        if t.get("status") in ("pending", "in_progress")
        and not t.get("locked_by")
        and not t.get("blocked_by")
    ]
    routable.sort(key=lambda t: priority_order.get(str(t.get("priority", "medium")), 2))

    idle_members = [m for m in members if m.get("agent_type") != "observer"]
    if not idle_members:
        idle_members = members

    # assign round-robin to least loaded
    loads: dict[str, int] = {m["name"]: 0 for m in idle_members}
    for t in tasks:
        owner = t.get("locked_by") or t.get("owner")
        if owner and owner in loads:
            loads[owner] += 1

    decisions: list[dict[str, Any]] = []
    for task in routable:
        best = min(loads, key=lambda k: loads[k])
        loads[best] += 1
        decisions.append({
            "task_id": task.get("id", "?"),
            "assigned_to": best,
            "priority": task.get("priority", "medium"),
            "task": task,
            "decision": "assign",
        })
    return decisions


def dispatch_decision(
    decisions: list[dict[str, Any]],
    message_type: str | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute dispatch: for each decision, produce a message to inject.

    Returns {status, dispatched: [{agent, task_id, message}], pending: [...]}
    """
    dispatched: list[dict[str, Any]] = []
    for d in decisions:
        if d.get("decision") == "assign":
            dispatched.append({
                "agent": d["assigned_to"],
                "task_id": d["task_id"],
                "message": {
                    "type": message_type or "message",
                    "to": d["assigned_to"],
                    "content": f"New task assigned: {d['task'].get('subject', d['task_id'])}",
                },
            })
    return {"status": "completed", "dispatched": dispatched, "pending": []}
