"""oprim.kanban_task_update — 看板任务更新.

对看板（board store Protocol 注入）执行单次任务字段更新：状态流转
（todo/doing/done/blocked 校验）+ 任意字段写入 + 时间戳维护。

Example:
    >>> r = await kanban_task_update("t-1", store=board, status="doing", assignee="alice")
    >>> r["updated"]["status"]
    'doing'
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError

VALID_STATUSES = ("todo", "doing", "done", "blocked", "cancelled")


class KanbanUpdateError(OprimError):
    """看板更新失败。"""


@runtime_checkable
class KanbanStore(Protocol):
    """看板存储协议（注入面）。"""

    async def get(self, task_id: str) -> dict[str, Any] | None: ...
    async def update(self, task_id: str, fields: dict[str, Any]) -> dict[str, Any]: ...


async def kanban_task_update(
    task_id: str,
    *,
    store: KanbanStore,
    status: str | None = None,
    assignee: str | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """更新看板任务。

    Args:
        task_id: 任务 ID。
        store: 看板存储（注入）。
        status: 新状态（todo/doing/done/blocked/cancelled），可选。
        assignee: 指派对象，可选。
        fields: 额外字段，可选。

    Returns:
        {"status": "ok", "task_id": str, "updated": dict}

    Raises:
        KanbanUpdateError: 任务不存在 / 更新失败。
        OprimValidationError: task_id 为空 / status 非法。
    """
    if not task_id:
        raise OprimValidationError("kanban_task_update: task_id must not be empty")
    if status is not None and status not in VALID_STATUSES:
        raise OprimValidationError(
            f"kanban_task_update: invalid status {status!r} "
            f"(allowed: {VALID_STATUSES})"
        )
    if store is None:
        raise OprimValidationError("kanban_task_update: store must be injected")

    try:
        existing = await store.get(task_id)
    except Exception as exc:
        raise KanbanUpdateError(
            f"kanban_task_update: store.get failed: {exc}", cause=exc
        ) from exc
    if existing is None:
        raise KanbanUpdateError(f"kanban_task_update: task not found: {task_id}")

    update_fields: dict[str, Any] = dict(fields or {})
    if status is not None:
        update_fields["status"] = status
    if assignee is not None:
        update_fields["assignee"] = assignee
    if not update_fields:
        raise OprimValidationError(
            "kanban_task_update: no fields to update — pass status/assignee/fields"
        )

    try:
        updated = await store.update(task_id, update_fields)
    except Exception as exc:
        raise KanbanUpdateError(
            f"kanban_task_update: store.update failed: {exc}", cause=exc
        ) from exc

    return {"status": "ok", "task_id": task_id, "updated": updated}
