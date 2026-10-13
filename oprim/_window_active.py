"""oprim._window_active — 维护/静默/冻结窗口命中判定 (aegis DESIGN §3.2/§5.3/§9).

纯计算:给定窗口定义与 now,判断 now 是否落在活跃窗口内。维护窗口、静默抑制、变更冻结、
演练调度共用此判定。now/start 由调用方提供(须同为 tz-aware 或同为 naive)。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from oprim._exceptions import OprimValidationError

_DAY = 86400

Recurrence = Literal["none", "daily", "weekly"]


class WindowStatus(BaseModel):
    active: bool
    recurrence: str


def _seconds_of_day(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def _in_time_of_day(now_sod: int, start_sod: int, duration_seconds: int) -> bool:
    """时段命中(支持跨午夜:end 超过一天则回绕)。"""
    end = start_sod + duration_seconds
    if end <= _DAY:
        return start_sod <= now_sod < end
    return now_sod >= start_sod or now_sod < (end - _DAY)


def window_active_check(
    *,
    now: datetime,
    start: datetime,
    duration_seconds: int,
    recurrence: Recurrence = "none",
    weekdays: list[int] | None = None,
) -> WindowStatus:
    """判断 now 是否落在活跃窗口内。

    Args:
        now: 当前时刻。
        start: 窗口起点。none=绝对起点;daily/weekly 仅取其时刻(与 weekday)。
        duration_seconds: 窗口时长秒。
        recurrence: "none"(一次性 [start, start+dur))、"daily"(每日同时段,支持跨午夜)、
            "weekly"(指定 weekday 的同时段,不得跨午夜)。
        weekdays: weekly 生效星期(0=周一..6=周日);None 时取 start.weekday()。

    Returns:
        WindowStatus(active, recurrence)。

    Raises:
        OprimValidationError: duration<=0、未知 recurrence、或 weekly 窗口跨午夜。
    """
    if duration_seconds <= 0:
        raise OprimValidationError(f"duration_seconds must be > 0, got {duration_seconds}")

    if recurrence == "none":
        active = start <= now < start + timedelta(seconds=duration_seconds)
    elif recurrence == "daily":
        active = _in_time_of_day(_seconds_of_day(now), _seconds_of_day(start), duration_seconds)
    elif recurrence == "weekly":
        if _seconds_of_day(start) + duration_seconds > _DAY:
            raise OprimValidationError(
                "weekly window must not cross midnight; use daily or split into two windows"
            )
        days = weekdays if weekdays is not None else [start.weekday()]
        active = now.weekday() in days and _in_time_of_day(
            _seconds_of_day(now), _seconds_of_day(start), duration_seconds
        )
    else:
        raise OprimValidationError(f"unknown recurrence {recurrence!r}")

    return WindowStatus(active=active, recurrence=recurrence)
