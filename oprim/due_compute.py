"""FSRS 到期判定能力

FSRS 状态 → 到期判定
"""

from datetime import datetime, timezone
from typing import Optional

def due_compute(*, card_dict: dict, now: Optional[datetime] = None) -> bool:
    """计算 FSRS Card 是否已到期。

    Parameters
    ----------
    card_dict : dict
        FSRS Card 的字典表示。
    now : datetime | None
        当前时间（UTC），默认使用当前时间。
        
    Returns
    -------
    bool
        如果卡片已到期（due <= now）或未设置复习时间（新卡片），则返回 True，否则返回 False。
    """
    due_iso = card_dict.get("due")
    if not due_iso:
        # 新卡片或无 due 的情况，算见到期或立即可复习
        return True
        
    try:
        if hasattr(due_iso, "isoformat"):
            due_dt = due_iso
        else:
            # 兼容 python 3.11 以前如果 Z 不带冒号可能出错，但这里假定标准 ISO
            due_dt = datetime.fromisoformat(due_iso.replace('Z', '+00:00'))
    except ValueError:
        return True

    now = now or datetime.now(timezone.utc)
    return now >= due_dt
