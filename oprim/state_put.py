"""状态持久化原子操作 (3O 范式 Phase 2 — oprim 元实现层).

单次原子 IO: 写入持久化存储。
不感知"会话树 (Session Tree)"等业务概念, 只认知 key 和 value。
"""

from __future__ import annotations

from obase import KVStoreProvider


async def put_state(
    key: str,  # ≤1个核心位置参数
    *,  # 其余强制 keyword-only
    value: dict,
    store: KVStoreProvider,
) -> None:
    """单次原子 IO: 写入持久化存储。

    Args:
        key: 存储键。
        value: 存储值 (JSON 可序列化字典)。
        store: 键值存储契约实现 (依赖注入)。
    """
    await store.put(key, value)
