"""状态读取原子操作 (3O 范式全栈 — oprim 元素 10).

单次原子 IO: 从持久化存储读取状态, 缺失即抛 KeyError (由上层裁决语义)。
"""

from __future__ import annotations

from obase import KVStoreProvider


async def fetch_state(
    key: str,
    *,
    store: KVStoreProvider,
) -> dict:
    """单次原子 IO: 从持久化存储读取状态。

    Args:
        key: 存储键。
        store: 键值存储契约实现 (依赖注入)。

    Raises:
        KeyError: 状态不存在。
    """
    data = await store.get(key)
    if not data:
        raise KeyError(f"State {key} not found")
    return data
