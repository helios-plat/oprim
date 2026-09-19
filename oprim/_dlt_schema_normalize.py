"""oprim.dlt_schema_normalize — 单次将半结构化 JSON 转化为标准 Schema 数据表.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: 纯计算.

例:
    >>> data = [{"Name": "Alice", "Age-Group": "adult"}]
    >>> result = dlt_schema_normalize(data)
    >>> result["status"]
    'success'
    >>> result["count"]
    1
"""

from __future__ import annotations

import contextlib
from typing import Any, Protocol


class PipelineStoreProtocol(Protocol):
    """DLT 管道存储 Protocol — 不 import obase，由调用方注入."""

    def auto_infer_and_load(
        self, table_name: str, records: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


def dlt_schema_normalize(
    raw_json_data: list[dict[str, Any]],
    *,
    pipeline_op: PipelineStoreProtocol | None = None,
) -> dict[str, Any]:
    """单次将半结构化 JSON 转化为 dlt 关系型 Schema 数据表 (dlt 机制).

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    处理:
    1. 字段名规范化：小写 + 下划线替换连字符/空格
    2. 值统一转为字符串（保证 Schema 一致性）
    3. 可选注入 pipeline_op 自动入库

    Args:
        raw_json_data: 半结构化 JSON 记录列表
        pipeline_op: DltPipelineStore Protocol 实例（可选）

    Returns:
        {
            "status": "success" | "empty",
            "count": int,
            "normalized_records": list[dict],
            "sample_record": dict,
        }
    """
    if not raw_json_data:
        return {"status": "empty", "normalized_records": [], "count": 0, "sample_record": {}}

    # 规范化每条记录
    normalized: list[dict[str, Any]] = []
    for item in raw_json_data:
        clean_item: dict[str, Any] = {}
        for key, value in item.items():
            # 字段名：小写 + 下划线替换非法字符
            clean_key = key.lower()
            clean_key = clean_key.replace("-", "_").replace(" ", "_")
            clean_key = "".join(c for c in clean_key if c.isalnum() or c == "_")
            # 值：统一字符串化
            if isinstance(value, (dict, list)):
                import json

                clean_item[clean_key] = json.dumps(value, ensure_ascii=False)
            else:
                clean_item[clean_key] = str(value)
        normalized.append(clean_item)

    # 可选入库
    if pipeline_op is not None:
        with contextlib.suppress(Exception):
            pipeline_op.auto_infer_and_load("normalized_table", normalized)

    return {
        "status": "success",
        "count": len(normalized),
        "normalized_records": normalized,
        "sample_record": normalized[0] if normalized else {},
    }
