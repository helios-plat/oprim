"""oprim.time_series — 时间序列工具门面模块。

暴露 _time_series.py 的公开原子函数。消费方通过
oprim.time_series.percentile_rank 等路径调用。
"""

from oprim._time_series import (
    cumulative_returns,
    ewma_smooth,
    gap_detect,
    lag_forward_fill,
    log_returns,
    percentile_rank,
    purge_embargo_split,
    realized_vol,
    resample_align,
    rolling_window_split,
    zscore_normalize,
)

__all__ = [
    "log_returns",
    "cumulative_returns",
    "rolling_window_split",
    "lag_forward_fill",
    "percentile_rank",
    "ewma_smooth",
    "realized_vol",
    "zscore_normalize",
    "gap_detect",
    "resample_align",
    "purge_embargo_split",
]
