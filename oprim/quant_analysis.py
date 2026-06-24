"""oprim.quant_analysis — 量化分析工具门面模块。

提供 compute_shapley_decomposition 等量化分析原子函数。
Helios 通过 oprim.quant_analysis.compute_shapley_decomposition 调用。
"""
from oprim._quant_analysis import compute_shapley_decomposition

__all__ = ["compute_shapley_decomposition"]
