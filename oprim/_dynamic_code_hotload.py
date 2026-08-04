"""oprim.dynamic_code_hotload — Python 动态代码编译、内存加载与热挂载.

单次将 LLM 自主生成的 Python 代码编译并热加载入当前 Python 进程；
临时文件用后即删。

Example:
    >>> r = dynamic_code_hotload("def add(a, b):\n    return a + b", module_name="dyn_add")
    >>> r["status"]
    'success'
    >>> r["exported_functions"]
    ['add']
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from typing import Any

from oprim._exceptions import OprimValidationError


def dynamic_code_hotload(
    python_code: str,
    *,
    module_name: str = "dynamic_skill",
) -> dict[str, Any]:
    """热加载 Python 代码。

    Args:
        python_code: 待加载的 Python 源码。
        module_name: 目标模块名（sys.modules 键）。

    Returns:
        {"status": "success", "module_name": str, "exported_functions": [str]}
        失败时 status="failed" + error（不 raise）。

    Raises:
        OprimValidationError: python_code 为空 / module_name 非法。
    """
    if not python_code or not python_code.strip():
        raise OprimValidationError("dynamic_code_hotload: python_code must not be empty")
    if not module_name or not module_name.strip():
        raise OprimValidationError("dynamic_code_hotload: module_name must not be empty")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        if spec is None or spec.loader is None:
            return {"status": "failed", "error": "Failed to create module spec."}

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        exported_funcs = [
            attr
            for attr in dir(mod)
            if callable(getattr(mod, attr)) and not attr.startswith("_")
        ]
        return {
            "status": "success",
            "module_name": module_name,
            "exported_functions": exported_funcs,
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if tmp_path is not None:
            import contextlib

            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
