"""Oprim — atomic operations library (Layer 1 meta-primitives). Lazy-loaded."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from oprim._version import __version__

_ELEMENT_MAP: dict[str, str] = {}
_SUBMODULE_SET: set[str] = set()


def _build_element_map() -> None:
    pkg_dir = Path(__file__).parent
    pkg_name = __package__ or "oprim"
    for py in sorted(pkg_dir.rglob("*.py")):
        rel_path = py.relative_to(pkg_dir)
        if rel_path.parts == ("__init__.py",):
            continue
        mod_parts = list(rel_path.with_suffix("").parts)
        if mod_parts[-1] == "__init__":
            mod_parts.pop()
        if not mod_parts:
            continue
        mod_path = pkg_name + "." + ".".join(mod_parts)
        stem = mod_parts[-1]
        _SUBMODULE_SET.add(stem)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in tree.body:
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.ImportFrom) and rel_path.name == "__init__.py":
                    for alias in node.names:
                        if alias.name != "*":
                            names.append(alias.asname or alias.name)
                for name in names:
                    if not name.startswith("_"):
                        cond = name not in _ELEMENT_MAP or (
                            not mod_path.split(".")[-1].startswith("_")
                            and _ELEMENT_MAP[name].split(".")[-1].startswith("_")
                        )
                        if cond:
                            _ELEMENT_MAP[name] = mod_path
        except Exception:
            continue


_build_element_map()

# KCState 归 obase（经 oprim._cognitive 单源、惰性暴露）。登记到元素表以保持
# `from oprim import KCState`（oskill 兼容）与 __all__ 可达，但不在 import 时 eager-load obase：
# __getattr__ 命中后 getattr(_cognitive, "KCState") 触发其模块级 __getattr__ 才 import obase。
_ELEMENT_MAP["KCState"] = "oprim._cognitive"  # re-export for oskill compatibility

# 无前缀 BKT 别名（单源 oprim._cognitive，经 oprim.bkt 别名层暴露）。
# bkt.py 不是 __init__ 包文件，AST 元素扫描不收录其 ImportFrom 别名，故在此显式登记，
# 以保持 `from oprim import classify_error` 可达（M2 collection 契约）。
_ELEMENT_MAP["classify_error"] = "oprim.bkt"


# llm_summarize 惰性加载（依赖 obase，不在没有 obase 的环境 eager-load）
def llm_summarize(*args, **kwargs):
    """惰性加载 llm_summarize，调用时才 import obase 依赖。"""
    from oprim._llm_summarize import llm_summarize as _fn

    return _fn(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return __version__
    if name in _ELEMENT_MAP:
        mod = importlib.import_module(_ELEMENT_MAP[name])
        return getattr(mod, name)
    if name in _SUBMODULE_SET:
        pkg_name = __package__ or "oprim"
        return importlib.import_module(f"{pkg_name}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(_ELEMENT_MAP.keys()) + list(_SUBMODULE_SET) + ["__version__"]))


__all__ = sorted(_ELEMENT_MAP.keys())


# --- Lazy-load wrappers for obase-dependent functions ---
def llm_complete(*args, **kwargs):
    from oprim.llm._llm_complete import llm_complete as _fn

    return _fn(*args, **kwargs)


def llm_stream(*args, **kwargs):
    from oprim.llm._llm_stream import llm_stream as _fn

    return _fn(*args, **kwargs)


def embed_text(*args, **kwargs):
    from oprim.llm._embed_text import embed_text as _fn

    return _fn(*args, **kwargs)


def image_generate(*args, **kwargs):
    from oprim.image_generate import image_generate as _fn

    return _fn(*args, **kwargs)


def image_understand(*args, **kwargs):
    from oprim.image_understand import image_understand as _fn

    return _fn(*args, **kwargs)


def tts_synthesize(*args, **kwargs):
    from oprim.tts_synthesize import tts_synthesize as _fn

    return _fn(*args, **kwargs)


# --- File parsers + structure extractor (lazy-loaded) ---
def file_parser_pdf(*args, **kwargs):
    from oprim._file_parser_pdf import file_parser_pdf as _fn

    return _fn(*args, **kwargs)


def file_parser_epub(*args, **kwargs):
    from oprim._file_parser_epub import file_parser_epub as _fn

    return _fn(*args, **kwargs)


def file_parser_html(*args, **kwargs):
    from oprim._file_parser_html import file_parser_html as _fn

    return _fn(*args, **kwargs)


# from oprim._file_parser_markdown import file_parser_markdown as file_parser_markdown
from oprim._file_parser_plaintext import (  # noqa: E402,I001
    file_parser_plaintext as file_parser_plaintext,
)
from oprim._document_structure_extractor import (  # noqa: E402,I001
    document_structure_extractor as document_structure_extractor,
)


def epub_toc_split(*args, **kwargs):
    from oprim._epub_toc_split import epub_toc_split as _fn

    return _fn(*args, **kwargs)


def _get_epub_book():
    from oprim._epub_toc_split import EpubBook

    return EpubBook
