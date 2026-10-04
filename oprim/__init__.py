"""Oprim — atomic operations library (Layer 1 meta-primitives). Lazy-loaded."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from oprim._layering import LAYER_ELEMENT, LAYER_INFRA, declared_exports, layer_from_tree
from oprim._version import __version__

_ELEMENT_MAP: dict[str, str] = {}
_SUBMODULE_SET: set[str] = set()
#: module path → declared layer (from `__oprim_layer__`). Authoritative for
#: dependency-direction checks; see `oprim._layering`.
_MODULE_LAYERS: dict[str, str] = {}


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
            layer = layer_from_tree(tree)
            _MODULE_LAYERS[mod_path] = layer
            # Infra modules are shared bases, not capabilities. Their unlisted
            # helpers stay reachable by direct module path
            # (`from oprim._ffprobe import probe_json`) but never enter the
            # element namespace. `__oprim_exports__` re-publishes the names that
            # are genuinely public API (e.g. the dataclasses in `_media_types`).
            #
            # Provider modules always export: their layer governs dependency
            # direction only, while the export surface is backward-compatibility
            # governed and known consumers do `from oprim import veo3_generate`.
            allowed: set[str] | None = None
            if layer == LAYER_INFRA:
                allowed = set(declared_exports(tree))
            for node in tree.body:
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.ImportFrom) and rel_path.name == "__init__.py":
                    for alias in node.names:
                        if alias.name != "*":
                            names.append(alias.asname or alias.name)
                for name in names:
                    if name.startswith("_"):
                        continue
                    if allowed is not None and name not in allowed:
                        continue
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


# ---------------------------------------------------------------------------
# Release manifest (consumed by HEVI's `test_three_o_contracts` and
# `scripts/ci/run_3o_v3_manifest_audit.py`).
#
# Derived from `_ELEMENT_MAP`, never hand-written. The previous hardcoded
# manifest listed 5 entries with signatures that did not match the real
# functions (`(request, /, *, provider, output_path)` — there is no `request`
# parameter), so it was a second, wrong source of truth of exactly the kind
# `_manifest.py` was. `_build_element_map` is the only authority now.
#
# `signature` is computed lazily by `element_signature()` rather than eagerly:
# introspecting every element would import the whole package at import time.
# The audit script treats `signature` as optional (`entry.get("signature")`).
# ---------------------------------------------------------------------------

#: Elements the manifest publishes. Keep this list to the canonical surface a
#: release is contracted on — not every one of the 1400+ discovered names.
_MANIFEST_ELEMENTS: tuple[str, ...] = (
    # media inspection
    "media_probe",
    "probe_duration",
    # transcription
    "transcribe_audio",
    # frames / waveform / segmentation
    "extract_video_frames",
    "extract_audio_waveform",
    "segment_media",
    "extract_media_segment",
    # render / mix / burn
    "render_html_to_mp4",
    "render_media",
    "audio_mix",
    "audio_normalize",
    "audio_video_merge",
    "subtitle_burn",
    # generation
    "video_generate",
    "tts_synthesize",
    "avatar_generate",
    "encode_voice_reference",
    # origin's 5d992f2 manifest also lists edge_tts_synthesize
    "edge_tts_synthesize",
    # safety / prompt
    "validate_html",
    "style_marker_prompt",
    # consumer-facing types (HEVI imports these from `oprim.hevi_types`;
    # the module itself is a namespace, not a resolvable entity, so the
    # manifest names the types rather than the module)
    "CanvasEdge",
    "CanvasNode",
    "ProviderCapability",
    "Subject",
    "VideoQuality",
)


def element_signature(name: str) -> str | None:
    """Real signature of a manifest element, or None if it cannot be resolved.

    Imports the owning module on demand. Kept out of the eager manifest build
    so `import oprim` stays cheap.
    """
    import importlib
    import inspect

    module_path = _ELEMENT_MAP.get(name)
    if module_path is None:
        return None
    try:
        obj = getattr(importlib.import_module(module_path), name)
    except Exception:  # pragma: no cover - defensive: manifest must never break import
        return None
    try:
        return f"{name}{inspect.signature(obj)}"
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return None


def _source_path_for(module_path: str) -> Path | None:
    """File that defines `module_path`, whether it is a module or a subpackage."""
    rel = Path(*module_path.split(".")[1:])
    direct = Path(__file__).parent / rel.with_suffix(".py")
    if direct.exists():
        return direct
    pkg_init = Path(__file__).parent / rel / "__init__.py"
    return pkg_init if pkg_init.exists() else None


def _element_depends_on(name: str) -> list[str]:
    """Intra-package modules the element's implementation imports."""
    module_path = _ELEMENT_MAP.get(name)
    if module_path is None:
        return []
    source = _source_path_for(module_path)
    if source is None:
        return []
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - defensive
        return []
    deps: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.split(".")[0] == "oprim"
        ):
            deps.add(node.module)
    return sorted(deps)


def _module_path_for(name: str) -> str | None:
    """Resolve a manifest name to the module that defines it.

    Two shapes are legitimate. Most elements resolve through `_ELEMENT_MAP`.
    A few are pure module re-exports (`hevi_types`) that consumers import as a
    namespace rather than a callable; those resolve to the module itself.
    """
    via_element = _ELEMENT_MAP.get(name)
    if via_element is not None:
        return via_element
    if _source_path_for(f"{__name__}.{name}") is not None:
        return f"{__name__}.{name}"
    return None


def _build_manifest() -> dict[str, object]:
    elements: list[dict[str, object]] = []
    for name in _MANIFEST_ELEMENTS:
        module_path = _module_path_for(name)
        if module_path is None:
            # Never publish a name discovery cannot resolve: the audit script
            # flags such entries as dangling.
            continue
        elements.append(
            {
                "name": name,
                "kind": "oprim",
                "module": module_path,
                "layer": _MODULE_LAYERS.get(module_path, LAYER_ELEMENT),
                "depends_on": _element_depends_on(name),
                "pillars": [],
            }
        )
    return {"package": __name__, "version": __version__, "elements": elements}


__manifest__: dict[str, object] = _build_manifest()
