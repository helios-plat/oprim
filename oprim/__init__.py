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
        if rel_path.parts == ("__init__.py",): continue
        mod_parts = list(rel_path.with_suffix("").parts)
        if mod_parts[-1] == "__init__": mod_parts.pop()
        if not mod_parts: continue
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
                        if alias.name != "*": names.append(alias.asname or alias.name)
                for name in names:
                    if not name.startswith("_"):
                        if name not in _ELEMENT_MAP or (
                            not mod_path.split(".")[-1].startswith("_") and _ELEMENT_MAP[name].split(".")[-1].startswith("_")
                        ):
                            _ELEMENT_MAP[name] = mod_path
        except Exception: continue

_build_element_map()

from oprim._cognitive import KCState  # re-export for oskill compatibility
from oprim._llm_summarize import llm_summarize as llm_summarize

def __getattr__(name: str) -> Any:
    if name == "__version__": return __version__
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

# --- Explicit re-exports (Pinning) ---
from oprim._exceptions import (
    OprimError, FileOprimError, GitOprimError, ShellOprimError,
    ParseOprimError, PathSecurityError, LLMOprimError, BudgetExceededError,
    PromptOprimError, SearchOprimError, HttpOprimError, SnapshotOprimError
)
from oprim.llm._types import (
    LLMResponse, StreamDelta, EmbedResult, ConversationSnapshot,
    ThinkingResult, SearchResult, HttpResponse
)
from oprim.llm._llm_complete import llm_complete
from oprim.llm._llm_stream import llm_stream
from oprim.llm._embed_text import embed_text
from oprim.prompt import (
    build_system_prompt, truncate_messages, extract_thinking, snapshot_conversation
)
from oprim.image_generate import image_generate
from oprim.image_understand import image_understand
from oprim.tts_synthesize import tts_synthesize

# --- Mneme elements (M-A batch) ---
from oprim.types import (
    SolveResult, SolveStep, StepCheckResult, Plot2DData, Three3DData,
    GradeResult, PeerPercentileResult
)
from oprim.compute_peer_percentile import compute_peer_percentile, compute_percentile_batch
from oprim.recognition_update import recognition_update, recognition_update_sequence
from oprim.compute_effortful_gain import compute_effortful_gain, compute_effortful_gain_from_arrays
from oprim.compute_feedback import compute_feedback, grade_answer
from oprim.file_type_detector import file_type_detector as file_type_detector

# File parsers + structure extractor (restored exports)
from oprim._file_parser_pdf import file_parser_pdf as file_parser_pdf
from oprim._file_parser_epub import file_parser_epub as file_parser_epub
from oprim._file_parser_html import file_parser_html as file_parser_html
from oprim._file_parser_markdown import file_parser_markdown as file_parser_markdown
from oprim._file_parser_plaintext import file_parser_plaintext as file_parser_plaintext
from oprim._document_structure_extractor import document_structure_extractor as document_structure_extractor
