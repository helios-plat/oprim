"""Oprim — atomic operations library (Layer 1 meta-primitives). Lazy-loaded.

3O layer: oprim (Layer 1).
PEP 562 lazy loading to prevent heavy dependency leakage.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from oprim._version import __version__

_ELEMENT_MAP: dict[str, str] = {}  # Element name -> submodule path
_SUBMODULE_SET: set[str] = set()   # Valid submodule names (stems)


def _build_element_map() -> None:
    pkg_dir = Path(__file__).parent
    pkg_name = __package__ or "oprim"
    
    # Recursive glob to find all .py files in subpackages
    for py in sorted(pkg_dir.rglob("*.py")):
        # Get relative path and convert to module path
        rel_path = py.relative_to(pkg_dir)
        
        # Don't scan the top-level __init__.py
        if rel_path.parts == ("__init__.py",):
            continue
            
        # Convert path to module parts: ('timeseries', 'stationarity') or ('technical', '__init__')
        mod_parts = list(rel_path.with_suffix("").parts)
        
        if mod_parts[-1] == "__init__":
            mod_parts.pop()  # Use package path instead of package.__init__
            if not mod_parts: continue # Should not happen given skip above
        
        mod_path = pkg_name + "." + ".".join(mod_parts)
        stem = mod_parts[-1]
        
        _SUBMODULE_SET.add(stem)
        
        try:
            # Static analysis to avoid importing the module
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in tree.body:
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.append(target.id)
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    names.append(elt.id)
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        names.append(node.target.id)
                elif isinstance(node, ast.ImportFrom):
                    # For __init__.py files that re-export from submodules
                    if rel_path.name == "__init__.py":
                        for alias in node.names:
                            if alias.name != "*":
                                names.append(alias.asname or alias.name)

                for name in names:
                    if not name.startswith("_"):
                        # Heuristic: if current is public and previous was private, overwrite
                        if name not in _ELEMENT_MAP or (
                            not mod_path.split(".")[-1].startswith("_") and _ELEMENT_MAP[name].split(".")[-1].startswith("_")
                        ):
                            _ELEMENT_MAP[name] = mod_path
        except Exception:
            continue


_build_element_map()


def __getattr__(name: str) -> Any:
    if name == "__version__":
        return __version__
    if name in _ELEMENT_MAP:
        mod = importlib.import_module(_ELEMENT_MAP[name])
        return getattr(mod, name)
    if name in _SUBMODULE_SET:
        pkg_name = __package__ or "oprim"
        # We need to find the full mod_path for this stem in SUBMODULE_SET?
        # Actually, if we just do pkg_name.name it might work for top-level.
        return importlib.import_module(f"{pkg_name}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(_ELEMENT_MAP.keys()) + list(_SUBMODULE_SET) + ["__version__"]))


__all__ = sorted(_ELEMENT_MAP.keys())
