"""oprim._layering — declarative layer contract for intra-package dependencies.

Why this module exists
----------------------
A module's *filename* says nothing about what it is. `_veo3_generate.py` starts
with an underscore, so the old `check_no_sibling_call.py` treated it as "not an
element" and skipped it — while `veo3_generate` sat in `oprim.__all__` as a
public capability and was called by sibling `video_generate`. That produced a
false green on a rule that was supposed to be load-bearing.

So layers are declared by the module itself, not inferred from its name:

    __oprim_layer__ = "element"   # default — a public OPrim
    __oprim_layer__ = "infra"     # shared private base; not a capability
    __oprim_layer__ = "provider"  # external/provider adapter; not a capability

Allowed dependency direction (SPEC §10):

    element  →  infra
    element  →  provider
    provider →  infra
    infra    →  (nothing intra-package except other infra)

Forbidden: element → element. That is a sibling OPrim call.

Public facade re-exports are not sibling calls
---------------------------------------------
The package convention is a public module wrapping a private implementation:
``video_generate.py`` re-exports ``_video_generate.video_generate``. That is one
capability, declared twice, and stays legal. Only a call from capability A's
implementation into capability B's implementation is forbidden.

Two orthogonal concerns, deliberately not conflated
--------------------------------------------------
* **Layer** governs dependency *direction* only.
* **Export visibility** (``__all__``) governs the *public API surface* and is
  backward-compatibility-governed. An ``infra`` module may still publish types
  through ``__oprim_exports__`` — shared base modules like ``_media_types`` are
  infra for dependency purposes but their dataclasses are public API. Provider
  modules likewise stay exported because known consumers do e.g.
  ``from oprim import veo3_generate``.
"""

from __future__ import annotations

__oprim_layer__ = "infra"

import ast
from pathlib import Path

#: Module-level dunder that declares a module's layer.
LAYER_ATTR = "__oprim_layer__"

#: Module-level dunder listing names an ``infra`` module still publishes.
EXPORTS_ATTR = "__oprim_exports__"

LAYER_ELEMENT = "element"
LAYER_INFRA = "infra"
LAYER_PROVIDER = "provider"

LAYERS: frozenset[str] = frozenset({LAYER_ELEMENT, LAYER_INFRA, LAYER_PROVIDER})

#: Layers whose modules must never be imported by an element module.
CALLABLE_LAYERS: frozenset[str] = frozenset({LAYER_ELEMENT})

def module_layer(source: str, *, filename: str = "<unknown>") -> str:
    """Return the declared layer of a module's source, defaulting to `element`.

    A module that declares an unknown layer is a hard error: silently treating it
    as `element` would re-open the blind spot this module exists to close.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {filename}: {exc}") from exc
    return layer_from_tree(tree)

def layer_from_tree(tree: ast.Module) -> str:
    """Read `__oprim_layer__` out of an already-parsed module."""
    for node in tree.body:
        target: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == LAYER_ATTR for t in node.targets
            ):
                target = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == LAYER_ATTR
        ):
            target = node.value
        if target is None:
            continue
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            layer = target.value
            if layer not in LAYERS:
                raise ValueError(
                    f"{LAYER_ATTR}={layer!r} is not one of {sorted(LAYERS)}"
                )
            return layer
        raise ValueError(f"{LAYER_ATTR} must be a plain string literal")
    return LAYER_ELEMENT

def declared_exports(tree: ast.Module) -> frozenset[str]:
    """Read `__oprim_exports__` — names an infra module still publishes."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == EXPORTS_ATTR for t in node.targets):
            continue
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple)):
            raise ValueError(f"{EXPORTS_ATTR} must be a literal list/tuple of names")
        names: list[str] = []
        for item in value.elts:
            if not (isinstance(item, ast.Constant) and isinstance(item.value, str)):
                raise ValueError(f"{EXPORTS_ATTR} entries must be plain string literals")
            names.append(item.value)
        return frozenset(names)
    return frozenset()

def layer_of_path(path: Path) -> str:
    """Return the declared layer of a module on disk."""
    return module_layer(path.read_text(encoding="utf-8"), filename=str(path))
