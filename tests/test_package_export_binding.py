"""Exported callables must win over same-named submodules, regardless of import order."""

from __future__ import annotations

import importlib
import importlib.util
import types

import oprim


def _collisions(same_module: bool) -> list[str]:
    out = []
    for name, target in oprim._ELEMENT_MAP.items():
        sub = f"oprim.{name}"
        if name not in oprim._SUBMODULE_SET or (target == sub) != same_module:
            continue
        try:
            importlib.util.find_spec(sub)
        except Exception:
            continue
        if importlib.util.find_spec(sub) is not None:
            out.append(name)
    return sorted(out)


def test_submodule_import_does_not_shadow_export_same_module() -> None:
    names = _collisions(same_module=True)
    assert names, "expected at least one oprim/<name>.py exporting <name>"
    name = names[0]
    mod = importlib.import_module(f"oprim.{name}")
    exported = getattr(oprim, name)
    assert not isinstance(exported, types.ModuleType)
    assert exported is getattr(mod, name)


def test_submodule_import_does_not_shadow_export_other_module() -> None:
    import oprim.apply_screen_filter  # noqa: F401  (submodule with the same name)

    assert oprim._ELEMENT_MAP["apply_screen_filter"] != "oprim.apply_screen_filter"
    assert callable(oprim.apply_screen_filter)
    assert not isinstance(oprim.apply_screen_filter, types.ModuleType)
    # the submodule itself stays reachable
    assert isinstance(importlib.import_module("oprim.apply_screen_filter"), types.ModuleType)


def test_plain_attribute_assignment_still_works() -> None:
    oprim._test_marker_attr = 1  # type: ignore[attr-defined]
    try:
        assert oprim._test_marker_attr == 1  # type: ignore[attr-defined]
    finally:
        del oprim._test_marker_attr  # type: ignore[attr-defined]
