"""Inject externally resolved asset references into a shot specification.

This is a pure, single-operation transformation.  Asset ownership and
resolution remain in Layer 4; the optional loader is an injected adapter.
"""
from __future__ import annotations


def asset_reference_inject(
    *,
    shot_spec: dict,
    asset_refs: dict,
    asset_loader,
) -> dict:
    """Return a copy of ``shot_spec`` with resolved asset references."""
    result = dict(shot_spec)
    resolved: dict[str, dict | str] = {}
    for key, asset_id in asset_refs.items():
        if not asset_id:
            continue
        if asset_loader is None:
            resolved[key] = asset_id
            continue
        try:
            data = asset_loader(key, asset_id)
        except Exception:
            data = None
        resolved[key] = data if data is not None else asset_id
    result["_assets"] = resolved
    return result


__all__ = ["asset_reference_inject"]
