"""Serialization submodule."""
__oprim_layer__ = "infra"  # publishes no element of its own; shared base

from oprim.serialization.canonical import canonical_json

__all__ = ["canonical_json"]
