"""Public HEVI type exports.

The original private module remains as a one-release compatibility shim.  New
consumers must import this physical, public module instead.
"""

__oprim_layer__ = "infra"  # shared base module, not a capability

from oprim._hevi_types import (
    CanvasEdge,
    CanvasNode,
    ProviderCapability,
    Subject,
    VideoQuality,
)

__all__ = [
    "CanvasEdge",
    "CanvasNode",
    "ProviderCapability",
    "Subject",
    "VideoQuality",
]
