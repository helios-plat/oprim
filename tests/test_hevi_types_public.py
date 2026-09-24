"""Public oprim.hevi_types consumer boundary (HEVI canvas/director imports)."""

from oprim.hevi_types import CanvasEdge, CanvasNode, ProviderCapability, Subject, VideoQuality
from oprim._hevi_types import CanvasEdge as _CanvasEdge


def test_public_hevi_types_reexport() -> None:
    assert CanvasEdge is _CanvasEdge
    for cls in (CanvasNode, ProviderCapability, Subject, VideoQuality):
        assert cls is not None
