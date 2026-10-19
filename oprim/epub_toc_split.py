"""Public re-export of epub_toc_split (omodul imports this as a submodule)."""

from oprim._epub_toc_split import EpubBook, epub_toc_split

__all__ = ["epub_toc_split", "EpubBook"]
