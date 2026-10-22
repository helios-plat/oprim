"""Public facade for LLM summarization."""
from __future__ import annotations
from typing import Any
from oprim._llm_summarize import SummarizeResult, llm_summarize as _llm_summarize

def llm_summarize(*args: Any, **kwargs: Any) -> Any:
    return _llm_summarize(*args, **kwargs)

__all__ = ["SummarizeResult", "llm_summarize"]
