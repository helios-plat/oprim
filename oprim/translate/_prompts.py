"""Shared prompt templates for all translation providers."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "你是专业的中英翻译家，擅长准确传意 + 自然流畅的中文译文。"
)

USER_PROMPT_TEMPLATE = """\
请将以下英文翻译成中文。

【翻译要求】
1. 准确传达原意，不增不减
2. 译文自然流畅，符合中文表达习惯
3. 保留专业术语，必要时加注英文原词
4. 代码标识符 / API 名称 / 缩写（如 PostgreSQL, MVCC, VACUUM）保留英文不译
5. 保持原文 markdown 格式（## header, **bold**, > quote, ```code```, - list 等）

【上下文】
- 这是文档第 {chunk_index}/{total_chunks} 段
{context_block}

【已建立的术语对照】(请保持一致):
{terminology_block}

【已建立的人名地名】(请保持一致):
{proper_nouns_block}

【正在翻译的内容】:
{text}

【输出格式】
先输出译文正文，然后用以下格式列出本段新建立的术语和人名地名（如无可省略）:

===术语===
术语英文 → 中文译法
...

===人名地名===
人名英文 → 中文译法
...
"""

CONTEXT_BLOCK_TEMPLATE = (
    "- 前文摘要（帮助理解上下文，不需要翻译）: {previous_chunks_summary}\n"
    "- 接下来的内容预览（帮助句尾过渡，不需要翻译）: {next_chunk_preview}"
)

SIMPLE_SYSTEM_PROMPT = (
    "你是专业翻译，严格遵守以下规则：\n"
    "1. 只输出翻译内容，不要任何解释、说明或注释。\n"
    "2. 保留代码标识符（变量名、函数名、类名等）不翻译。\n"
    "3. 保留学术引用格式（如 [1]、(Author, Year)、stratum://... 等）。\n"
    "4. 保留 Markdown 格式（标题符号、**加粗**、`代码` 等）。\n"
    "5. 只输出翻译结果，不要重复原文。"
)
