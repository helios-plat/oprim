"""Tests for oprim.structural_chunk."""

import pytest
from oprim.structural_chunk import structural_chunk


def test_empty_text_returns_empty():
    assert structural_chunk(text="") == []


def test_whitespace_only_returns_empty():
    assert structural_chunk(text="   \n\n  ") == []


def test_no_headers_single_chunk():
    text = "This is some plain text without any headers. It is long enough to pass the min_chars check."
    result = structural_chunk(text=text)
    assert len(result) == 1
    assert result[0]["heading"] == ""
    assert result[0]["level"] == 0


def test_h1_header_creates_chunk():
    text = "# Introduction\n\nThis is the introduction content that is long enough."
    result = structural_chunk(text=text)
    assert len(result) == 1
    assert result[0]["heading"] == "Introduction"
    assert result[0]["level"] == 1


def test_h2_inside_h1_context_path_includes_parent():
    text = (
        "# Chapter One\n\nSome chapter content here.\n\n"
        "## Section A\n\nSection content that is long enough to pass the minimum chars check."
    )
    result = structural_chunk(text=text)
    # Find the H2 chunk
    h2_chunks = [c for c in result if c["level"] == 2]
    assert len(h2_chunks) == 1
    assert "Chapter One" in h2_chunks[0]["context_path"]


def test_multiple_sections_from_multiple_headers():
    text = (
        "# Header One\n\nContent for section one that is definitely long enough to pass.\n\n"
        "# Header Two\n\nContent for section two that is definitely long enough to pass.\n\n"
        "# Header Three\n\nContent for section three that is definitely long enough to pass."
    )
    result = structural_chunk(text=text)
    assert len(result) == 3


def test_small_chunk_below_min_chars_skipped():
    text = "# Big Section\n\nThis is a substantial section with enough content to pass.\n\n## Tiny\n\nHi"
    result = structural_chunk(text=text, min_chars=50)
    headings = [c["heading"] for c in result]
    assert "Tiny" not in headings


def test_large_chunk_split_at_paragraph_boundary():
    # Create a section that exceeds max_chars=200
    para1 = "A" * 120
    para2 = "B" * 120
    text = f"# Big Section\n\n{para1}\n\n{para2}"
    result = structural_chunk(text=text, max_chars=200)
    # Should produce 2 chunks from the single header section
    big_chunks = [c for c in result if c["heading"] == "Big Section"]
    assert len(big_chunks) == 2


def test_chunk_id_sequential_format():
    text = (
        "# Section One\n\nContent for section one that is definitely long enough.\n\n"
        "# Section Two\n\nContent for section two that is definitely long enough."
    )
    result = structural_chunk(text=text)
    assert result[0]["chunk_id"] == "chunk_001"
    assert result[1]["chunk_id"] == "chunk_002"


def test_heading_contains_no_hash_symbols():
    text = "## My Heading\n\nContent that is long enough to pass the minimum char check here."
    result = structural_chunk(text=text)
    assert len(result) == 1
    assert "#" not in result[0]["heading"]
    assert result[0]["heading"] == "My Heading"


def test_level_field_is_int():
    text = "# H1\n\nContent that is definitely long enough to pass the minimum.\n\n## H2\n\nMore content that is definitely long enough to pass the minimum."
    result = structural_chunk(text=text)
    for chunk in result:
        assert isinstance(chunk["level"], int)
    levels = [c["level"] for c in result]
    assert 1 in levels
    assert 2 in levels


def test_char_count_matches_content_length():
    text = "# Section\n\nThis is the content of this section and it should be long enough."
    result = structural_chunk(text=text)
    assert len(result) == 1
    assert result[0]["char_count"] == len(result[0]["content"])


def test_nested_h1_h2_h3_hierarchy():
    text = (
        "# Top Level\n\nTop level content that is long enough.\n\n"
        "## Second Level\n\nSecond level content that is long enough.\n\n"
        "### Third Level\n\nThird level content that is definitely long enough to pass."
    )
    result = structural_chunk(text=text)
    h3_chunks = [c for c in result if c["level"] == 3]
    assert len(h3_chunks) == 1
    assert "Top Level" in h3_chunks[0]["context_path"]
    assert "Second Level" in h3_chunks[0]["context_path"]


def test_context_path_empty_for_top_level_h1():
    text = "# Top Section\n\nThis content is long enough to pass min_chars threshold here."
    result = structural_chunk(text=text)
    assert len(result) == 1
    assert result[0]["context_path"] == []


def test_context_path_resets_on_sibling_h1():
    text = (
        "# First Chapter\n\nFirst chapter content that is definitely long enough to pass.\n\n"
        "## Sub Section\n\nSub section content that is definitely long enough to pass.\n\n"
        "# Second Chapter\n\nSecond chapter content that is definitely long enough to pass."
    )
    result = structural_chunk(text=text)
    second_chapter = [c for c in result if c["heading"] == "Second Chapter"]
    assert len(second_chapter) == 1
    # Second Chapter is H1 so context_path should be empty
    assert second_chapter[0]["context_path"] == []
