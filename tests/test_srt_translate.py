"""Tests for oprim.srt_translate."""

from __future__ import annotations

from pathlib import Path

import pytest

from oprim.srt_translate import SRTParseError, SRTTranslateError, srt_translate

VALID_SRT = """\
1
00:00:00,000 --> 00:00:02,000
你好世界

2
00:00:02,000 --> 00:00:04,000
再见
"""


class MockLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def __call__(self, *, prompt: str) -> str:
        return self._response


class TestSRTTranslate:
    async def test_normal_translation(self, tmp_path: Path) -> None:
        src = tmp_path / "zh.srt"
        src.write_text(VALID_SRT)
        out = tmp_path / "en.srt"
        llm = MockLLM("Hello world\nGoodbye")
        result = await srt_translate(
            src_srt_path=src, target_lang="en", llm=llm, output_path=out
        )
        assert result == out
        content = out.read_text()
        assert "Hello world" in content
        assert "00:00:00,000 --> 00:00:02,000" in content

    async def test_srt_not_found(self, tmp_path: Path) -> None:
        llm = MockLLM("")
        with pytest.raises(SRTParseError, match="not found"):
            await srt_translate(
                src_srt_path=tmp_path / "missing.srt",
                target_lang="en", llm=llm, output_path=tmp_path / "out.srt",
            )

    async def test_invalid_srt_format(self, tmp_path: Path) -> None:
        src = tmp_path / "bad.srt"
        src.write_text("1\nNOT A TIMESTAMP\nText\n")
        llm = MockLLM("")
        with pytest.raises(SRTParseError, match="Invalid timestamp"):
            await srt_translate(
                src_srt_path=src, target_lang="en", llm=llm, output_path=tmp_path / "out.srt"
            )

    async def test_empty_srt_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.srt"
        src.write_text("")
        llm = MockLLM("")
        with pytest.raises(SRTParseError, match="empty"):
            await srt_translate(
                src_srt_path=src, target_lang="en", llm=llm, output_path=tmp_path / "out.srt"
            )

    async def test_llm_count_mismatch(self, tmp_path: Path) -> None:
        src = tmp_path / "zh.srt"
        src.write_text(VALID_SRT)
        llm = MockLLM("Only one line")  # expects 2
        with pytest.raises(SRTTranslateError, match="returned 1 lines, expected 2"):
            await srt_translate(
                src_srt_path=src, target_lang="en", llm=llm, output_path=tmp_path / "out.srt"
            )

    async def test_batch_boundary(self, tmp_path: Path) -> None:
        # 3 entries with batch_size=2 → 2 LLM calls
        srt_content = ""
        for i in range(3):
            srt_content += f"{i+1}\n00:00:{i:02d},000 --> 00:00:{i+1:02d},000\nLine {i}\n\n"
        src = tmp_path / "multi.srt"
        src.write_text(srt_content)
        out = tmp_path / "out.srt"

        call_count = 0

        class BatchLLM:
            async def __call__(self, *, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                # Count expected lines from prompt
                count = int(prompt.split("following ")[1].split(" subtitle")[0])
                return "\n".join(f"Translated {i}" for i in range(count))

        await srt_translate(
            src_srt_path=src, target_lang="en", llm=BatchLLM(), output_path=out, batch_size=2
        )
        assert call_count == 2
        assert out.exists()
