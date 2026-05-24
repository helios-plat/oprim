"""Tests for P6-B2 oprim modules: image_to_video, face_animation, motion_prompt_translate,
audience_sentiment_analyze, audience_feedback_extract, youtube_video_stats,
youtube_comments_fetch, bilibili_video_stats, bilibili_comments_fetch,
video_quality_metrics, vlm_video_analyze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from obase import ProviderRegistry


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# image_to_video
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageToVideo:
    async def test_success(self, tmp_path: Path) -> None:
        from oprim.image_to_video import image_to_video

        img = tmp_path / "img.png"
        img.write_bytes(b"PNG")
        out = tmp_path / "out.mp4"

        async def _gen(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"video")

        ProviderRegistry.register(category="image_to_video", name="mock", fn=_gen)
        result = await image_to_video(
            provider="mock", reference_image=img, motion_prompt="pan", output_path=out,
        )
        assert result == out

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        from oprim.image_to_video import ImageToVideoProviderNotFoundError, image_to_video

        img = tmp_path / "img.png"
        img.write_bytes(b"PNG")
        with pytest.raises(ImageToVideoProviderNotFoundError):
            await image_to_video(
                provider="nope", reference_image=img, motion_prompt="x",
                output_path=tmp_path / "o.mp4",
            )

    async def test_image_not_found(self, tmp_path: Path) -> None:
        from oprim.image_to_video import ImageToVideoError, image_to_video

        with pytest.raises(ImageToVideoError, match="not found"):
            await image_to_video(
                provider="x", reference_image=tmp_path / "nope.png",
                motion_prompt="x", output_path=tmp_path / "o.mp4",
            )

    async def test_provider_failure(self, tmp_path: Path) -> None:
        from oprim.image_to_video import ImageToVideoError, image_to_video

        img = tmp_path / "img.png"
        img.write_bytes(b"PNG")

        async def _fail(**kw: Any) -> None:
            raise RuntimeError("boom")

        ProviderRegistry.register(category="image_to_video", name="bad", fn=_fail)
        with pytest.raises(ImageToVideoError, match="boom"):
            await image_to_video(
                provider="bad", reference_image=img, motion_prompt="x",
                output_path=tmp_path / "o.mp4",
            )

    async def test_no_output_produced(self, tmp_path: Path) -> None:
        from oprim.image_to_video import ImageToVideoError, image_to_video

        img = tmp_path / "img.png"
        img.write_bytes(b"PNG")

        async def _noop(**kw: Any) -> None:
            pass  # doesn't create output

        ProviderRegistry.register(category="image_to_video", name="noop", fn=_noop)
        with pytest.raises(ImageToVideoError, match="did not produce"):
            await image_to_video(
                provider="noop", reference_image=img, motion_prompt="x",
                output_path=tmp_path / "o.mp4",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# face_animation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceAnimation:
    async def test_success(self, tmp_path: Path) -> None:
        from oprim.face_animation import face_animation

        img = tmp_path / "face.png"
        img.write_bytes(b"PNG")
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"WAV")
        out = tmp_path / "out.mp4"

        async def _gen(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"video")

        ProviderRegistry.register(category="face_animation", name="mock", fn=_gen)
        result = await face_animation(
            provider="mock", portrait_image=img, audio_path=audio, output_path=out,
        )
        assert result == out

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        from oprim.face_animation import FaceAnimationProviderNotFoundError, face_animation

        img = tmp_path / "f.png"
        img.write_bytes(b"x")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        with pytest.raises(FaceAnimationProviderNotFoundError):
            await face_animation(
                provider="nope", portrait_image=img, audio_path=audio,
                output_path=tmp_path / "o.mp4",
            )

    async def test_portrait_not_found(self, tmp_path: Path) -> None:
        from oprim.face_animation import FaceAnimationError, face_animation

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        with pytest.raises(FaceAnimationError, match="Portrait not found"):
            await face_animation(
                provider="x", portrait_image=tmp_path / "nope.png",
                audio_path=audio, output_path=tmp_path / "o.mp4",
            )

    async def test_audio_not_found(self, tmp_path: Path) -> None:
        from oprim.face_animation import FaceAnimationError, face_animation

        img = tmp_path / "f.png"
        img.write_bytes(b"x")
        with pytest.raises(FaceAnimationError, match="Audio not found"):
            await face_animation(
                provider="x", portrait_image=img,
                audio_path=tmp_path / "nope.wav", output_path=tmp_path / "o.mp4",
            )

    async def test_provider_failure(self, tmp_path: Path) -> None:
        from oprim.face_animation import FaceAnimationError, face_animation

        img = tmp_path / "f.png"
        img.write_bytes(b"x")
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")

        async def _fail(**kw: Any) -> None:
            raise RuntimeError("fail")

        ProviderRegistry.register(category="face_animation", name="bad", fn=_fail)
        with pytest.raises(FaceAnimationError, match="fail"):
            await face_animation(
                provider="bad", portrait_image=img, audio_path=audio,
                output_path=tmp_path / "o.mp4",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# motion_prompt_translate
# ═══════════════════════════════════════════════════════════════════════════════

class TestMotionPromptTranslate:
    async def test_success(self) -> None:
        from oprim.motion_prompt_translate import motion_prompt_translate

        llm = lambda **kw: {"content": "smooth cinematic pan left"}  # noqa: E731
        result = await motion_prompt_translate(
            natural_language_motion="slow pan left", llm=llm,
        )
        assert "pan" in result

    async def test_empty_motion_raises(self) -> None:
        from oprim.motion_prompt_translate import MotionTranslateError, motion_prompt_translate

        with pytest.raises(MotionTranslateError, match="must not be empty"):
            await motion_prompt_translate(natural_language_motion="", llm=lambda **kw: {})

    async def test_llm_failure(self) -> None:
        from oprim.motion_prompt_translate import MotionTranslateError, motion_prompt_translate

        def _fail(**kw: Any) -> dict[str, Any]:
            raise RuntimeError("LLM down")

        with pytest.raises(MotionTranslateError, match="LLM call failed"):
            await motion_prompt_translate(natural_language_motion="zoom", llm=_fail)

    async def test_empty_llm_response(self) -> None:
        from oprim.motion_prompt_translate import MotionTranslateError, motion_prompt_translate

        with pytest.raises(MotionTranslateError, match="empty translation"):
            await motion_prompt_translate(
                natural_language_motion="zoom", llm=lambda **kw: {"content": ""},
            )

    async def test_different_target_provider(self) -> None:
        from oprim.motion_prompt_translate import motion_prompt_translate

        calls: list[dict[str, Any]] = []

        def _llm(**kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return {"content": "result"}

        await motion_prompt_translate(
            natural_language_motion="tilt up", llm=_llm, target_provider="runway",
        )
        assert "runway" in str(calls[0]["messages"])


# ═══════════════════════════════════════════════════════════════════════════════
# audience_sentiment_analyze
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudienceSentimentAnalyze:
    async def test_success(self) -> None:
        from oprim.audience_sentiment_analyze import audience_sentiment_analyze

        resp = json.dumps({
            "positive_pct": 0.6, "negative_pct": 0.2,
            "neutral_pct": 0.2, "top_keywords": ["good"],
        })
        llm = lambda **kw: {"content": resp}  # noqa: E731
        result = await audience_sentiment_analyze(comments=["nice!"], llm=llm)
        assert result.positive_pct == 0.6

    async def test_empty_comments(self) -> None:
        from oprim.audience_sentiment_analyze import (
            SentimentAnalyzeError,
            audience_sentiment_analyze,
        )

        with pytest.raises(SentimentAnalyzeError, match="must not be empty"):
            await audience_sentiment_analyze(comments=[], llm=lambda **kw: {})

    async def test_llm_failure(self) -> None:
        from oprim.audience_sentiment_analyze import (
            SentimentAnalyzeError,
            audience_sentiment_analyze,
        )

        def _fail(**kw: Any) -> dict[str, Any]:
            raise RuntimeError("down")

        with pytest.raises(SentimentAnalyzeError, match="LLM call failed"):
            await audience_sentiment_analyze(comments=["x"], llm=_fail)

    async def test_invalid_json(self) -> None:
        from oprim.audience_sentiment_analyze import (
            SentimentAnalyzeError,
            audience_sentiment_analyze,
        )

        with pytest.raises(SentimentAnalyzeError, match="invalid JSON"):
            await audience_sentiment_analyze(
                comments=["x"], llm=lambda **kw: {"content": "not json"},
            )

    async def test_validation_failure(self) -> None:
        from oprim.audience_sentiment_analyze import (
            SentimentAnalyzeError,
            audience_sentiment_analyze,
        )

        bad = json.dumps({"positive_pct": "not_a_float"})
        with pytest.raises(SentimentAnalyzeError, match="Validation failed"):
            await audience_sentiment_analyze(
                comments=["x"], llm=lambda **kw: {"content": bad},
            )


# ═══════════════════════════════════════════════════════════════════════════════
# audience_feedback_extract
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudienceFeedbackExtract:
    async def test_success(self) -> None:
        from oprim.audience_feedback_extract import audience_feedback_extract

        resp = json.dumps({
            "positive_points": ["clear"], "negative_points": ["fast"],
            "questions": ["why?"], "suggestions": ["slow down"],
        })
        result = await audience_feedback_extract(
            comments=["too fast"], llm=lambda **kw: {"content": resp},
        )
        assert result.suggestions == ["slow down"]

    async def test_empty_comments(self) -> None:
        from oprim.audience_feedback_extract import FeedbackExtractError, audience_feedback_extract

        with pytest.raises(FeedbackExtractError, match="must not be empty"):
            await audience_feedback_extract(comments=[], llm=lambda **kw: {})

    async def test_llm_failure(self) -> None:
        from oprim.audience_feedback_extract import FeedbackExtractError, audience_feedback_extract

        def _fail(**kw: Any) -> dict[str, Any]:
            raise RuntimeError("down")

        with pytest.raises(FeedbackExtractError, match="LLM call failed"):
            await audience_feedback_extract(comments=["x"], llm=_fail)

    async def test_invalid_json(self) -> None:
        from oprim.audience_feedback_extract import FeedbackExtractError, audience_feedback_extract

        with pytest.raises(FeedbackExtractError, match="invalid JSON"):
            await audience_feedback_extract(
                comments=["x"], llm=lambda **kw: {"content": "nope"},
            )

    async def test_validation_failure(self) -> None:
        from oprim.audience_feedback_extract import FeedbackExtractError, audience_feedback_extract

        bad = json.dumps({"positive_points": 123})
        with pytest.raises(FeedbackExtractError, match="Validation failed"):
            await audience_feedback_extract(
                comments=["x"], llm=lambda **kw: {"content": bad},
            )


# ═══════════════════════════════════════════════════════════════════════════════
# youtube_video_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestYouTubeVideoStats:
    async def test_success(self) -> None:
        from oprim.youtube_video_stats import youtube_video_stats

        mock_stats = {
            "items": [{
                "statistics": {"viewCount": "100", "likeCount": "10", "commentCount": "5"},
                "contentDetails": {"duration": "PT1M30S"},
            }]
        }
        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: mock_stats,
            ))
            result = await youtube_video_stats(video_id="abc", oauth_token="tok")
        assert result.views == 100
        assert result.duration_s == 90.0

    async def test_auth_failure(self) -> None:
        from oprim.youtube_video_stats import YouTubeStatsError, youtube_video_stats

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=401, text="unauthorized",
            ))
            with pytest.raises(YouTubeStatsError, match="invalid"):
                await youtube_video_stats(video_id="x", oauth_token="bad")

    async def test_video_not_found(self) -> None:
        from oprim.youtube_video_stats import YouTubeStatsError, youtube_video_stats

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: {"items": []},
            ))
            with pytest.raises(YouTubeStatsError, match="not found"):
                await youtube_video_stats(video_id="nope", oauth_token="tok")

    async def test_rate_limit(self) -> None:
        from oprim.youtube_video_stats import YouTubeStatsError, youtube_video_stats

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=429, text="rate limited",
            ))
            with pytest.raises(YouTubeStatsError, match="429"):
                await youtube_video_stats(video_id="x", oauth_token="tok")

    async def test_pydantic_output(self) -> None:
        from oprim._providers.youtube_api import VideoStats

        s = VideoStats(
            video_id="x", views=1, likes=2, comments_count=3, duration_s=60.0,
        )
        assert s.video_id == "x"


# ═══════════════════════════════════════════════════════════════════════════════
# youtube_comments_fetch
# ═══════════════════════════════════════════════════════════════════════════════

class TestYouTubeCommentsFetch:
    async def test_single_page(self) -> None:
        from oprim.youtube_comments_fetch import youtube_comments_fetch

        page_data = {
            "items": [{
                "id": "c1",
                "snippet": {"topLevelComment": {"snippet": {
                    "authorDisplayName": "user",
                    "textDisplay": "great",
                    "publishedAt": "2024-01-01T00:00:00Z",
                    "likeCount": 1,
                }}},
            }],
        }
        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: page_data,
            ))
            result = await youtube_comments_fetch(video_id="x", oauth_token="t")
        assert len(result) == 1
        assert result[0].text == "great"

    async def test_pagination(self) -> None:
        from oprim.youtube_comments_fetch import youtube_comments_fetch

        page1 = {
            "items": [{"id": "c1", "snippet": {"topLevelComment": {"snippet": {
                "authorDisplayName": "u", "textDisplay": "a",
                "publishedAt": "2024-01-01T00:00:00Z", "likeCount": 0,
            }}}}],
            "nextPageToken": "page2",
        }
        page2 = {
            "items": [{"id": "c2", "snippet": {"topLevelComment": {"snippet": {
                "authorDisplayName": "u", "textDisplay": "b",
                "publishedAt": "2024-01-01T00:00:00Z", "likeCount": 0,
            }}}}],
        }
        call_count = [0]

        def _resp(*a: Any, **kw: Any) -> AsyncMock:
            call_count[0] += 1
            data = page1 if call_count[0] == 1 else page2
            return AsyncMock(status_code=200, json=lambda: data)

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=_resp)
            result = await youtube_comments_fetch(video_id="x", oauth_token="t", max_count=5)
        assert len(result) == 2

    async def test_max_count_boundary(self) -> None:
        from oprim.youtube_comments_fetch import youtube_comments_fetch

        page = {"items": [
            {"id": f"c{i}", "snippet": {"topLevelComment": {"snippet": {
                "authorDisplayName": "u", "textDisplay": f"t{i}",
                "publishedAt": "2024-01-01T00:00:00Z", "likeCount": 0,
            }}}} for i in range(10)
        ]}
        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: page,
            ))
            result = await youtube_comments_fetch(video_id="x", oauth_token="t", max_count=3)
        assert len(result) == 3

    async def test_auth_failure(self) -> None:
        from oprim.youtube_comments_fetch import YouTubeCommentsError, youtube_comments_fetch

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=401, text="bad",
            ))
            with pytest.raises(YouTubeCommentsError):
                await youtube_comments_fetch(video_id="x", oauth_token="bad")

    async def test_empty_comments(self) -> None:
        from oprim.youtube_comments_fetch import youtube_comments_fetch

        with patch("oprim._providers.youtube_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: {"items": []},
            ))
            result = await youtube_comments_fetch(video_id="x", oauth_token="t")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# bilibili_video_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestBilibiliVideoStats:
    async def test_success(self) -> None:
        from oprim.bilibili_video_stats import bilibili_video_stats

        mock_data = {"code": 0, "data": {"stat": {
            "view": 500, "like": 50, "coin": 10, "favorite": 20, "share": 5,
        }}}
        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: mock_data,
            ))
            result = await bilibili_video_stats(bvid="BV1xx", cookies={"SESSDATA": "x"})
        assert result.views == 500

    async def test_cookies_invalid(self) -> None:
        from oprim.bilibili_video_stats import BilibiliStatsError, bilibili_video_stats

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(
                status_code=200, json=lambda: {"code": -101, "message": "not logged in"},
            ))
            with pytest.raises(BilibiliStatsError):
                await bilibili_video_stats(bvid="BV1xx", cookies={})

    async def test_rate_limit(self) -> None:
        from oprim.bilibili_video_stats import BilibiliStatsError, bilibili_video_stats

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(status_code=429))
            with pytest.raises(BilibiliStatsError, match="429"):
                await bilibili_video_stats(bvid="BV1xx", cookies={})

    async def test_pydantic_model(self) -> None:
        from oprim._providers.bilibili_api import BiliVideoStats

        s = BiliVideoStats(bvid="BV1", views=1, likes=2, coins=3, favorites=4, shares=5)
        assert s.bvid == "BV1"

    async def test_http_error(self) -> None:
        from oprim.bilibili_video_stats import BilibiliStatsError, bilibili_video_stats

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(status_code=500))
            with pytest.raises(BilibiliStatsError):
                await bilibili_video_stats(bvid="BV1xx", cookies={})


# ═══════════════════════════════════════════════════════════════════════════════
# bilibili_comments_fetch
# ═══════════════════════════════════════════════════════════════════════════════

class TestBilibiliCommentsFetch:
    async def test_success(self) -> None:
        from oprim.bilibili_comments_fetch import bilibili_comments_fetch

        info_data = {"code": 0, "data": {"aid": 12345, "stat": {}}}
        comments_data = {"code": 0, "data": {
            "replies": [{"rpid": 1, "member": {"uname": "u"}, "content": {"message": "hi"},
                         "ctime": 1700000000, "like": 3}],
            "page": {"count": 1},
        }}
        call_count = [0]

        def _resp(*a: Any, **kw: Any) -> AsyncMock:
            call_count[0] += 1
            data = info_data if call_count[0] == 1 else comments_data
            return AsyncMock(status_code=200, json=lambda: data)

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=_resp)
            result = await bilibili_comments_fetch(bvid="BV1xx", cookies={"SESSDATA": "x"})
        assert len(result) == 1
        assert result[0].text == "hi"

    async def test_api_failure(self) -> None:
        from oprim.bilibili_comments_fetch import (
            BilibiliCommentsError,
            bilibili_comments_fetch,
        )

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=AsyncMock(status_code=500))
            with pytest.raises(BilibiliCommentsError):
                await bilibili_comments_fetch(bvid="BV1xx", cookies={})

    async def test_empty_replies(self) -> None:
        from oprim.bilibili_comments_fetch import bilibili_comments_fetch

        info_data = {"code": 0, "data": {"aid": 1, "stat": {}}}
        comments_data = {"code": 0, "data": {"replies": None, "page": {"count": 0}}}
        call_count = [0]

        def _resp(*a: Any, **kw: Any) -> AsyncMock:
            call_count[0] += 1
            data = info_data if call_count[0] == 1 else comments_data
            return AsyncMock(status_code=200, json=lambda: data)

        with patch("oprim._providers.bilibili_api.httpx.AsyncClient") as mc:
            instance = AsyncMock()
            mc.return_value.__aenter__ = AsyncMock(return_value=instance)
            mc.return_value.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=_resp)
            result = await bilibili_comments_fetch(bvid="BV1xx", cookies={})
        assert result == []

    async def test_max_count(self) -> None:
        from oprim._providers.bilibili_api import BiliComment, BiliCommentsPage

        page = BiliCommentsPage(comments=[
            BiliComment(comment_id="1", author="u", text="t",
                        published_at="2024-01-01T00:00:00Z", likes=0),
        ] * 5, has_next=False)
        assert len(page.comments) == 5

    async def test_pydantic_model(self) -> None:
        from oprim._providers.bilibili_api import BiliComment

        c = BiliComment(
            comment_id="1", author="u", text="hi",
            published_at="2024-01-01T00:00:00+00:00", likes=5,
        )
        assert c.author == "u"


# ═══════════════════════════════════════════════════════════════════════════════
# video_quality_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestVideoQualityMetrics:
    async def test_success(self, tmp_path: Path) -> None:
        from oprim.video_quality_metrics import video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        ffprobe_out = json.dumps({
            "format": {"duration": "10.5", "bit_rate": "1500000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264",
                 "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        })
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(ffprobe_out.encode(), b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            result = await video_quality_metrics(video_path=video)

        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0
        assert result.codec_video == "h264"
        assert result.codec_audio == "aac"

    async def test_no_audio_stream(self, tmp_path: Path) -> None:
        from oprim.video_quality_metrics import video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")

        ffprobe_out = json.dumps({
            "format": {"duration": "5", "bit_rate": "800000"},
            "streams": [
                {"codec_type": "video", "codec_name": "vp9",
                 "width": 720, "height": 480, "r_frame_rate": "25/1"},
            ],
        })
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(ffprobe_out.encode(), b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            result = await video_quality_metrics(video_path=video)

        assert result.codec_audio is None
        assert result.audio_lufs is None

    async def test_file_not_found(self, tmp_path: Path) -> None:
        from oprim.video_quality_metrics import VideoQualityError, video_quality_metrics

        with pytest.raises(VideoQualityError, match="not found"):
            await video_quality_metrics(video_path=tmp_path / "nope.mp4")

    async def test_ffprobe_missing(self, tmp_path: Path) -> None:
        from oprim.video_quality_metrics import VideoQualityError, video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(VideoQualityError, match="ffprobe not found"):
                await video_quality_metrics(video_path=video)

    async def test_ffprobe_failure(self, tmp_path: Path) -> None:
        from oprim.video_quality_metrics import VideoQualityError, video_quality_metrics

        video = tmp_path / "v.mp4"
        video.write_bytes(b"fake")
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"error"))
            proc.returncode = 1
            mock_exec.return_value = proc
            with pytest.raises(VideoQualityError, match="ffprobe failed"):
                await video_quality_metrics(video_path=video)


# ═══════════════════════════════════════════════════════════════════════════════
# vlm_video_analyze
# ═══════════════════════════════════════════════════════════════════════════════

class TestVLMVideoAnalyze:
    async def test_success(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import vlm_video_analyze

        f = tmp_path / "frame.png"
        f.write_bytes(b"PNG")

        async def _vlm(**kw: Any) -> dict[str, str]:
            return {"content": "A person walking"}

        ProviderRegistry.register(category="vlm", name="mock", fn=_vlm)
        result = await vlm_video_analyze(
            provider="mock", frames=[f], prompt="Describe",
        )
        assert "walking" in result

    async def test_empty_frames(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze

        with pytest.raises(VLMVideoAnalyzeError, match="frames must not be empty"):
            await vlm_video_analyze(provider="x", frames=[], prompt="x")

    async def test_empty_prompt(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze

        f = tmp_path / "f.png"
        f.write_bytes(b"x")
        with pytest.raises(VLMVideoAnalyzeError, match="prompt must not be empty"):
            await vlm_video_analyze(provider="x", frames=[f], prompt="")

    async def test_frame_not_found(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze

        with pytest.raises(VLMVideoAnalyzeError, match="Frame not found"):
            await vlm_video_analyze(
                provider="x", frames=[tmp_path / "nope.png"], prompt="x",
            )

    async def test_provider_not_found(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze

        f = tmp_path / "f.png"
        f.write_bytes(b"x")
        with pytest.raises(VLMVideoAnalyzeError, match="not found"):
            await vlm_video_analyze(provider="nope", frames=[f], prompt="x")

    async def test_vlm_failure(self, tmp_path: Path) -> None:
        from oprim.vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze

        f = tmp_path / "f.png"
        f.write_bytes(b"x")

        async def _fail(**kw: Any) -> None:
            raise RuntimeError("VLM down")

        ProviderRegistry.register(category="vlm", name="bad", fn=_fail)
        with pytest.raises(VLMVideoAnalyzeError, match="VLM call failed"):
            await vlm_video_analyze(provider="bad", frames=[f], prompt="x")
