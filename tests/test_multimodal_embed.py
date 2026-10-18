import pytest
from unittest.mock import AsyncMock, patch

from oprim._multimodal_embed import MultimodalEmbedError, multimodal_embed


@pytest.mark.asyncio
async def test_missing_client_and_obase_unavailable_raises():
    with patch.dict("sys.modules", {"obase.embedding_gemma": None}):
        with pytest.raises(MultimodalEmbedError):
            await multimodal_embed(text="hi")


@pytest.mark.asyncio
async def test_with_client():
    client = AsyncMock()
    client.embed = AsyncMock(return_value=[0.1, 0.2])
    res = await multimodal_embed(text="hi", client=client)
    assert res == [0.1, 0.2]
    client.embed.assert_awaited()
