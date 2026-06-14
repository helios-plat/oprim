"""Tests — H-B E组: 网络 IO 扩展 (validate_api_key / upload_share / revoke_share / fetch_models_dev / load_skill_raw)."""
from __future__ import annotations

from pathlib import Path

import pytest
import respx
import httpx

from oprim._hb_network import (
    ModelSpec,
    fetch_models_dev,
    load_skill_raw,
    revoke_share,
    upload_share,
    validate_api_key,
)


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_api_key_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        await validate_api_key("", provider="anthropic")


@pytest.mark.asyncio
async def test_validate_api_key_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        await validate_api_key("sk-123", provider="nonexistent_xyz")


@pytest.mark.asyncio
@respx.mock
async def test_validate_api_key_valid() -> None:
    respx.get("https://api.anthropic.com/v1/models").mock(return_value=httpx.Response(200, json={"models": []}))
    result = await validate_api_key("sk-valid", provider="anthropic")
    assert result is True


@pytest.mark.asyncio
@respx.mock
async def test_validate_api_key_invalid_401() -> None:
    respx.get("https://api.anthropic.com/v1/models").mock(return_value=httpx.Response(401))
    result = await validate_api_key("sk-bad", provider="anthropic")
    assert result is False


@pytest.mark.asyncio
@respx.mock
async def test_validate_api_key_openai() -> None:
    respx.get("https://api.openai.com/v1/models").mock(return_value=httpx.Response(200, json={}))
    result = await validate_api_key("sk-openai-key", provider="openai")
    assert result is True


# ---------------------------------------------------------------------------
# upload_share
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_share_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="empty"):
        await upload_share({"x": 1}, endpoint="")


@pytest.mark.asyncio
async def test_upload_share_invalid_endpoint() -> None:
    with pytest.raises(ValueError, match="http"):
        await upload_share({"x": 1}, endpoint="ftp://bad.example.com")


@pytest.mark.asyncio
@respx.mock
async def test_upload_share_success() -> None:
    respx.post("https://share.example.com/upload").mock(
        return_value=httpx.Response(200, json={"url": "https://share.example.com/s/abc"})
    )
    url = await upload_share({"session": "data"}, endpoint="https://share.example.com/upload")
    assert url == "https://share.example.com/s/abc"


@pytest.mark.asyncio
@respx.mock
async def test_upload_share_413() -> None:
    from oprim._exceptions import HttpOprimError
    respx.post("https://share.example.com/upload").mock(return_value=httpx.Response(413))
    with pytest.raises(HttpOprimError, match="413"):
        await upload_share({"big": "x" * 1000}, endpoint="https://share.example.com/upload")


@pytest.mark.asyncio
@respx.mock
async def test_upload_share_failure() -> None:
    from oprim._exceptions import HttpOprimError
    respx.post("https://share.example.com/upload").mock(return_value=httpx.Response(500, text="err"))
    with pytest.raises(HttpOprimError):
        await upload_share({}, endpoint="https://share.example.com/upload")


# ---------------------------------------------------------------------------
# revoke_share
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_revoke_share_success_204() -> None:
    respx.delete("https://share.example.com/s/abc").mock(return_value=httpx.Response(204))
    await revoke_share("https://share.example.com/s/abc")  # no exception


@pytest.mark.asyncio
@respx.mock
async def test_revoke_share_already_gone_404() -> None:
    respx.delete("https://share.example.com/s/gone").mock(return_value=httpx.Response(404))
    await revoke_share("https://share.example.com/s/gone")  # idempotent


@pytest.mark.asyncio
@respx.mock
async def test_revoke_share_idempotent_409() -> None:
    respx.delete("https://share.example.com/s/x").mock(return_value=httpx.Response(409))
    await revoke_share("https://share.example.com/s/x")  # idempotent


@pytest.mark.asyncio
@respx.mock
async def test_revoke_share_failure() -> None:
    from oprim._exceptions import HttpOprimError
    respx.delete("https://share.example.com/s/bad").mock(return_value=httpx.Response(500, text="err"))
    with pytest.raises(HttpOprimError):
        await revoke_share("https://share.example.com/s/bad")


# ---------------------------------------------------------------------------
# fetch_models_dev
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_dev_dict_format() -> None:
    payload = {
        "anthropic": [
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet",
             "context_length": 200000, "pricing": {"input": 3.0, "output": 15.0}}
        ]
    }
    respx.get("https://models.dev/api/models.json").mock(
        return_value=httpx.Response(200, json=payload)
    )
    models = await fetch_models_dev()
    assert len(models) == 1
    assert models[0].id == "claude-3-5-sonnet"
    assert models[0].provider == "anthropic"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_dev_empty() -> None:
    respx.get("https://models.dev/api/models.json").mock(
        return_value=httpx.Response(200, json={})
    )
    models = await fetch_models_dev()
    assert models == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_dev_network_error() -> None:
    from oprim._exceptions import HttpOprimError
    respx.get("https://models.dev/api/models.json").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(HttpOprimError):
        await fetch_models_dev()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_dev_http_error() -> None:
    from oprim._exceptions import HttpOprimError
    respx.get("https://models.dev/api/models.json").mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )
    with pytest.raises(HttpOprimError):
        await fetch_models_dev()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_dev_list_format() -> None:
    payload = [{"id": "gpt-4o", "name": "GPT-4o", "provider": "openai", "context_length": 128000}]
    respx.get("https://models.dev/api/models.json").mock(
        return_value=httpx.Response(200, json=payload)
    )
    models = await fetch_models_dev()
    assert models[0].id == "gpt-4o"
    assert models[0].provider == "openai"


# ---------------------------------------------------------------------------
# load_skill_raw
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_skill_raw_normal(tmp_path: Path) -> None:
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("# SKILL: web_search\n\nDoes web searches.\n")
    raw = await load_skill_raw(skill_file)
    assert "web_search" in raw


@pytest.mark.asyncio
async def test_load_skill_raw_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await load_skill_raw(tmp_path / "MISSING.md")


@pytest.mark.asyncio
async def test_load_skill_raw_empty(tmp_path: Path) -> None:
    f = tmp_path / "SKILL.md"
    f.write_text("")
    raw = await load_skill_raw(f)
    assert raw == ""


@pytest.mark.asyncio
async def test_load_skill_raw_returns_string(tmp_path: Path) -> None:
    f = tmp_path / "SKILL.md"
    content = "# Test Skill\n\n```python\ndef fn(): pass\n```\n"
    f.write_text(content)
    raw = await load_skill_raw(f)
    assert isinstance(raw, str)
    assert raw == content


@pytest.mark.asyncio
async def test_load_skill_raw_large(tmp_path: Path) -> None:
    f = tmp_path / "SKILL.md"
    f.write_text("x" * 100_000)
    raw = await load_skill_raw(f)
    assert len(raw) == 100_000
