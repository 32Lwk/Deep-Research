"""スタブの ProviderRegistry で /api/providers と /api/chat を外部キーなしで検証する。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

import deepresearch.web as web
from deepresearch.providers.base import LlmProvider, Message
from deepresearch.providers.registry import ProviderRegistry


class StubChatProvider(LlmProvider):
    name = "stub"

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        last = messages[-1].content if messages else ""
        return f"echo:{last}"


@pytest.fixture
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> ProviderRegistry:
    stub = StubChatProvider()
    reg = ProviderRegistry(
        providers={name: stub for name in ("openai", "anthropic", "gemini", "plamo")},
    )
    monkeypatch.setattr(web, "registry", reg)
    return reg


@pytest.fixture
def client() -> TestClient:
    with TestClient(web.app) as c:
        yield c


def test_api_providers_lists_stub_providers(client: TestClient, stub_registry: ProviderRegistry) -> None:
    r = client.get("/api/providers")
    assert r.status_code == 200
    data = r.json()
    assert set(data["available"]) == {"anthropic", "gemini", "openai", "plamo"}
    assert isinstance(data.get("chat_models"), dict)
    errs = data.get("chat_models_errors")
    assert isinstance(errs, dict)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "plamo"])
def test_api_chat_echoes_each_provider(
    client: TestClient,
    stub_registry: ProviderRegistry,
    provider: str,
) -> None:
    r = client.post("/api/chat", json={"provider": provider, "message": "ping"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("error") is None
    assert body["text"] == "echo:ping"
    assert body.get("provider") == provider


def test_api_chat_rejects_unknown_provider(client: TestClient, stub_registry: ProviderRegistry) -> None:
    r = client.post("/api/chat", json={"provider": "unknown", "message": "x"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("error"), str)
    assert "unknown" in body["error"]


def test_api_chat_models_matches_catalog_shape(client: TestClient, stub_registry: ProviderRegistry) -> None:
    r = client.get("/api/chat/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "errors" in data
