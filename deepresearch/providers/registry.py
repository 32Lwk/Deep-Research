from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepresearch.settings import Settings

from .anthropic_provider import AnthropicProvider
from .base import LlmProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


@dataclass(frozen=True)
class ProviderRegistry:
    providers: dict[str, LlmProvider]

    @staticmethod
    def from_settings(s: Settings) -> "ProviderRegistry":
        providers: dict[str, LlmProvider] = {}

        if s.openai_key():
            providers["openai"] = OpenAIProvider(api_key=s.openai_key() or "")

        if s.anthropic_key():
            providers["anthropic"] = AnthropicProvider(api_key=s.anthropic_key() or "")

        if s.gemini_key():
            providers["gemini"] = GeminiProvider(api_key=s.gemini_key() or "")

        # PLaMo: OpenAI-compatible
        plamo_key = s.plamo_key()
        if plamo_key or s.PLAMO_BASE_URL:
            # Official docs: Chat Completions endpoint
            # POST https://api.platform.preferredai.jp/v1/chat/completions
            # base_url should be https://api.platform.preferredai.jp/v1
            base_url = s.PLAMO_BASE_URL or "https://api.platform.preferredai.jp/v1"
            providers["plamo"] = OpenAIProvider(api_key=plamo_key or "EMPTY", base_url=base_url)

        return ProviderRegistry(providers=providers)

    def get(self, name: str) -> LlmProvider:
        if name not in self.providers:
            raise KeyError(f"provider not configured: {name}")
        return self.providers[name]

    def available(self) -> list[str]:
        return sorted(self.providers.keys())

