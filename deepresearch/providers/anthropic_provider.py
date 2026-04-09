from __future__ import annotations

from typing import AsyncIterator, Sequence

from anthropic import AsyncAnthropic

from .base import LlmProvider, LlmResponseChunk, Message


class AnthropicProvider(LlmProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    def _to_prompt(self, messages: Sequence[Message]) -> tuple[str, list[dict]]:
        system = ""
        converted: list[dict] = []
        for m in messages:
            if m.role == "system":
                system += m.content + "\n"
            else:
                converted.append({"role": m.role, "content": m.content})
        return system.strip(), converted

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        system, converted = self._to_prompt(messages)
        r = await self._client.messages.create(
            model=model,
            system=system or None,
            messages=converted,
            temperature=temperature,
            max_tokens=max_output_tokens or 1024,
        )
        parts = []
        for b in r.content:
            if b.type == "text":
                parts.append(b.text)
        return "".join(parts).strip()

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LlmResponseChunk]:
        system, converted = self._to_prompt(messages)
        async with self._client.messages.stream(
            model=model,
            system=system or None,
            messages=converted,
            temperature=temperature,
            max_tokens=max_output_tokens or 1024,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield LlmResponseChunk(text=text, is_final=False)
        yield LlmResponseChunk(text="", is_final=True)

