from __future__ import annotations

from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from .base import LlmProvider, LlmResponseChunk, Message


class OpenAIProvider(LlmProvider):
    name = "openai"

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        r = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LlmResponseChunk]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield LlmResponseChunk(text=delta.content, is_final=False)
        yield LlmResponseChunk(text="", is_final=True)

