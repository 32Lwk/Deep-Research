from __future__ import annotations

from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from .base import LlmProvider, LlmResponseChunk, Message


def _openai_compatible_chat_model_id(model_id: str) -> bool:
    """Chat Completions で使いがちなモデルに絞る（一覧 API は非チャット系も含む）。"""
    m = model_id.lower()
    if not m:
        return False
    if any(x in m for x in ("embedding", "whisper", "tts", "dall-e", "moderation")):
        return False
    if m.startswith(("davinci", "babbage", "ada-", "curie", "text-")):
        return False
    if "-realtime" in m or m.endswith("-audio") or "transcribe" in m:
        return False
    if m.endswith("-instruct"):
        return False
    return True


class OpenAIProvider(LlmProvider):
    name = "openai"

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def list_chat_models(self) -> list[str]:
        raw: list[str] = []
        async for item in self._client.models.list():
            mid = getattr(item, "id", None)
            if isinstance(mid, str) and mid.strip():
                raw.append(mid.strip())
        filtered = [m for m in raw if _openai_compatible_chat_model_id(m)]
        if filtered:
            return sorted(set(filtered))
        # フィルタが厳しすぎて空になった場合は一覧をそのまま返す（Chat で失敗する ID はユーザー選択時に判別）
        if raw:
            return sorted(set(raw))
        return []

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

