from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Sequence


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class LlmResponseChunk:
    text: str
    is_final: bool = False


class LlmProvider:
    name: str

    async def list_chat_models(self) -> list[str]:
        """各プロバイダの API で利用可能なチャット用モデル ID を返す（未対応時は空）。"""
        return []

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LlmResponseChunk]:
        text = await self.complete(
            model=model, messages=messages, temperature=temperature, max_output_tokens=max_output_tokens
        )
        yield LlmResponseChunk(text=text, is_final=True)

