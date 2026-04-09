from __future__ import annotations

import asyncio
from typing import AsyncIterator, Sequence

from google import genai
from google.genai import errors as genai_errors

from .base import LlmProvider, LlmResponseChunk, Message


class GeminiProvider(LlmProvider):
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_prompt(messages: Sequence[Message]) -> str:
        system = "\n".join([m.content for m in messages if m.role == "system"]).strip()
        user_text = "\n\n".join([f"{m.role.upper()}:\n{m.content}" for m in messages if m.role != "system"]).strip()
        return (system + "\n\n" + user_text).strip() if system else user_text

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        # google-genai sync client: offload so the asyncio loop can keep serving WS/UI.
        prompt = self._build_prompt(messages)
        cfg = {
            "temperature": temperature,
            **({"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}),
        }
        fallbacks = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
        tried: list[str] = []
        for m in [model] + [fb for fb in fallbacks if fb != model]:
            tried.append(m)

            def _call(model_name: str = m) -> str:
                resp = self._client.models.generate_content(model=model_name, contents=prompt, config=cfg)
                return (resp.text or "").strip()

            try:
                return await asyncio.to_thread(_call)
            except genai_errors.ClientError as e:
                if getattr(e, "status_code", None) in (404, 400):
                    continue
                raise
        raise RuntimeError(f"Gemini model failed. tried={tried}")

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[LlmResponseChunk]:
        prompt = self._build_prompt(messages)
        cfg = {
            "temperature": temperature,
            **({"max_output_tokens": max_output_tokens} if max_output_tokens is not None else {}),
        }
        fallbacks = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"]
        tried: list[str] = []
        last_err: BaseException | None = None

        for mname in [model] + [fb for fb in fallbacks if fb != model]:
            tried.append(mname)
            try:
                # See google.genai docs: async for chunk in await client.aio.models.generate_content_stream(...)
                stream_it = await self._client.aio.models.generate_content_stream(
                    model=mname,
                    contents=prompt,
                    config=cfg,
                )
                prev_full = ""
                async for resp in stream_it:
                    full = (resp.text or "")
                    if not full:
                        continue
                    # Chunks may be cumulative or incremental; normalize to deltas.
                    if full.startswith(prev_full):
                        delta = full[len(prev_full) :]
                        prev_full = full
                    else:
                        delta = full
                        prev_full = prev_full + delta
                    if delta:
                        yield LlmResponseChunk(text=delta)
                return
            except genai_errors.ClientError as e:
                last_err = e
                if getattr(e, "status_code", None) in (404, 400):
                    continue
                raise

        raise RuntimeError(f"Gemini stream failed. tried={tried}") from last_err
