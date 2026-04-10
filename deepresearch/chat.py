from __future__ import annotations

import asyncio
from typing import Any

from deepresearch.providers import Message, ProviderRegistry
from deepresearch.static_chat_models import curated_chat_models_for_providers

# 単発チャット用の既定モデル（routing.yaml の研究パイプラインとは別）
# Anthropic: 旧 claude-3-5-sonnet-* は廃止・エイリアス変更があり得るため、Messages API で一般的な Sonnet 4 系を既定にする。
CHAT_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-flash",
    "plamo": "plamo-2.0-prime",
}

# UI の選択肢（網羅ではなく「よく使う」候補）。
# 実際に利用可能なモデルは契約/リージョン/公開状況で変わるため、必要なら UI 側でカスタム入力も許容する。
CHAT_MODEL_OPTIONS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-20251001"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "plamo": ["plamo-2.0-prime"],
}


def chat_default_models_for(available: list[str]) -> dict[str, str]:
    """登録済みプロバイダは必ず既定モデル文字列を返す（将来プロバイダ追加時も空にならない）。"""
    out: dict[str, str] = {}
    for p in available:
        if p in CHAT_DEFAULT_MODELS:
            out[p] = CHAT_DEFAULT_MODELS[p]
        else:
            out[p] = "gpt-4o-mini"
    return out


def chat_model_options_for(available: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in available:
        if p in CHAT_MODEL_OPTIONS:
            out[p] = list(CHAT_MODEL_OPTIONS[p])
        elif p in CHAT_DEFAULT_MODELS:
            d = CHAT_DEFAULT_MODELS[p]
            out[p] = [d]
        else:
            out[p] = ["gpt-4o-mini"]
    return out


async def fetch_chat_models_catalog(
    registry: ProviderRegistry,
    *,
    query_live: bool = False,
) -> dict[str, Any]:
    """
    チャット用モデル ID 一覧。
    - 常に `static_chat_models` の静的一覧をベースにする（API が使えなくても UI で選べる）。
    - query_live=True のときだけ各プロバイダの List Models を並列取得し、結果を併合する。
    """
    avail = registry.available()
    defaults = chat_default_models_for(avail)
    curated = curated_chat_models_for_providers(avail, defaults)
    petite = chat_model_options_for(avail)

    if not query_live:
        return {"models": curated, "errors": {}}

    async def run_one(name: str) -> tuple[str, list[str], str | None]:
        base = set(curated.get(name, [])) | set(petite.get(name, []))
        try:
            prov = registry.get(name)
            found = await prov.list_chat_models()
            merged = sorted(base | set(found))
            if not merged:
                merged = sorted(base)
            return name, merged, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            return name, sorted(base), err

    pairs = await asyncio.gather(*(run_one(n) for n in avail))
    models_out: dict[str, list[str]] = {}
    errors_out: dict[str, str] = {}
    for name, models, err in pairs:
        models_out[name] = models
        if err:
            errors_out[name] = err
    return {"models": models_out, "errors": errors_out}


async def chat_turn(
    registry: ProviderRegistry,
    *,
    provider: str,
    message: str,
    model: str | None = None,
    system: str | None = None,
    max_output_tokens: int = 2048,
    temperature: float = 0.3,
) -> tuple[str, str]:
    """
    登録済みプロバイダへ1ターンの user メッセージを送り、テキストと実際に使った model を返す。
    """
    p = registry.get(provider)
    m = (model or "").strip() or CHAT_DEFAULT_MODELS.get(provider)
    if not m:
        raise ValueError(f"no default model for provider: {provider}")

    msgs: list[Message] = []
    if system and system.strip():
        msgs.append(Message(role="system", content=system.strip()))
    msgs.append(Message(role="user", content=message.strip()))

    text = await p.complete(
        model=m,
        messages=msgs,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return text.strip(), m
