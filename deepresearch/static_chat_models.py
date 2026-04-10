"""
チャット UI 用の「静的」モデル ID 一覧。

各社の公開ドキュメント・モデル表をもとに Cursor 側で収集・整備したものです。
環境によって List Models API が使えない場合でも、ここをベースに必ず選択肢を出します。

実際に呼び出せる ID は契約・リージョン・廃止予定で変わるため、
一覧にあっても Chat が 400 になる場合は「カスタム…」で ID を指定してください。
"""

from __future__ import annotations

# OpenAI Chat Completions でよく使う ID（スナップショット含む）
_OPENAI: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "chatgpt-4o-latest",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-mini",
    "o3",
    "o3-mini",
    "o3-2025-04-16",
    "o4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.2",
    "gpt-5.2-mini",
)

# Anthropic Messages API（モデル表・エイリアス周辺）
_ANTHROPIC: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-20251001",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
)

# Gemini generate_content 向け（テキスト中心。画像・TTS 専用は除外しがちだが主要 Flash/Pro は網羅）
_GEMINI: tuple[str, ...] = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-preview-05-2025",
    "gemini-2.5-pro-preview-05-2025",
    "gemma-2-9b-it",
    "gemma-2-27b-it",
)

# PLaMo（OpenAI 互換・公式例に基づく代表 ID）
_PLAMO: tuple[str, ...] = (
    "plamo-2.0-prime",
    "plamo-2.0-100b",
    "plamo-2.0-8b",
)

CURATED_CHAT_MODELS: dict[str, tuple[str, ...]] = {
    "openai": _OPENAI,
    "anthropic": _ANTHROPIC,
    "gemini": _GEMINI,
    "plamo": _PLAMO,
}


def curated_chat_models_for_providers(
    available: list[str],
    default_model_by_provider: dict[str, str],
) -> dict[str, list[str]]:
    """各プロバイダの静的一覧。既定モデルを先頭にし、重複を除く。"""
    out: dict[str, list[str]] = {}
    for p in available:
        d = (default_model_by_provider.get(p) or "").strip()
        raw = list(CURATED_CHAT_MODELS.get(p, ()))
        seen: set[str] = set()
        ordered: list[str] = []
        if d:
            ordered.append(d)
            seen.add(d)
        for mid in raw:
            if mid not in seen:
                ordered.append(mid)
                seen.add(mid)
        if not ordered and d:
            ordered = [d]
        out[p] = ordered
    return out
