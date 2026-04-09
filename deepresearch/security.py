from __future__ import annotations

import re
from typing import Any


_API_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{10,}"),
]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def mask_secrets(text: str) -> str:
    masked = text
    for pat in _API_KEY_PATTERNS:
        masked = pat.sub("[REDACTED_KEY]", masked)
    masked = _EMAIL_RE.sub("[REDACTED_EMAIL]", masked)
    return masked


INJECTION_HINTS = [
    "ignore previous instructions",
    "disregard previous",
    "system prompt",
    "you are chatgpt",
    "tool call",
    "exfiltrate",
]


def detect_prompt_injection(page_text: str) -> dict[str, Any]:
    lower = page_text.lower()
    hits = [h for h in INJECTION_HINTS if h in lower]
    return {"has_hints": bool(hits), "hints": hits[:10]}

