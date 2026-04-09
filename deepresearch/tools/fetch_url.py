from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from deepresearch.security import detect_prompt_injection


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str | None
    text_md: str
    injection: dict[str, Any]


class UrlFetcher:
    def __init__(self, *, concurrency: int = 20) -> None:
        self._limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True, limits=self._limits)

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch(self, url: str) -> FetchedPage:
        r = await self._client.get(url, headers={"User-Agent": "DeepResearchMAS/0.1"})
        r.raise_for_status()
        html = r.text

        def _parse_html(h: str) -> tuple[str | None, str, dict[str, Any]]:
            soup = BeautifulSoup(h, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else None

            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = soup.get_text("\n")
            inj = detect_prompt_injection(text)

            body = str(soup.body) if soup.body else h
            text_md = md(body, heading_style="ATX")
            return title, text_md, inj

        # HTML parsing / markdownify can be CPU-heavy; offload to a thread to keep the event loop responsive.
        title, text_md, inj = await asyncio.to_thread(_parse_html, html)
        return FetchedPage(url=url, title=title, text_md=text_md, injection=inj)

