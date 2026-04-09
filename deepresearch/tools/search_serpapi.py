from __future__ import annotations

from typing import Any

import httpx


class SerpApiClient:
    def __init__(self, api_key: str, *, concurrency: int = 4) -> None:
        self.api_key = api_key
        self._limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0), limits=self._limits)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(self, q: str, *, hl: str = "ja", gl: str = "jp", num: int = 10) -> list[dict[str, Any]]:
        params = {
            "engine": "google",
            "q": q,
            "api_key": self.api_key,
            "hl": hl,
            "gl": gl,
            "num": num,
        }
        r = await self._client.get("https://serpapi.com/search.json", params=params)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results") or []
        out: list[dict[str, Any]] = []
        for item in organic:
            link = item.get("link")
            if not link:
                continue
            out.append(
                {
                    "title": item.get("title"),
                    "link": link,
                    "snippet": item.get("snippet"),
                    "position": item.get("position"),
                    "source": "serpapi",
                    "query": q,
                }
            )
        return out

