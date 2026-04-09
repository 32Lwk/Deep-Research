from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import networkx as nx

from deepresearch.providers import Message, ProviderRegistry
from deepresearch.security import mask_secrets
from deepresearch.storage import RunPaths, append_jsonl, event, write_json, write_text
from deepresearch.tools.fetch_url import FetchedPage, UrlFetcher
from deepresearch.tools.search_serpapi import SerpApiClient
from deepresearch.export_pdf import md_to_pdf, pandoc_available


@dataclass(frozen=True)
class RunConfig:
    topic: str
    seed_urls: list[str]
    token_budget_total: int
    token_budget_split: dict[str, float]
    local_docs: list[dict[str, Any]] | None = None
    max_rounds: int = 3
    max_sources: int = 12
    search_hl_gl: list[tuple[str, str]] = (("ja", "jp"), ("en", "us"))


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def budget_for(role: str, total: int, split: dict[str, float]) -> int:
    w = _clamp01(split.get(role, 0.0))
    return max(256, int(total * w))


DEFAULT_SPLIT = {"io": 0.40, "synthesis": 0.30, "debate": 0.20, "verify": 0.10}


class Orchestrator:
    def __init__(
        self,
        *,
        run_id: str,
        paths: RunPaths,
        registry: ProviderRegistry,
        serpapi_key: str | None,
        llm_models: dict[str, tuple[str, str]],
        debate_alt: tuple[str, str] | None = None,
        debate_alt_take_first_n: int = 0,
        debate_routes: dict[str, tuple[str, str]] | None = None,
        export_cfg: dict[str, Any] | None = None,
        limits_cfg: dict[str, Any] | None = None,
        url_fetcher: UrlFetcher,
        search_client: SerpApiClient | None,
        emit_event: callable[[dict[str, Any]], None],
    ) -> None:
        self.run_id = run_id
        self.paths = paths
        self.registry = registry
        self.serpapi_key = serpapi_key
        self.llm_models = llm_models
        self.debate_alt = debate_alt
        self.debate_alt_take_first_n = debate_alt_take_first_n
        self.debate_routes = debate_routes or {}
        self.export_cfg = export_cfg or {}
        self.limits_cfg = limits_cfg or {}
        self.url_fetcher = url_fetcher
        self.search_client = search_client
        self.emit_event = emit_event

        self.graph = nx.DiGraph()

    def _agent_route_snapshot(self) -> dict[str, dict[str, str]]:
        snap: dict[str, dict[str, str]] = {}
        for k in ("brief", "synthesis", "verify", "panorama"):
            if k in self.llm_models:
                p, m = self.llm_models[k]
                snap[k] = {"provider": p, "model": m}
        for aid, (p, m) in self.debate_routes.items():
            snap[aid] = {"provider": p, "model": m}
        return snap

    def _cap_output_tokens(self, requested: int, *, provider: str, model: str) -> int:
        # Global per-call cap (prevents 400 due to overlarge max_tokens)
        per_call_cap = int(self.limits_cfg.get("max_output_tokens_per_call", 4096))
        per_provider = self.limits_cfg.get("per_provider") or {}
        if isinstance(per_provider, dict) and provider in per_provider:
            try:
                per_call_cap = int(per_provider[provider])
            except Exception:
                pass
        capped = min(requested, per_call_cap) if per_call_cap > 0 else requested

        # Best-effort model caps for common OpenAI models (completion tokens)
        # If unknown, rely on per_call_cap.
        if provider in ("openai", "plamo"):
            if "gpt-4.1-mini" in model:
                capped = min(capped, 32768)
            elif "gpt-4.1" in model:
                capped = min(capped, 32768)
            elif "gpt-4o-mini" in model:
                capped = min(capped, 16384)
            elif model.startswith("gpt-4o"):
                capped = min(capped, 16384)
        return max(16, int(capped))

    async def _llm_text(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
        role: str | None = None,
        agent_id: str | None = None,
        phase: str | None = None,
    ) -> str:
        p = self.registry.get(provider)
        msgs = [Message(role="system", content=system), Message(role="user", content=user)]
        max_tokens = self._cap_output_tokens(max_tokens, provider=provider, model=model)
        self.emit_event(
            event(
                self.run_id,
                "llm_call_start",
                role=role,
                agent_id=agent_id,
                phase=phase,
                provider=provider,
                model=model,
                max_output_tokens=max_tokens,
            )
        )
        try:
            out = await p.complete(model=model, messages=msgs, temperature=0.2, max_output_tokens=max_tokens)
            return out
        except Exception as e:
            # Best-effort fallback for OpenAI-like model naming issues
            self.emit_event(event(self.run_id, "llm_error", provider=provider, model=model, error=str(e)))
            fallback_models = []
            if provider in ("openai", "plamo"):
                fallback_models = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4o"]
            for fb in fallback_models:
                if fb == model:
                    continue
                try:
                    self.emit_event(event(self.run_id, "llm_retry", provider=provider, model=fb))
                    out = await p.complete(model=fb, messages=msgs, temperature=0.2, max_output_tokens=max_tokens)
                    return out
                except Exception as e2:
                    self.emit_event(event(self.run_id, "llm_error", provider=provider, model=fb, error=str(e2)))
            raise

    async def _llm_stream(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        *,
        max_tokens: int,
        role: str | None = None,
        agent_id: str | None = None,
        phase: str | None = None,
    ) -> AsyncIterator[str]:
        p = self.registry.get(provider)
        msgs = [Message(role="system", content=system), Message(role="user", content=user)]
        max_tokens = self._cap_output_tokens(max_tokens, provider=provider, model=model)
        self.emit_event(
            event(
                self.run_id,
                "llm_call_start",
                role=role,
                agent_id=agent_id,
                phase=phase,
                provider=provider,
                model=model,
                max_output_tokens=max_tokens,
            )
        )
        async for ch in p.stream(model=model, messages=msgs, temperature=0.2, max_output_tokens=max_tokens):
            if ch.text:
                yield ch.text

    async def build_brief(self, topic: str, *, max_tokens: int) -> dict[str, Any]:
        provider, model = self.llm_models["brief"]
        system = (
            "あなたは調査の司令塔補助です。ユーザーの1文質問から、調査ブリーフをJSONで作成してください。"
            "必ず日本語で。曖昧さがある場合は合理的に仮定し、assumptionsに列挙する。"
            "質問が「あなたたちは」「このシステムは」「DeepResearchは」「Arenaの各AIは」など、"
            "このリサーチ用アプリやマルチエージェント構成そのものについてのメタ質問のときは、"
            "調査対象を「本アプリのパイプライン（ブリーフ→討論→検索→統合→検証）と接続されるLLMプロバイダ」に置き、"
            "利用者が今話しているチャットUI（例: ChatGPT）の製品説明と混同しないこと。"
            "その場合 initial_search_queries に特定ベンダー名（OpenAI等）を安易に入れない。"
        )
        user = f"""質問:\n{topic}\n\n出力JSONスキーマ:\n{{
  "objective": "...",
  "background": "...",
  "key_questions": ["..."],
  "out_of_scope": ["..."],
  "initial_search_queries": {{"ja": ["..."], "en": ["..."]}},
  "assumptions": ["..."]
}}"""
        chunks: list[str] = []
        async for t in self._llm_stream(
            provider,
            model,
            system,
            user,
            max_tokens=max_tokens,
            role="brief",
            agent_id="brief",
            phase="brief",
        ):
            chunks.append(t)
            self.emit_event(event(self.run_id, "brief_chunk", agent_id="brief", provider=provider, model=model, text=mask_secrets(t)))
        text = "".join(chunks)
        self.emit_event(event(self.run_id, "brief_raw", agent_id="brief", provider=provider, model=model, text=mask_secrets(text)))
        # best-effort JSON extraction
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {
                "objective": topic,
                "background": "",
                "key_questions": [topic],
                "out_of_scope": [],
                "initial_search_queries": {"ja": [topic], "en": [topic]},
                "assumptions": ["モデル出力がJSONとして解析できなかったためフォールバック"],
            }
        import json

        return json.loads(m.group(0))

    async def debate_multi(self, brief: dict[str, Any], *, max_tokens_total: int) -> list[dict[str, Any]]:
        """
        10体規模のMVPとして、複数の「自由討論」役を並列実行する。
        Orchestrator/BriefBuilder/Synthesizer/Skeptic/Panorama/Security を含めると全体で概ね10役になる想定。
        """
        provider, model = self.llm_models["debate"]
        agents = [
            ("debater_general", "広く観点を洗い出す"),
            ("debater_contrarian", "反対意見・反証・弱点を探す"),
            ("debater_quant", "数値・指標・比較軸を提案する"),
            ("debater_risk", "リスク・限界・失敗要因を挙げる"),
            ("debater_practice", "実務/導入観点（運用・コスト・体制）を挙げる"),
            ("debater_sources", "一次ソース候補（公式/論文/統計）を提案する"),
            # PLaMo等、別プロバイダを「自由発言1人」として割り当てる枠
            ("debater_free", "自由発言（発散/突飛な仮説/別視点）を出す"),
        ]

        per_agent = max(256, max_tokens_total // max(1, len(agents)))

        async def one(agent_id: str, focus: str, *, provider_: str, model_: str) -> dict[str, Any]:
            system = (
                f"あなたは自由討論エージェント（{agent_id}）です。フォーカス: {focus}。"
                "ブリーフを読み、重要な観点、反対意見、調査すべき論点を箇条書きで出してください。"
                "出力は箇条書きのみ。"
                "この呼び出しだけで完結させ、他エージェントの役割に踏み込まない。"
            )
            user = f"ブリーフ:\n{brief}\n\n追加すべき観点を出してください。"
            self.emit_event(event(self.run_id, "debate_agent_start", agent_id=agent_id, provider=provider_, model=model_))
            text = await self._llm_text(
                provider_,
                model_,
                system,
                user,
                max_tokens=per_agent,
                role="debate",
                agent_id=agent_id,
                phase="debate",
            )
            self.emit_event(event(self.run_id, "debate_notes", agent_id=agent_id, text=mask_secrets(text)))
            lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
            return {"agent_id": agent_id, "focus": focus, "lines": lines[:30], "raw": text}

        tasks = []
        for (a, f) in agents:
            if self.debate_routes:
                p_, m_ = self.debate_routes.get(a, (provider, model))
            elif self.debate_alt and a == "debater_free":
                p_, m_ = self.debate_alt
            else:
                p_, m_ = provider, model
            tasks.append(one(a, f, provider_=p_, model_=m_))

        # 1エージェント失敗で全体が止まらないようにする（PLaMoが落ちていても会話は継続）
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, Exception):
                self.emit_event(event(self.run_id, "debate_agent_error", error=str(r)))
                continue
            out.append(r)
        return out

    async def panorama_stop_suggestion(
        self,
        *,
        round_idx: int,
        total_rounds: int,
        known_urls: list[str],
        new_urls_added: int,
        verify_notes_md: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        provider, model = self.llm_models.get("panorama", self.llm_models["verify"])
        system = (
            "あなたは俯瞰（Panorama）エージェントです。介入は最小限にし、調査を続けるべきか停止すべきかを提案してください。"
            "判断基準: 新しい根拠が増えているか、検証メモで致命的な穴が残るか、コストに見合うか。"
            "出力はJSONのみ。"
        )
        user = f"""状況:
- round: {round_idx+1}/{total_rounds}
- known_urls: {len(known_urls)}
- new_urls_added_this_round: {new_urls_added}

検証メモ:
{verify_notes_md}

出力JSON:
{{
  \"should_stop\": true/false,
  \"reason\": \"...\",
  \"next_search_queries\": {{\"ja\": [\"...\"], \"en\": [\"...\"]}}
}}"""
        text = await self._llm_text(
            provider,
            model,
            system,
            user,
            max_tokens=max_tokens,
            role="panorama",
            agent_id="panorama",
            phase=f"round_{round_idx+1}",
        )
        self.emit_event(event(self.run_id, "panorama_raw", agent_id="panorama", provider=provider, model=model, text=mask_secrets(text)))
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {"should_stop": False, "reason": "panorama_json_parse_failed", "next_search_queries": {"ja": [], "en": []}}
        import json

        return json.loads(m.group(0))

    async def search(self, queries: dict[str, list[str]], *, max_results_per_query: int = 8) -> list[dict[str, Any]]:
        if not self.search_client:
            self.emit_event(event(self.run_id, "search_skipped", reason="SERPAPI_API_KEY not set"))
            return []
        results: list[dict[str, Any]] = []
        for (hl, gl) in [("ja", "jp"), ("en", "us")]:
            for q in queries.get(hl, [])[:3]:
                self.emit_event(event(self.run_id, "search_query", q=q, hl=hl, gl=gl))
                try:
                    items = await self.search_client.search(q, hl=hl, gl=gl, num=max_results_per_query)
                    results.extend(items)
                except Exception as e:
                    self.emit_event(event(self.run_id, "search_error", q=q, error=str(e)))
        # de-dup by link
        seen: set[str] = set()
        dedup: list[dict[str, Any]] = []
        for r in results:
            link = r.get("link")
            if not link or link in seen:
                continue
            seen.add(link)
            dedup.append(r)
        return dedup

    async def fetch_sources(self, urls: list[str]) -> list[FetchedPage]:
        sem = asyncio.Semaphore(20)
        out: list[FetchedPage] = []

        async def one(u: str) -> None:
            async with sem:
                try:
                    self.emit_event(event(self.run_id, "fetch_start", url=u))
                    page = await self.url_fetcher.fetch(u)
                    self.emit_event(
                        event(
                            self.run_id,
                            "fetch_done",
                            url=u,
                            title=page.title,
                            injection=page.injection,
                        )
                    )
                    out.append(page)
                except Exception as e:
                    self.emit_event(event(self.run_id, "fetch_error", url=u, error=str(e)))

        await asyncio.gather(*[one(u) for u in urls])
        return out

    async def synthesize_report_stream(
        self, *, brief: dict[str, Any], pages: list[FetchedPage], debate_notes: list[str], max_tokens: int
    ) -> AsyncIterator[str]:
        provider, model = self.llm_models["synthesis"]
        system = (
            "あなたはDeepResearchの統合エージェントです。必ず日本語。"
            "章立ては「結論/要点/背景/調査/分析/反証/提案/参考文献」。"
            "重要主張には番号参照[1][2]を付け、最後に参考文献一覧をURL付きで並べる。"
            "推論は推論と明示する。引用はページ抜粋を短く使い、過剰な断定を避ける。"
            "ソースが空、または根拠URLがないときは、ウェブを調べたかのような断定や具体的URLをでっち上げない。"
            "メタ質問（「あなたたちについて」等）では、本アプリのマルチエージェント調査パイプラインと、"
            "討論で得た観点を説明し、特定のチャット製品（ChatGPT単体など）をこのアプリの実体と決めつけない。"
        )
        sources = []
        for i, p in enumerate(pages, start=1):
            sources.append(f"[{i}] {p.title or ''}\nURL: {p.url}\n抜粋(要約用):\n{p.text_md[:2000]}")
        sources_block = "\n\n".join(sources)
        user = (
            f"ブリーフ:\n{brief}\n\n自由討論メモ:\n{debate_notes}\n\nソース:\n\n{sources_block}\n\n"
            "上記をもとにレポートを作成してください。"
        )
        async for t in self._llm_stream(
            provider,
            model,
            system,
            user,
            max_tokens=max_tokens,
            role="synthesis",
            agent_id="synthesis",
            phase="synthesis",
        ):
            yield t

    async def verify(self, report_md: str, *, max_tokens: int) -> str:
        provider, model = self.llm_models["verify"]
        system = (
            "あなたは検証（Skeptic）エージェントです。レポートの矛盾、根拠不足、重要主張の未引用を指摘し、"
            "追加で必要な調査クエリ（ja/en）を提案してください。出力はMarkdown。"
        )
        user = f"レポート:\n\n{report_md}\n\n指摘と追加クエリを出してください。"
        chunks: list[str] = []
        async for t in self._llm_stream(
            provider,
            model,
            system,
            user,
            max_tokens=max_tokens,
            role="verify",
            agent_id="verify",
            phase="verify",
        ):
            chunks.append(t)
            self.emit_event(event(self.run_id, "verify_chunk", agent_id="verify", provider=provider, model=model, text=mask_secrets(t)))
        return "".join(chunks).strip()

    async def run(self, cfg: RunConfig) -> dict[str, Any]:
        self.emit_event(
            event(
                self.run_id,
                "run_start",
                topic=cfg.topic,
                seed_urls=cfg.seed_urls,
                agent_routes=self._agent_route_snapshot(),
            )
        )

        brief = await self.build_brief(
            cfg.topic, max_tokens=budget_for("io", cfg.token_budget_total, cfg.token_budget_split)
        )

        debate_runs = await self.debate_multi(
            brief, max_tokens_total=budget_for("debate", cfg.token_budget_total, cfg.token_budget_split)
        )
        debate_notes: list[str] = []
        for r in debate_runs:
            debate_notes.extend(r["lines"])
        # de-dup
        seen_note: set[str] = set()
        debate_notes = [n for n in debate_notes if not (n in seen_note or seen_note.add(n))]

        known_urls: list[str] = []
        pages_by_url: dict[str, FetchedPage] = {}
        search_results_all: list[dict[str, Any]] = []

        next_queries = brief.get("initial_search_queries") or {"ja": [cfg.topic], "en": [cfg.topic]}
        verify_md = ""
        report_md = ""

        # local docs (uploaded files) as evidence pages
        if cfg.local_docs:
            for i, d in enumerate(cfg.local_docs, start=1):
                pseudo_url = f"file://{d.get('filename','doc'+str(i))}"
                pages_by_url[pseudo_url] = FetchedPage(
                    url=pseudo_url,
                    title=d.get("filename"),
                    text_md=d.get("text_md") or "",
                    injection={"has_hints": False, "hints": [], "note": "local_doc"},
                )
                known_urls.append(pseudo_url)

        for round_idx in range(cfg.max_rounds):
            self.emit_event(event(self.run_id, "round_start", round=round_idx + 1))

            search_results = await self.search(next_queries)
            search_results_all.extend(search_results)

            candidate_urls = cfg.seed_urls + [r["link"] for r in search_results]
            dedup_urls: list[str] = []
            seen_url: set[str] = set(known_urls)
            for u in candidate_urls:
                if u in seen_url:
                    continue
                seen_url.add(u)
                dedup_urls.append(u)

            remaining = max(0, cfg.max_sources - len(known_urls))
            add_urls = dedup_urls[:remaining]
            known_urls.extend(add_urls)

            if add_urls:
                new_pages = await self.fetch_sources(add_urls)
                for p in new_pages:
                    pages_by_url[p.url] = p

            pages = list(pages_by_url.values())
            write_json(
                self.paths.evidence_json,
                {
                    "brief": brief,
                    "debate_routes": {k: {"provider": a, "model": b} for k, (a, b) in self.debate_routes.items()},
                    "debate_runs": [
                        {"agent_id": r["agent_id"], "focus": r["focus"], "lines": r["lines"]} for r in debate_runs
                    ],
                    "debate_notes": debate_notes,
                    "search_results": search_results_all,
                    "pages": [
                        {"url": p.url, "title": p.title, "injection": p.injection, "text_md": p.text_md[:20000]}
                        for p in pages
                    ],
                },
            )

            self.emit_event(event(self.run_id, "synthesis_start", round=round_idx + 1, sources=len(pages)))
            chunks: list[str] = []
            async for t in self.synthesize_report_stream(
                brief=brief,
                pages=pages,
                debate_notes=debate_notes,
                max_tokens=budget_for("synthesis", cfg.token_budget_total, cfg.token_budget_split),
            ):
                chunks.append(t)
                self.emit_event(event(self.run_id, "synthesis_chunk", text=mask_secrets(t)))
            report_md = "".join(chunks).strip()
            write_text(self.paths.report_md, report_md + "\n")

            verify_md = await self.verify(
                report_md, max_tokens=budget_for("verify", cfg.token_budget_total, cfg.token_budget_split)
            )
            self.emit_event(event(self.run_id, "verify_notes", round=round_idx + 1, text=mask_secrets(verify_md)))

            pano = await self.panorama_stop_suggestion(
                round_idx=round_idx,
                total_rounds=cfg.max_rounds,
                known_urls=known_urls,
                new_urls_added=len(add_urls),
                verify_notes_md=verify_md,
                max_tokens=max(256, budget_for("verify", cfg.token_budget_total, cfg.token_budget_split) // 2),
            )
            self.emit_event(event(self.run_id, "panorama_suggestion", suggestion=pano))

            if pano.get("should_stop") is True:
                self.emit_event(event(self.run_id, "round_stop", round=round_idx + 1, reason=pano.get("reason")))
                break

            next_queries = (pano.get("next_search_queries") or {"ja": [], "en": []}) if isinstance(pano, dict) else {"ja": [], "en": []}
            if not next_queries.get("ja") and not next_queries.get("en"):
                next_queries = {"ja": [cfg.topic], "en": [cfg.topic]}

        pages = list(pages_by_url.values())
        report_obj = {
            "run_id": self.run_id,
            "topic": cfg.topic,
            "brief": brief,
            "debate_notes": debate_notes,
            "sources": [{"url": p.url, "title": p.title, "injection": p.injection} for p in pages],
            "report_md": report_md,
            "verify_notes_md": verify_md,
        }
        write_json(self.paths.report_json, report_obj)

        # PDF export (best-effort)
        try:
            pdf_path = self.paths.run_dir / "report.pdf"
            if not bool(self.export_cfg.get("pdf_enabled", True)):
                self.emit_event(event(self.run_id, "pdf_skipped", reason="disabled"))
            else:
                pandoc_path = str(self.export_cfg.get("pandoc_path") or "pandoc")
                if pandoc_available(pandoc_path):
                    # NOTE: pandoc invocation is blocking; run in a thread so the web server stays responsive.
                    await asyncio.to_thread(
                        md_to_pdf,
                        self.paths.report_md,
                        pdf_path,
                        pandoc_path=pandoc_path,
                        pdf_engine=self.export_cfg.get("pdf_engine"),
                        mainfont=self.export_cfg.get("mainfont"),
                        extra_args=self.export_cfg.get("extra_args") or None,
                    )
                    self.emit_event(event(self.run_id, "pdf_done", path=str(pdf_path)))
                else:
                    self.emit_event(event(self.run_id, "pdf_skipped", reason="pandoc_not_found"))
        except Exception as e:
            self.emit_event(event(self.run_id, "pdf_error", error=str(e)))

        # Graph (simple)
        g = {"nodes": [], "edges": []}
        for p in pages:
            g["nodes"].append({"id": p.url, "type": "evidence", "label": p.title or p.url})
        g["nodes"].append({"id": "report", "type": "report", "label": "report"})
        for p in pages:
            g["edges"].append({"source": p.url, "target": "report", "type": "used_in"})
        write_json(self.paths.graph_json, g)

        self.emit_event(event(self.run_id, "run_done"))
        return report_obj

