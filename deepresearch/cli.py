from __future__ import annotations

import argparse
import asyncio
import os

from deepresearch.core import DEFAULT_SPLIT, Orchestrator, RunConfig, new_run_id
from deepresearch.providers import ProviderRegistry
from deepresearch.routing import load_routing_config, resolve_debate_routes
from deepresearch.settings import Settings
from deepresearch.storage import append_jsonl, event, init_run_dir
from deepresearch.tools.fetch_url import UrlFetcher
from deepresearch.tools.search_serpapi import SerpApiClient


def _default_models(reg: ProviderRegistry) -> dict[str, tuple[str, str]]:
    # provider preference order; pick first available
    prefs = reg.available()
    if not prefs:
        raise SystemExit("No LLM providers configured. Set API keys in .env.")
    p0 = prefs[0]
    # MVP defaults; user can edit later
    return {
        "brief": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "debate": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "synthesis": (p0, "gpt-4.1" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "verify": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "panorama": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
    }


async def _run(topic: str, seed_urls: list[str], token_budget_total: int) -> str:
    s = Settings()
    reg = ProviderRegistry.from_settings(s)
    rcfg = load_routing_config()
    run_id = new_run_id()
    paths = init_run_dir(s.RUNS_DIR, run_id)

    def emit(ev):
        append_jsonl(paths.events_jsonl, ev)

    search_conc = rcfg.concurrency.get("search", s.SEARCH_CONCURRENCY)
    url_conc = rcfg.concurrency.get("url_fetch", s.URL_FETCH_CONCURRENCY)
    search_client = SerpApiClient(s.SERPAPI_API_KEY, concurrency=search_conc) if s.SERPAPI_API_KEY else None
    url_fetcher = UrlFetcher(concurrency=url_conc)

    _roles = frozenset({"brief", "debate", "synthesis", "verify", "panorama"})
    role_models = {k: (v.provider, v.model) for k, v in rcfg.roles.items() if k in _roles}
    if not role_models:
        role_models = _default_models(reg)

    avail = set(reg.available())
    fb = role_models.get("debate") or role_models.get("brief")
    if not fb:
        p0 = sorted(avail)[0] if avail else "openai"
        fb = (p0, "gpt-4o-mini" if p0 == "openai" else "gemini-2.5-flash")
    debate_routes = resolve_debate_routes(rcfg, available_providers=avail, debate_fallback=fb)

    orch = Orchestrator(
        run_id=run_id,
        paths=paths,
        registry=reg,
        serpapi_key=s.SERPAPI_API_KEY,
        llm_models=role_models,
        debate_alt=(rcfg.plamo_debate.provider, rcfg.plamo_debate.model) if rcfg.plamo_debate.enabled else None,
        debate_alt_take_first_n=rcfg.plamo_debate.take_first_n if rcfg.plamo_debate.enabled else 0,
        debate_routes=debate_routes,
        export_cfg=rcfg.export,
        limits_cfg=rcfg.limits,
        url_fetcher=url_fetcher,
        search_client=search_client,
        emit_event=emit,
    )

    total = token_budget_total or rcfg.budgets_total_tokens
    split = rcfg.budgets_split or DEFAULT_SPLIT
    cfg = RunConfig(topic=topic, seed_urls=seed_urls, token_budget_total=total, token_budget_split=split)
    await orch.run(cfg)

    if search_client:
        await search_client.close()
    await url_fetcher.close()
    return str(paths.run_dir)


def main() -> None:
    p = argparse.ArgumentParser(prog="deepresearch")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run deep research")
    run.add_argument("topic", type=str)
    run.add_argument("--seed-url", action="append", default=[], dest="seed_urls")
    run.add_argument("--token-budget", type=int, default=300_000, dest="token_budget")

    args = p.parse_args()

    if args.cmd == "run":
        out_dir = asyncio.run(_run(args.topic, args.seed_urls, args.token_budget))
        print(out_dir)


if __name__ == "__main__":
    main()

