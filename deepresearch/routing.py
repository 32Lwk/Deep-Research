from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(key) or default

        return _ENV_VAR_RE.sub(repl, obj)
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class RoleRoute:
    provider: str
    model: str


@dataclass(frozen=True)
class PlamoDebateConfig:
    enabled: bool
    provider: str
    model: str
    take_first_n: int


@dataclass(frozen=True)
class RoutingConfig:
    roles: dict[str, RoleRoute]
    plamo_debate: PlamoDebateConfig
    debate_agents: dict[str, RoleRoute]
    debate_diversify: bool
    budgets_total_tokens: int
    budgets_split: dict[str, float]
    concurrency: dict[str, int]
    search_defaults: dict[str, Any]
    export: dict[str, Any]
    limits: dict[str, Any]


# 討論の「本編」エージェント（debater_free は PLaMo 等で別扱い）
DEBATE_MAIN_AGENT_IDS: tuple[str, ...] = (
    "debater_general",
    "debater_contrarian",
    "debater_quant",
    "debater_risk",
    "debater_practice",
    "debater_sources",
)


def default_model_for_provider(provider: str) -> str:
    return {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.5-flash",
        "plamo": "plamo-2.0-prime",
    }.get(provider, "gpt-4o-mini")


def resolve_debate_routes(
    rcfg: RoutingConfig,
    *,
    available_providers: set[str],
    debate_fallback: tuple[str, str],
) -> dict[str, tuple[str, str]]:
    """
    各 debater_* に (provider, model) を割り当てる。
    debater_free は plamo_debate が有効ならそちらを優先し、失敗時はフォールバック。
    """
    out: dict[str, tuple[str, str]] = {}
    fb_p, fb_m = debate_fallback
    pd = rcfg.plamo_debate
    if pd.enabled and pd.provider in available_providers:
        out["debater_free"] = (pd.provider, pd.model)
    else:
        out["debater_free"] = (fb_p, fb_m)

    if rcfg.debate_agents:
        for aid in DEBATE_MAIN_AGENT_IDS:
            rr = rcfg.debate_agents.get(aid)
            if rr and rr.provider in available_providers:
                out[aid] = (rr.provider, rr.model)
            else:
                out[aid] = (fb_p, fb_m)
        return out

    if rcfg.debate_diversify:
        pref_order = ("openai", "anthropic", "gemini", "plamo")
        pool = [p for p in pref_order if p in available_providers]
        if pd.enabled and pd.provider in pool:
            main_pool = [p for p in pool if p != pd.provider]
            if not main_pool:
                main_pool = pool
        else:
            main_pool = pool
        if not main_pool:
            main_pool = [fb_p] if fb_p in available_providers else list(available_providers)[:1]
        if not main_pool:
            main_pool = [fb_p]
        for i, aid in enumerate(DEBATE_MAIN_AGENT_IDS):
            p = main_pool[i % len(main_pool)]
            model = fb_m if p == fb_p else default_model_for_provider(p)
            out[aid] = (p, model)
        return out

    for aid in DEBATE_MAIN_AGENT_IDS:
        out[aid] = (fb_p, fb_m)
    return out


def load_routing_config(path: str | Path = "routing.yaml") -> RoutingConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = _expand_env_vars(data)

    roles_raw = (data.get("roles") or {})
    roles: dict[str, RoleRoute] = {}
    for role, rr in roles_raw.items():
        if role == "plamo_debate":
            continue
        if not isinstance(rr, dict) or "provider" not in rr or "model" not in rr:
            continue
        roles[role] = RoleRoute(provider=str(rr["provider"]), model=str(rr["model"]))

    debate_agents_raw = data.get("debate_agents") or {}
    debate_agents: dict[str, RoleRoute] = {}
    if isinstance(debate_agents_raw, dict):
        for aid, rr in debate_agents_raw.items():
            if isinstance(rr, dict) and "provider" in rr and "model" in rr:
                debate_agents[str(aid)] = RoleRoute(provider=str(rr["provider"]), model=str(rr["model"]))

    debate_diversify = bool(data.get("debate_diversify", False))

    pd = data.get("roles", {}).get("plamo_debate") or data.get("plamo_debate") or {}
    if not pd:
        pd = {"enabled": False, "provider": "plamo", "model": "gpt-4.1-mini", "take_first_n": 0}
    plamo_debate = PlamoDebateConfig(
        enabled=bool(pd.get("enabled", False)),
        provider=str(pd.get("provider", "plamo")),
        model=str(pd.get("model", "gpt-4.1-mini")),
        take_first_n=int(pd.get("take_first_n", 0)),
    )

    budgets = data.get("budgets") or {}
    split = budgets.get("split") or {}
    concurrency = data.get("concurrency") or {}
    search_defaults = data.get("search_defaults") or {}
    export = data.get("export") or {}
    limits = data.get("limits") or {}

    return RoutingConfig(
        roles=roles,
        plamo_debate=plamo_debate,
        debate_agents=debate_agents,
        debate_diversify=debate_diversify,
        budgets_total_tokens=int(budgets.get("total_tokens", 300_000)),
        budgets_split={k: float(v) for k, v in split.items()},
        concurrency={k: int(v) for k, v in concurrency.items()},
        search_defaults=search_defaults,
        export=export,
        limits=limits,
    )

