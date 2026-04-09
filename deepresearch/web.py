from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from deepresearch.core import DEFAULT_SPLIT, Orchestrator, RunConfig, new_run_id
from deepresearch.providers import ProviderRegistry
from deepresearch.routing import load_routing_config, resolve_debate_routes
from deepresearch.settings import Settings
from deepresearch.storage import RunPaths, append_jsonl, init_run_dir, read_json
from deepresearch.tools.fetch_url import UrlFetcher
from deepresearch.tools.extract_files import extract_md, extract_pdf_to_md
from deepresearch.tools.search_serpapi import SerpApiClient


@dataclass
class RunState:
    run_id: str
    paths: RunPaths
    queue: "asyncio.Queue[dict[str, Any]]"
    task: asyncio.Task | None = None
    topic: str | None = None
    seed_urls: list[str] | None = None


app = FastAPI()
settings = Settings()
registry = ProviderRegistry.from_settings(settings)
rcfg = load_routing_config()

run_states: dict[str, RunState] = {}

_ORCH_ROLES = frozenset({"brief", "debate", "synthesis", "verify", "panorama"})


def _default_models() -> dict[str, tuple[str, str]]:
    role_models = {k: (v.provider, v.model) for k, v in rcfg.roles.items() if k in _ORCH_ROLES}
    if role_models:
        return role_models
    avail = registry.available()
    if not avail:
        return {}
    p0 = avail[0]
    return {
        "brief": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "debate": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "synthesis": (p0, "gpt-4.1" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "verify": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
        "panorama": (p0, "gpt-4.1-mini" if p0 == "openai" else "claude-3-5-sonnet-latest"),
    }


def _debate_routes_for_run(llm_models: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    avail = set(registry.available())
    fb = llm_models.get("debate") or llm_models.get("brief")
    if not fb:
        p0 = sorted(avail)[0] if avail else "openai"
        fb = (p0, "gpt-4o-mini" if p0 == "openai" else "gemini-2.5-flash")
    return resolve_debate_routes(rcfg, available_providers=avail, debate_fallback=fb)


def routing_preview_payload() -> dict[str, Any]:
    avail = sorted(registry.available())
    llm = _default_models()
    fb = llm.get("debate") or llm.get("brief") or ("gemini", "gemini-2.5-flash")
    dr = resolve_debate_routes(rcfg, available_providers=set(avail), debate_fallback=fb)
    return {
        "roles": {k: {"provider": p, "model": m} for k, (p, m) in llm.items()},
        "debate_routes": {k: {"provider": p, "model": m} for k, (p, m) in dr.items()},
        "debate_diversify": rcfg.debate_diversify,
        "available_providers": avail,
    }


INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>DeepResearch MAS</title>
    <style>
      body { font-family: ui-sans-serif, system-ui; margin: 20px; }
      textarea { width: 100%; height: 90px; }
      pre { white-space: pre-wrap; background: #111; color: #eee; padding: 12px; border-radius: 8px; }
      .row { display: flex; gap: 12px; }
      .col { flex: 1; }
      input[type=text] { width: 100%; }
    </style>
  </head>
  <body>
    <h2>DeepResearch MAS (MVP)</h2>
    <div class="row">
      <div class="col">
        <label>Topic</label>
        <textarea id="topic"></textarea>
        <label>Seed URLs (one per line)</label>
        <textarea id="seedUrls"></textarea>
        <button id="start">Start</button>
        <p id="meta"></p>
      </div>
      <div class="col">
        <label>Streaming events</label>
        <pre id="log"></pre>
      </div>
    </div>

    <script>
      const log = document.getElementById('log');
      const meta = document.getElementById('meta');
      document.getElementById('start').onclick = async () => {
        log.textContent = '';
        meta.textContent = '';
        const topic = document.getElementById('topic').value.trim();
        const seedUrls = document.getElementById('seedUrls').value.split('\\n').map(s => s.trim()).filter(Boolean);
        const resp = await fetch('/api/runs', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({})});
        const data = await resp.json();
        meta.textContent = `run_id=${data.run_id}`;
        const ws = new WebSocket(`ws://${location.host}/api/runs/${data.run_id}/events`);
        ws.onmessage = (ev) => {
          log.textContent += ev.data + "\\n";
        };
        ws.onclose = () => {
          log.textContent += "\\n[closed]\\n";
        };

        await fetch(`/api/runs/${data.run_id}/start`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({topic, seed_urls: seedUrls})});
      };
    </script>
  </body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/providers")
def providers() -> dict[str, Any]:
    return {"available": registry.available()}


@app.post("/api/runs")
async def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    if not registry.available():
        return {"error": "No providers configured"}

    run_id = new_run_id()
    paths = init_run_dir(settings.RUNS_DIR, run_id)
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    st = RunState(run_id=run_id, paths=paths, queue=q, task=None, topic=None, seed_urls=[])
    run_states[run_id] = st

    # runner is started by /start after metadata/files are ready
    st.task = None
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    st = run_states.get(run_id)
    if not st:
        return {"error": "unknown run_id"}
    if st.task and not st.task.done():
        return {"error": "already running"}

    topic = (payload.get("topic") or "").strip()
    seed_urls = payload.get("seed_urls") or []
    st.topic = topic
    st.seed_urls = seed_urls

    # load extracted local docs if present
    local_docs: list[dict[str, Any]] = []
    evidence_path = st.paths.run_dir / "uploaded_evidence.json"
    if evidence_path.exists():
        try:
            local_docs = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception:
            local_docs = []

    async def runner() -> None:
        search_client: SerpApiClient | None = None
        url_fetcher: UrlFetcher | None = None
        try:
            # Create network clients inside the background task so /start can respond immediately.
            search_conc = rcfg.concurrency.get("search", settings.SEARCH_CONCURRENCY)
            url_conc = rcfg.concurrency.get("url_fetch", settings.URL_FETCH_CONCURRENCY)
            search_client = (
                SerpApiClient(settings.SERPAPI_API_KEY, concurrency=search_conc) if settings.SERPAPI_API_KEY else None
            )
            url_fetcher = UrlFetcher(concurrency=url_conc)

            def emit(ev: dict[str, Any]) -> None:
                append_jsonl(st.paths.events_jsonl, ev)
                st.queue.put_nowait(ev)

            llm_models = _default_models()
            debate_routes = _debate_routes_for_run(llm_models)
            orch = Orchestrator(
                run_id=run_id,
                paths=st.paths,
                registry=registry,
                serpapi_key=settings.SERPAPI_API_KEY,
                llm_models=llm_models,
                debate_alt=(rcfg.plamo_debate.provider, rcfg.plamo_debate.model) if rcfg.plamo_debate.enabled else None,
                debate_alt_take_first_n=rcfg.plamo_debate.take_first_n if rcfg.plamo_debate.enabled else 0,
                debate_routes=debate_routes,
                export_cfg=rcfg.export,
                limits_cfg=rcfg.limits,
                url_fetcher=url_fetcher,
                search_client=search_client,
                emit_event=emit,
            )
            cfg = RunConfig(
                topic=topic,
                seed_urls=seed_urls,
                local_docs=local_docs,
                token_budget_total=int(rcfg.budgets_total_tokens or 300_000),
                token_budget_split=rcfg.budgets_split or DEFAULT_SPLIT,
            )
            await orch.run(cfg)
        except asyncio.CancelledError:
            append_jsonl(st.paths.events_jsonl, {"type": "run_cancelled", "run_id": run_id})
            st.queue.put_nowait({"type": "run_cancelled", "run_id": run_id})
            raise
        except Exception as e:
            append_jsonl(st.paths.events_jsonl, {"type": "run_error", "run_id": run_id, "error": str(e)})
            st.queue.put_nowait({"type": "run_error", "run_id": run_id, "error": str(e)})
        finally:
            if search_client:
                await search_client.close()
            if url_fetcher:
                await url_fetcher.close()

    st.task = asyncio.create_task(runner())
    return {"run_id": run_id, "status": "started"}


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    st = run_states.get(run_id)
    if not st or not st.task:
        return {"error": "unknown run_id"}
    if st.task.done():
        return {"run_id": run_id, "status": "already_done"}
    st.task.cancel()
    return {"run_id": run_id, "status": "cancelling"}


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str) -> dict[str, Any]:
    st = run_states.get(run_id)
    if not st:
        return {"error": "unknown run_id"}
    if not st.task:
        return {"run_id": run_id, "status": "created"}
    if st.task.cancelled():
        return {"run_id": run_id, "status": "cancelled"}
    if st.task.done():
        exc = st.task.exception()
        return {"run_id": run_id, "status": "done", "error": str(exc) if exc else None}
    return {"run_id": run_id, "status": "running"}


@app.post("/api/runs/{run_id}/files")
async def upload_files(run_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    st = run_states.get(run_id)
    if not st:
        return {"error": "unknown run_id"}
    uploads_dir = st.paths.run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[dict[str, Any]] = []
    for f in files:
        dst = uploads_dir / f.filename
        content = await f.read()
        dst.write_bytes(content)
        if dst.suffix.lower() == ".pdf":
            md = extract_pdf_to_md(dst)
            extracted.append({"filename": f.filename, "kind": "pdf", "text_md": md[:20000]})
        elif dst.suffix.lower() in [".md", ".markdown"]:
            md = extract_md(dst)
            extracted.append({"filename": f.filename, "kind": "md", "text_md": md[:20000]})
        else:
            extracted.append({"filename": f.filename, "kind": "unknown", "text_md": ""})

    # Save as local evidence extension (best-effort)
    evidence_path = st.paths.run_dir / "uploaded_evidence.json"
    evidence_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_id": run_id, "saved": [e["filename"] for e in extracted]}


@app.get("/api/settings/routing-preview")
def get_routing_preview() -> dict[str, Any]:
    return routing_preview_payload()


@app.get("/api/settings/routing.yaml")
def get_routing_yaml() -> PlainTextResponse:
    p = Path("routing.yaml")
    return PlainTextResponse(p.read_text(encoding="utf-8"))


@app.put("/api/settings/routing.yaml")
def put_routing_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    global rcfg
    text = payload.get("text")
    if not isinstance(text, str):
        return {"error": "text required"}
    Path("routing.yaml").write_text(text, encoding="utf-8")
    rcfg = load_routing_config()
    return {"status": "ok"}


@app.websocket("/api/runs/{run_id}/events")
async def ws_events(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    st = run_states.get(run_id)
    if not st:
        await ws.send_text(json.dumps({"error": "unknown run_id"}))
        await ws.close()
        return

    try:
        while True:
            ev = await st.queue.get()
            await ws.send_text(json.dumps(ev, ensure_ascii=False))
    except WebSocketDisconnect:
        return


@app.get("/api/runs/{run_id}/report.md")
def get_report_md(run_id: str) -> PlainTextResponse:
    p = Path(settings.RUNS_DIR) / run_id / "report.md"
    if not p.exists():
        return PlainTextResponse("not found", status_code=404)
    return PlainTextResponse(p.read_text(encoding="utf-8"))

@app.get("/api/runs/{run_id}/graph.json")
def get_graph_json(run_id: str) -> dict[str, Any]:
    p = Path(settings.RUNS_DIR) / run_id / "graph.json"
    if not p.exists():
        return {"error": "not found"}
    return read_json(p)


@app.get("/api/runs/{run_id}/report.pdf")
def get_report_pdf(run_id: str) -> PlainTextResponse:
    p = Path(settings.RUNS_DIR) / run_id / "report.pdf"
    if not p.exists():
        return PlainTextResponse("not found", status_code=404)
    # Serving binary via PlainText isn't ideal for MVP; frontend can just link to file path in runs/ for now.
    return PlainTextResponse("pdf generated at: " + str(p))


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    base = Path(settings.RUNS_DIR)
    if not base.exists():
        return {"runs": []}
    runs = []
    for d in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        run_id = d.name
        runs.append(
            {
                "run_id": run_id,
                "has_report": (d / "report.md").exists(),
                "has_pdf": (d / "report.pdf").exists(),
                "updated_at": d.stat().st_mtime,
            }
        )
    return {"runs": runs}


@app.get("/api/runs/{run_id}/report.json")
def get_report_json(run_id: str) -> dict[str, Any]:
    p = Path(settings.RUNS_DIR) / run_id / "report.json"
    if not p.exists():
        return {"error": "not found"}
    return read_json(p)

@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str) -> dict[str, Any]:
    p = Path(settings.RUNS_DIR) / run_id / "events.jsonl"
    if not p.exists():
        return {"error": "not found"}
    lines = p.read_text(encoding="utf-8").splitlines()
    # return last N events for UI
    tail = lines[-500:]
    return {"run_id": run_id, "events": tail}


def main() -> None:
    import uvicorn

    uvicorn.run("deepresearch.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

