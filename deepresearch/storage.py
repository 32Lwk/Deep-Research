from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    events_jsonl: Path
    messages_jsonl: Path
    evidence_json: Path
    report_md: Path
    report_json: Path
    graph_json: Path


def init_run_dir(runs_dir: str, run_id: str) -> RunPaths:
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        events_jsonl=run_dir / "events.jsonl",
        messages_jsonl=run_dir / "messages.jsonl",
        evidence_json=run_dir / "evidence.json",
        report_md=run_dir / "report.md",
        report_json=run_dir / "report.json",
        graph_json=run_dir / "graph.json",
    )


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = orjson.dumps(obj).decode("utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event(run_id: str, type_: str, **fields: Any) -> dict[str, Any]:
    return {"ts_ms": _now_ms(), "run_id": run_id, "type": type_, **fields}

