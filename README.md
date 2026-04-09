# DeepResearch MAS (Multi-Agent System)

複数LLM（OpenAI / Claude / Gemini / PLaMo(OpenAI互換)）とWeb検索（SerpAPI）を組み合わせ、10体程度の役割エージェントが対話しながら調査し、引用付きレポート（Markdown/PDF）と機械可読JSONを生成するためのMVPです。

## セットアップ

1) 依存関係

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2) `.env`

このリポジトリは `.env` を読み込みます。最低限、以下のいずれかを設定してください（既存のキー名も互換対応します）。

- OpenAI: `OPENAI_API_KEY`（互換: `OpenAI_API_KEY`）
- Claude: `ANTHROPIC_API_KEY`（互換: `Claude_API_KEY`）
- Gemini: `GEMINI_API_KEY`（互換: `Gemini_API_KEY`）
- PLaMo(OpenAI互換): `PLAMO_BASE_URL` / `PLAMO_API_KEY` / `PLAMO_MODEL`（互換: `PLaMo_API_KEY`）
- SerpAPI: `SERPAPI_API_KEY`（未設定の場合、検索はスキップされます）

## 実行

### CLI（最短）

```bash
python -m deepresearch.cli run "調査したいテーマ"
```

成果物は `./runs/<run_id>/` に保存されます（`report.md`, `report.json`, `graph.jsonl` など）。

### Web（FastAPI）

```bash
python -m deepresearch.web
```

- `GET /` 簡易UI（旧）
- `POST /api/runs` run作成
- `POST /api/runs/{run_id}/files` ファイルアップロード（PDF/MD）
- `POST /api/runs/{run_id}/start` run開始
- `POST /api/runs/{run_id}/stop` 停止（ソフトキャンセル）
- `WS /api/runs/{run_id}/events` イベント購読（ストリーミング）

### Frontend（React）

別ターミナルで起動します（Viteのproxyで `/api` をFastAPIへ転送）。

```bash
cd frontend
npm install
npm run dev
```

ブラウザで `http://127.0.0.1:5173/`

## PDF出力について

PDFは Pandoc が利用可能な場合のみ自動生成します（`pandoc` がPATHにあること）。生成物は `./runs/<run_id>/report.pdf` です。

