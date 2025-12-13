# Data Analyst Agent

CSV chatbot that pairs a FastAPI backend (Python) with a Next.js 15 frontend. It uses OpenAI’s Responses API, tool-calling, and a sandboxed Python REPL to analyze uploaded CSVs without sending the raw file to the model.

## Tech stack
- **Backend:** FastAPI, Python 3.11, OpenAI Responses API, sandboxed Python REPL (resource limits: 1 GB RAM, ≤5 subprocesses, 15‑min exec timeout).
- **Frontend:** Next.js 15.1, React, shadcn/ui, `react-markdown` + `remark-gfm` for Markdown chat rendering.
- **Containerization:** Docker & docker-compose.

## Prerequisites
- Docker and docker-compose installed.
- An OpenAI API key with access to the Responses API.

## Environment
Create a `.env` file in the repo root:
```
OPENAI_API_KEY=sk-...
# Optional: ENABLE_RESPONSES_STREAMING=0|1 (default 0 – final-only replies)
# Optional: DATA_ROOT=/data  (where uploads are stored inside the backend container)
```
The compose file mounts `.env` into the backend.

## Run with Docker
```bash
# from repo root
docker compose build --no-cache
docker compose up -d

# stop
docker compose down
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

Uploads are kept in the `data_root` named volume (inside container at `/data`).

## Development tips
- To see agent logs: `docker compose logs -f backend`
- To rebuild after code changes: `docker compose build backend frontend && docker compose up -d`

## How it works (high level)
1) Upload CSV (`/upload`): registers the file path in the session and seeds a hint message; resets history if a new file replaces the old one.
2) Chat (`/chat` or `/chat/stream`): backend runs the agent loop
   - Sends history + system prompt to OpenAI Responses
   - Executes tool calls locally (`python_repl`, `list_csv_files`, `load_csv_handle`)
   - Keeps only role/content messages and tool call/output records in history; “reasoning” blocks are ignored for history
   - Auto-compacts history when token budget (~100k) is hit (internal, not shown to user)
3) Frontend renders Markdown chat bubbles and an activity rail (tool calls, status, tokens).

## Hot reload (optional local dev)
For quickest iteration on the frontend you can run Next.js locally (requires Node 20+):
```bash
cd frontend
npm install
npm run dev
```
Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` in a `.env.local` under `frontend/` if you run backend via Docker.

## Testing
Basic syntax check:
```bash
python -m py_compile backend/agent.py backend/main.py backend/session.py
```
Add your own data-centric tests under `backend/tests/` as needed.
