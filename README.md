# Data Analyst Agent

**AI-powered CSV analysis chatbot**. Upload your data, ask questions, get insights with tables and charts.

![Data Analyst Agent UI](images/readme_homepage_image.png)

## Features

- Natural language queries on CSV data
- 400MB uploads with automatic Parquet conversion
- Interactive tables and chart generation
- Streaming progress via SSE
- Sandboxed Python REPL for analysis

## Quick Start

```bash
git clone <repository-url>
cd data-analyst
git config core.hooksPath .githooks

# Create .env
OPENAI_API_KEY=sk-...

# Build + run
docker compose build --no-cache
docker compose up -d
```

Access:
- Frontend: http://localhost:3001
- Backend: http://localhost:8105

## Tech Stack

- Backend: FastAPI + Python 3.11, OpenAI Responses API
- Frontend: Next.js + React, Tailwind, Recharts
- Runtime: Docker + docker-compose

## API

- `POST /upload`
- `POST /chat`
- `POST /chat/stream`
- `POST /reset`
- `GET /health`

## Configuration

- `OPENAI_API_KEY` (required)
- `ENABLE_RESPONSES_STREAMING` (0/1)
- `DATA_ROOT` (default: `/data`)
- `DATA_EVENT_ENCRYPT` (0/1)
- `DATA_EVENT_KEY`

Docs: see `Claude.md`

## License

Apache License 2.0. See `LICENSE`.
