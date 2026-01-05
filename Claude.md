# Claude.md - CSV Data Analyst Agent

## Project Overview

This is a **CSV Data Analysis Agent** - an intelligent chatbot that allows users to upload CSV files and interact with them through natural language queries. The agent uses OpenAI's GPT-5.1 model with the Responses API and executes Python code in a sandboxed environment to perform data analysis, generate insights, create visualizations, and deliver results through a modern web interface.

### Core Capabilities
- Upload CSV files (up to 400MB) or provide server file paths
- Automatic conversion from CSV to Parquet format for efficient data processing
- Natural language queries about data with Python-backed analysis
- Real-time streaming responses via Server-Sent Events (SSE)
- Interactive data tables with pagination
- Chart generation (bar, column, stacked_column, line, scatter)
- Sandboxed Python REPL with resource limits (2GB RAM, 8 subprocesses, 15-min timeout)
- Token budget management (100k tokens) with automatic history compaction

---

## Architecture

### Tech Stack

**Backend:**
- FastAPI (Python 3.11)
- OpenAI Responses API (`gpt-5.1` model)
- pandas + numpy for data analysis
- PyArrow for Parquet operations
- cryptography for event payload encryption
- Resource-limited Python REPL (sandboxed execution)

**Frontend:**
- Next.js 15.1 (App Router)
- React 18.3.1
- TypeScript
- Tailwind CSS + shadcn/ui components
- react-markdown + remark-gfm (Markdown rendering)
- Recharts (chart visualization)

**Deployment:**
- Docker + docker-compose
- Backend: 4GB memory limit, port 8105
- Frontend: port 3001
- Persistent volume for data storage

---

## Backend Architecture

### Core Components

#### 1. main.py - FastAPI Application

**Endpoints:**
- `POST /upload` - Upload CSV file or specify server path; converts to Parquet; registers dataset in session
- `POST /chat` - Synchronous chat endpoint; returns final reply + data events
- `POST /chat/stream` - SSE streaming endpoint; emits partial tokens, tool calls, and events
- `POST /reset` - Reset session history and clear exports
- `GET /health` - Health check

**CSV to Parquet Conversion:**
- Uses PyArrow for streaming conversion (fallback to Pandas for complex formats)
- Handles encoding issues, gzip/zip files
- Stores in `/data/{session_id}/` directory
- Compression: zstd by default

#### 2. agent.py - Agent Loop Implementation

**Core Function:** `run_agent_turn()` - Executes one user turn through the agent loop

**Agent Loop Logic:**
1. Append user message to session history
2. Build model input (system prompt + normalized history)
3. Call OpenAI Responses API with tools enabled
4. Parse response output items:
   - `type: "message"` → Extract text content
   - `type: "function_call"` → Execute tool and append `function_call_output`
   - `type: "reasoning"` → Internal reasoning (not shown to user)
5. Loop until no more function calls (max 12 turns)
6. Return final reply + data events

**Token Management:**
- Tracks input/output tokens from OpenAI usage
- When total exceeds TOKEN_LIMIT (100k), compacts history
- Keeps recent messages + tool outputs + summary

**Data Events Extraction:**
- Extracts DataFrame and Chart events from tool results
- Encrypts payloads (base64 or Fernet encryption)
- Emits via SSE or returns in sync response

**Async Streaming:** `run_agent_turn_async()` - Generator for SSE streaming
- Yields events: `status`, `partial`, `tool_call`, `tool_result`, `token_usage`, `reply`, `error`, `data_frame`, `chart`

#### 3. session.py - Session & REPL Management

**SessionState:**
- `session_id`: UUID for session tracking
- `messages`: Conversation history (dicts for Responses API)
- `python`: PythonSession instance (persistent REPL)
- `total_tokens`: Cumulative token count
- `csv_registry`: Maps logical dataset names to parquet paths + metadata
- `summary`: Compacted history summary

**PythonSession:**
- Persistent Python namespace (globals dict) with pandas/numpy preloaded
- `run(code)` - Executes code with resource limits; returns structured result
- **Resource Limits:**
  - Memory: 2GB (via RLIMIT_AS)
  - Subprocesses: 8 max (via RLIMIT_NPROC)
  - Timeout: 15 minutes (via signal.SIGALRM)
- **Result Serialization:**
  - DataFrames → `{kind: "dataframe", head, columns, row_count, dtypes}`
  - Series/dict/list → Converted to DataFrame
  - Scalars → Wrapped in 1-row DataFrame
  - Objects → JSON serialization with fallback
- **Memory Management:**
  - Tracks DataFrames in namespace
  - Prunes old DataFrames when total exceeds 500MB
  - Keeps most recent + protected names

#### 4. tools.py - Tool Registry

**Tool Architecture:**
- `Tool` dataclass: `name`, `schema` (OpenAI function schema), `run(session, args)`
- `TOOL_REGISTRY`: Dict mapping tool names to Tool instances
- `TOOLING_SPEC`: List of OpenAI function schemas for API calls

**Available Tools:**
1. **python_repl** - Execute Python code in persistent REPL
2. **list_datasets** - List registered dataset names
3. **get_dataset_info** - Get schema info (columns, dtypes, row_count) without loading
4. **load_dataset_full** - Load entire Parquet dataset into DataFrame
5. **load_dataset_sample** - Load first N rows for quick inspection
6. **load_dataset_columns** - Load specific columns (all rows)
7. **find_value_columns** - Scan dataset to find which columns contain a value
8. **send_data_to_ui_as_df** - Emit DataFrame to UI as data event
9. **send_chart_to_ui** - Emit chart spec for visualization

**Parquet Reading Strategy:**
- Uses PyArrow for streaming reads (memory_map=False for container stability)
- Thread limits (1 thread) to avoid pthread failures in constrained containers
- Fallback batch reading for resource-constrained environments

#### 5. data_events.py - Data Delivery

**make_data_frame_event:**
- Extracts DataFrame metadata (columns, dtypes, row_count)
- Limits rows to FINAL_DF_ROW_LIMIT (2000) and cells to FINAL_DF_CELL_LIMIT (200k)
- Encrypts payload (base64 or Fernet)
- Returns `data_frame` events

**make_chart_event:**
- Validates chart spec (type, x_field, series)
- Supported types: bar, column, stacked_column, line, scatter
- Limits: 200 rows, 10 series max
- Validates numeric y_field for cartesian charts
- Returns `chart` event or rejection reason

---

## Frontend Architecture

### Key Components

#### 1. app/page.tsx - Landing/Upload Page
- Upload CSV file or specify server path
- Calls `/upload` endpoint
- Redirects to `/chat?session={id}&csv={name}` after upload

#### 2. app/chat/ChatClient.tsx - Main Chat Interface

**Layout:**
- **Left Rail:** Data upload controls, current file info, tips
- **Center Panel:** 3 tabs (Summary, Data, Charts) + message input
- **Right Rail:** Activity log (tool calls, tokens, status)

**State Management:**
- `messages`: Chat history with role/content
- `dataFrames`: Array of DataFrameEvent (most recent first)
- `charts`: Array of ChartPayload (most recent first)
- `activity`: Tool call logs + status events
- `tokens`: input/output/total token counts
- `activeTab`: "summary" | "data" | "charts"

**SSE Event Handling:**
- Connects to `/chat/stream` endpoint
- Parses SSE events: `partial`, `reply`, `status`, `tool_call`, `tool_result`, `token_usage`, `data_frame`, `chart`, `chart_rejected`, `error`
- **data_frame events:** Decrypts payload → Updates dataFrames state → Shows in Data tab
- **chart events:** Validates spec → Updates charts state → Shows in Charts tab
- **partial events:** Streams text updates to assistant message
- **reply events:** Finalizes assistant message

#### 3. components/ui/paginated-table.tsx - Data Table
- Displays DataFrame with pagination (10 rows/page)
- Shows: logical_name, columns, row_count, dtypes
- Responsive table with horizontal scroll

#### 4. components/ui/chart-card.tsx - Chart Visualization
- Uses Recharts components (LineChart, BarChart, ScatterChart)
- Supports: bar, column, stacked_column, line, scatter
- Custom formatters for X-axis (date detection)
- Color palette for series
- Labels, legend, tooltip

---

## Integration Flow

### Complete User Interaction Flow

1. **Upload:**
   - User uploads CSV → Frontend POST /upload
   - Backend converts CSV → Parquet, registers in session
   - Returns session_id + csv_name
   - Frontend redirects to /chat with session params

2. **Chat Turn:**
   - User sends message → Frontend POST /chat/stream
   - Backend agent loop:
     - Appends user message to history
     - Calls OpenAI Responses API with system prompt + tools
     - Model returns function_calls (e.g., `list_datasets`, `load_dataset_full`, `python_repl`)
     - Backend executes tools → Returns structured results
     - Model processes results → Calls more tools or returns final message
   - Backend emits SSE events: `tool_call`, `tool_result`, `data_frame`, `chart`, `reply`

3. **Data Rendering:**
   - `data_frame` event → Frontend decrypts → Stores in dataFrames state
   - User switches to "Data" tab → PaginatedTable renders DataFrame
   - `chart` event → Frontend validates + decrypts → Stores in charts state
   - User switches to "Charts" tab → ChartCard renders chart with Recharts

4. **Activity Tracking:**
   - All tool calls, token usage, status updates logged in right rail
   - Shows: tool name, result kind, token counts

---

## Key Design Patterns

### 1. Parquet-First Architecture
- All CSV uploads immediately converted to Parquet
- Tools read Parquet (not CSV) for performance
- Streaming reads to handle large files

### 2. OpenAI Responses API Integration
- Uses `responses.create()` with function calling
- Strict schemas (`strict: true`) require all properties in `required` array
- Parallel tool calls supported (default behavior)
- Reasoning blocks ignored in history

### 3. Event-Driven Data Delivery
- Data/charts delivered via encrypted events (separate from text replies)
- Frontend decrypts on-the-fly
- Enables immediate table/chart rendering before summary completes

### 4. Resource Safety
- REPL sandboxing with memory/process/time limits
- Thread reduction to 1 (ARROW_NUM_THREADS=1, OMP_NUM_THREADS=1)
- DataFrame pruning to prevent memory bloat
- Token budget enforcement with auto-compaction

### 5. Streaming UX
- SSE for real-time updates
- Partial text streaming (when enabled)
- Activity rail shows tool execution progress
- Tab-based organization (Summary/Data/Charts)

---

## System Prompt & Agent Behavior

### System Prompt Strategy

The system prompt instructs the model to:
- Use tools for all data access (never guess or hallucinate data values)
- Prefer full-data analysis over sampling when feasible
- Deliver tables via `send_data_to_ui_as_df` events (not inline markdown)
- Create charts only when they add clarity (optional)
- Ask for clarification when requirements are ambiguous
- Stop calling tools once the answer is ready
- Return concise, insight-focused summaries

### Tool Calling Best Practices

**Strict Schemas:**
- OpenAI Responses API with `strict: true` requires all properties listed in `required`
- Even optional fields must be in `required` array with default values
- Example: chart tool has `x_label`, `y_label`, `title`, `note` with `default: ""`

**Parallel Tool Calls:**
- Responses API may return multiple `function_call` items in one turn
- Backend handles them independently (same turn)
- Collects all calls → Executes each → Returns all outputs
- If one fails, returns error for that call_id; others proceed

---

## Environment & Configuration

### Environment Variables

Create `.env` file in repo root:

```bash
OPENAI_API_KEY=sk-...

# Optional: Enable streaming (default: 0)
ENABLE_RESPONSES_STREAMING=0

# Optional: Data storage path (default: ./data)
DATA_ROOT=/data

# Optional: Enable event encryption (default: 0)
DATA_EVENT_ENCRYPT=0

# Optional: Encryption key for events
DATA_EVENT_KEY=your-secret-key
```

### Docker Setup

**docker-compose.yml:**
- **Backend Service:** Python 3.11, FastAPI, port 8105, 4GB memory limit, volume for /data
- **Frontend Service:** Node 20, Next.js, port 3001, depends on backend
- **Network:** Services communicate on compose network (backend:8105)

**Commands:**
```bash
# Build and start
docker compose build --no-cache
docker compose up -d

# View logs
docker compose logs -f backend

# Stop
docker compose down
```

**Endpoints:**
- Backend: http://localhost:8105
- Frontend: http://localhost:3001

---

## Development

### Local Development (Hot Reload)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8105
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Will run on port 3001
# Set NEXT_PUBLIC_API_BASE=http://localhost:8105 in .env.local
```

### Testing

**Backend:**
```bash
# Syntax check
python -m py_compile backend/agent.py backend/main.py backend/session.py

# Run tests (if available)
cd backend
pytest
```

**Frontend:**
```bash
cd frontend
npm run build
npm run lint
```

---

## Key File Locations

### Backend
- `backend/main.py` - FastAPI app and endpoints
- `backend/agent.py` - Agent loop and Responses API integration
- `backend/session.py` - Session management and Python REPL
- `backend/tools.py` - Tool registry and definitions
- `backend/data_events.py` - Data event generation and encryption
- `backend/requirements.txt` - Python dependencies

### Frontend
- `frontend/app/page.tsx` - Landing/upload page
- `frontend/app/chat/ChatClient.tsx` - Main chat interface
- `frontend/components/ui/paginated-table.tsx` - Data table component
- `frontend/components/ui/chart-card.tsx` - Chart visualization component
- `frontend/lib/decrypt.ts` - Payload decryption utilities
- `frontend/package.json` - Node dependencies

### Configuration
- `docker-compose.yml` - Docker orchestration
- `.env` - Environment variables (not in git)
- `.gitignore` - Git ignore patterns

### Documentation
- `README.md` - Quick start guide
- `AGENTS.md` - Agent behavior and API guide
- `Claude.md` - This file (comprehensive reference)
- `plans/` - Architectural plans and design documents

---

## Current Implementation Status

### Completed Features ✅
- Full CSV upload and Parquet conversion
- OpenAI Responses API integration with function calling
- 9 working tools (python_repl, dataset operations, UI emission)
- Sandboxed Python REPL with resource limits
- Token budget management with history compaction
- SSE streaming for real-time updates
- Data table rendering with pagination
- Chart generation and visualization (5 types)
- Event-driven data delivery with encryption
- Activity logging and token tracking
- Memory management (DataFrame pruning)

### Planned Improvements ⏳

**From plans/overhaul-plan.md:**
- Event/Op dispatcher with proper async queuing
- Full streaming implementation (currently `USE_STREAMING=0`)
- Telemetry hooks and structured logging
- Metrics backend (Prometheus-style)
- Session persistence (Redis)

**From plans/data-delivery-plan.md:**
- Export for large datasets (planned, not yet implemented)
- Download buttons for oversized results

---

## Security & Safety

### Sandboxing
- Python REPL runs with resource limits (RLIMIT_AS, RLIMIT_NPROC, SIGALRM)
- 2GB RAM limit per session
- 8 subprocess max
- 15-minute execution timeout
- No network access in production (firewall/container isolation)

### Data Privacy
- Event payloads encrypted with Fernet or base64
- No sensitive data logged
- Session-isolated storage
- 400MB upload limit to prevent DoS

### Best Practices
- Validate all tool arguments
- Sanitize outputs before returning to model
- Truncate large payloads (2000 rows, 200k cells)
- Clear error messages without exposing internals
- Audit logging for all tool executions

---

## Troubleshooting

### Common Issues

**CSV Upload Fails:**
- Check file size (max 400MB)
- Verify encoding (try UTF-8, latin1)
- Check for gzip/zip format (auto-detected)
- Look for backend logs: `docker compose logs backend`

**Python REPL Timeout:**
- Increase timeout in `session.py` (default 15 min)
- Check memory usage (may hit 2GB limit)
- Verify DataFrame sizes aren't excessive

**Charts Not Rendering:**
- Check browser console for decryption errors
- Verify chart spec has required fields
- Ensure data has ≤200 rows, ≤10 series
- Check y_field is numeric for bar/line/scatter

**Token Limit Exceeded:**
- History auto-compacts at 100k tokens
- Increase TOKEN_LIMIT if needed
- Check conversation length
- Use `/reset` endpoint to clear history

**Container Memory Issues:**
- Backend limited to 4GB in docker-compose
- Increase memory_limit if needed
- Check DataFrame pruning is working
- Monitor with `docker stats`

---

## API Reference

### REST Endpoints

#### POST /upload
Upload a CSV file or register a server path.

**Request:**
```typescript
// Multipart form data
{
  file?: File,           // CSV file
  path?: string,         // Server path (alternative to file)
  session_id?: string    // Optional session ID (creates new if omitted)
}
```

**Response:**
```typescript
{
  session_id: string,
  csv_name: string,
  message: string
}
```

#### POST /chat
Synchronous chat endpoint.

**Request:**
```typescript
{
  session_id: string,
  message: string
}
```

**Response:**
```typescript
{
  reply: string,
  tokens: {
    input: number,
    output: number,
    total: number
  },
  data_frames: EncryptedPayload[],
  charts: EncryptedPayload[]
}
```

#### POST /chat/stream
SSE streaming chat endpoint.

**Request:**
```typescript
{
  session_id: string,
  message: string
}
```

**Response:** SSE stream with events:
- `status` - Agent status updates
- `partial` - Partial text tokens
- `tool_call` - Tool invocation
- `tool_result` - Tool execution result
- `token_usage` - Token counts
- `data_frame` - DataFrame event
- `chart` - Chart event
- `reply` - Final assistant message
- `error` - Error details

#### POST /reset
Reset session history and clear data.

**Request:**
```typescript
{
  session_id: string
}
```

**Response:**
```typescript
{
  status: "ok",
  message: string
}
```

---

## Contributing

### Code Style
- Backend: Follow PEP 8, use type hints
- Frontend: TypeScript strict mode, ESLint rules
- Use descriptive variable names
- Add docstrings for functions
- Comment complex logic

### Git Workflow
1. Create feature branch from `main`
2. Make changes with clear commit messages
3. Test locally with Docker
4. Submit PR with description
5. Address review feedback

### Adding New Tools
1. Define schema in `backend/tools.py`
2. Implement `run(session, args)` method
3. Register in `TOOL_REGISTRY`
4. Update system prompt if needed
5. Add tests and documentation

---

## Resources

### Documentation
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js 15 Documentation](https://nextjs.org/docs)
- [Recharts Documentation](https://recharts.org/)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

### Related Files
- See `plans/` folder for detailed architectural plans
- See `AGENTS.md` for agent behavior guide
- See `README.md` for quick start

---

## License

[Add your license information here]

## Contact

[Add contact information here]

---

*Last Updated: 2026-01-05*
*Version: 1.0.0*
