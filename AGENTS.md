# Agent Behavior Guide

This document explains how the CSV Data Analyst Agent works, including its core behaviors, tool usage patterns, and API integration with OpenAI's Responses API.

## Overview

The CSV analysis agent combines a FastAPI backend with a Next.js frontend to provide natural language data analysis. It uses OpenAI's Responses API (GPT-5.1), tool calling, and a sandboxed Python REPL to analyze uploaded CSVs. All uploads are converted to Parquet immediately for efficient processing.

## Core Agent Behavior

### Agent Loop Execution

Each user turn runs through an agent loop:
1. **Receive Message** - User sends natural language query
2. **Build Input** - Combine system prompt + conversation history + user message
3. **Call Model** - Send to OpenAI Responses API with tools enabled
4. **Parse Response** - Extract messages, function_calls, reasoning
5. **Execute Tools** - Run requested functions locally
6. **Return Results** - Append tool outputs and loop until complete
7. **Emit Events** - Send data/chart events to frontend

**Key Rules:**
- Tools must be executed safely with structured error handling
- Tool outputs are serialized into `function_call_output` entries
- Stop calling tools once a final answer is ready
- Track token usage; compact history when budget exceeded (100k tokens)
- Final responses are concise and insight-focused

### Responses API Integration

The agent uses OpenAI's Responses API with these patterns:

**Request Structure:**
```python
response = client.responses.create(
    model="gpt-5.1",
    input=[
        {"type": "system_message", "content": SYSTEM_PROMPT},
        *normalized_history,
        {"type": "user_message", "content": user_query}
    ],
    tools=TOOLING_SPEC,
    parallel_tool_calls=True,  # Default
    tool_choice="auto"
)
```

**Response Output Items:**
- `type: "message"` - Assistant text content
- `type: "function_call"` - Tool invocation (name, arguments, call_id)
- `type: "reasoning"` - Internal reasoning (not stored in history)

**Function Call Handling:**
```python
# Parse function calls
for item in response.output:
    if item.type == "function_call":
        result = execute_tool(item.name, item.arguments)
        append_function_call_output(item.call_id, result)

# Loop until no more calls
if has_function_calls:
    continue_agent_loop()
else:
    return_final_reply()
```

### Parallel Tool Calling

The Responses API supports parallel tool execution:
- Multiple `function_call` items may be returned in one response
- Backend handles them independently (same turn)
- Collects all calls → Executes each → Returns all outputs
- If one fails, return error for that call_id; others proceed
- REPL calls are serialized (stateful); other tools may run in parallel

### Token Management

**Budgeting:**
- Track input + output tokens from OpenAI usage
- Cumulative total stored in session state
- TOKEN_LIMIT = 100,000 tokens

**Compaction Strategy:**
When limit exceeded:
1. Generate summary of conversation history
2. Keep recent messages (last 5-10 turns)
3. Preserve important tool outputs
4. Replace older messages with summary
5. Continue with reduced context

**Implementation:** `backend/agent.py` - `compact_history()`

## Available Tools (9)

### Data Access Tools

**1. python_repl**
```python
{
  "name": "python_repl",
  "description": "Execute Python code in persistent REPL",
  "parameters": {
    "code": str  # Python code to execute
  }
}
```
- Stateful execution (variables persist)
- Returns structured JSON: `{kind, text/head/columns/row_count/dtypes}`
- Resource limited: 2GB RAM, 8 subprocesses, 15-min timeout

**2. list_datasets**
```python
{
  "name": "list_datasets",
  "description": "List registered dataset names",
  "parameters": {}
}
```
- Returns list of available CSV/Parquet names
- Shows logical names registered in session

**3. get_dataset_info**
```python
{
  "name": "get_dataset_info",
  "description": "Get schema without loading data",
  "parameters": {
    "name": str  # Dataset logical name
  }
}
```
- Fast schema inspection
- Returns: columns, dtypes, row_count
- No data loaded into memory

**4. load_dataset_full**
```python
{
  "name": "load_dataset_full",
  "description": "Load entire Parquet file",
  "parameters": {
    "name": str,      # Dataset logical name
    "df_name": str    # Variable name in REPL
  }
}
```
- Loads full dataset into REPL as DataFrame
- Use only when full-table load required
- Prefer sampling or column selection for large files

**5. load_dataset_sample**
```python
{
  "name": "load_dataset_sample",
  "description": "Load first N rows for inspection",
  "parameters": {
    "name": str,      # Dataset logical name
    "df_name": str,   # Variable name in REPL
    "nrows": int      # Number of rows to load
  }
}
```
- Fast preview of data
- Recommended first step for large datasets

**6. load_dataset_columns**
```python
{
  "name": "load_dataset_columns",
  "description": "Load specific columns (all rows)",
  "parameters": {
    "name": str,         # Dataset logical name
    "columns": [str],    # List of column names
    "df_name": str       # Variable name in REPL
  }
}
```
- Memory-efficient for wide datasets
- Use when only specific columns needed

**7. find_value_columns**
```python
{
  "name": "find_value_columns",
  "description": "Search for value across columns",
  "parameters": {
    "name": str,     # Dataset logical name
    "value": str     # Value to search for
  }
}
```
- Scans all columns for matching value
- Returns list of columns containing the value

### UI Emission Tools

**8. send_data_to_ui_as_df**
```python
{
  "name": "send_data_to_ui_as_df",
  "description": "Emit DataFrame to UI as table",
  "parameters": {
    "df_name": str,           # DataFrame variable in REPL
    "logical_name": str       # Display name for UI
  }
}
```
- Emits encrypted `data_frame` event
- Limits: 2000 rows, 200k cells
- Frontend renders as paginated table

**9. send_chart_to_ui**
```python
{
  "name": "send_chart_to_ui",
  "description": "Emit chart spec for visualization",
  "parameters": {
    "df_name": str,                    # DataFrame in REPL
    "logical_name": str,               # Display name
    "chart_type": str,                 # bar|column|stacked_column|line|scatter
    "x_field": str,                    # X-axis column
    "x_label": str,                    # X-axis label (optional)
    "y_label": str,                    # Y-axis label (optional)
    "series": [                        # Data series (1-10)
      {
        "name": str,                   # Series name
        "y_field": str                 # Y-axis column
      }
    ],
    "title": str,                      # Chart title (optional)
    "note": str                        # Additional note (optional)
  }
}
```
- Validates chart spec before emission
- Limits: 200 rows, 10 series max
- Y-fields must be numeric for cartesian charts
- Returns `chart` event or `chart_rejected` with reason

## Data Event Flow

### Event Types

**data_frame Event:**
```json
{
  "type": "data_frame",
  "logical_name": "sales_data",
  "columns": ["date", "product", "revenue"],
  "dtypes": {"date": "object", "product": "object", "revenue": "float64"},
  "row_count": 1500,
  "rows": [/* encrypted payload */]
}
```

**chart Event:**
```json
{
  "type": "chart",
  "chart_type": "bar",
  "title": "Revenue by Product",
  "x_field": "product",
  "series": [{"name": "Revenue", "y_field": "revenue"}],
  "data": [/* encrypted payload */]
}
```

**chart_rejected Event:**
```json
{
  "type": "chart_rejected",
  "reason": "Too many rows (500 > 200 limit)"
}
```

### Encryption

**Backend (`backend/data_events.py`):**
- Uses Fernet symmetric encryption or base64 encoding
- Controlled by `DATA_EVENT_ENCRYPT` env variable
- Key from `DATA_EVENT_KEY` env variable

**Frontend (`frontend/lib/decrypt.ts`):**
- Tries JSON parse → base64url decode → plain string
- Handles both encrypted and unencrypted payloads
- Errors logged to console

## Output Rules for the Agent

### Text Responses

**DO:**
- Provide concise summaries and insights
- Explain key findings in plain language
- Suggest next steps or aggregations
- Use markdown formatting for readability

**DON'T:**
- Inline large tables (>20 rows) in text
- Duplicate data already sent via events
- Use non-ASCII unless specifically required
- Over-explain obvious results

### Data Delivery

**Tables:**
- Always use `send_data_to_ui_as_df` for tabular results
- Text reply should reference the table, not repeat it
- Example: "I've sent a table with 1,500 rows showing..."

**Charts:**
- Use only when visualization adds clarity
- Optional for most queries
- Keep chart simple (≤10 series)
- Example: "I've created a bar chart showing revenue by product."

**Large Results:**
- For >2000 rows, send sample + summary
- Suggest aggregations or filters
- Use `load_dataset_columns` to reduce width

## Session & Storage

### SessionState Structure
```python
@dataclass
class SessionState:
    session_id: str                    # UUID
    messages: List[Dict]               # Conversation history
    python: PythonSession              # Persistent REPL
    total_tokens: int                  # Cumulative count
    csv_registry: Dict[str, CsvInfo]   # name -> path + metadata
    summary: str                       # Compacted history summary
```

### PythonSession (REPL)

**Initialization:**
- Pre-loaded: `pandas`, `numpy`
- Persistent globals dict
- Session-isolated namespace

**Execution:**
```python
result = session.python.run(code)
# Returns: {kind, text|head|columns|row_count|dtypes}
```

**Resource Limits:**
- Memory: 2GB (RLIMIT_AS)
- Subprocesses: 8 max (RLIMIT_NPROC)
- Timeout: 15 minutes (SIGALRM)

**Memory Management:**
- Tracks DataFrame sizes
- Auto-prunes when total > 500MB
- Keeps most recent + protected names

### Dataset Registry

**Structure:**
```python
csv_registry = {
    "sales_2024": {
        "path": "/data/abc123/sales_2024.parquet",
        "columns": ["date", "product", "revenue"],
        "row_count": 10000,
        "dtypes": {...}
    }
}
```

**Parquet Storage:**
- Location: `DATA_ROOT/{session_id}/`
- Compression: zstd
- Format: PyArrow-compatible Parquet

## System Architecture

### Backend Endpoints

**POST /upload**
- Accepts CSV file or server path
- Converts to Parquet
- Registers in session
- Returns: `{session_id, csv_name, message}`

**POST /chat**
- Synchronous endpoint
- Returns final reply + data events
- Response: `{reply, tokens, data_frames, charts}`

**POST /chat/stream**
- SSE streaming endpoint
- Emits: `status`, `partial`, `tool_call`, `tool_result`, `data_frame`, `chart`, `reply`
- Real-time updates

**POST /reset**
- Clears session history
- Deletes exports
- Keeps dataset registry

**GET /health**
- Simple health check
- Returns: `{"status": "ok"}`

### Frontend Structure

**Layout (3-panel):**
- **Left Rail:** Upload controls, file info, tips
- **Center Panel:** 3 tabs (Summary, Data, Charts) + input
- **Right Rail:** Activity log (tool calls, tokens)

**State Management:**
- `messages` - Chat history
- `dataFrames` - Decrypted table data (newest first)
- `charts` - Chart specs (newest first)
- `activity` - Tool execution logs
- `tokens` - Usage tracking

**SSE Event Handling:**
- `partial` → Stream text updates
- `tool_call` → Log tool usage
- `data_frame` → Decrypt & render table
- `chart` → Render visualization
- `reply` → Finalize message
- `error` → Show error message

## Environment Configuration

### Required Variables
```bash
OPENAI_API_KEY=sk-...  # OpenAI API key
```

### Optional Variables
```bash
# Streaming (0 or 1, default: 0)
ENABLE_RESPONSES_STREAMING=0

# Data storage path (default: /data)
DATA_ROOT=/data

# Event encryption (0 or 1, default: 0)
DATA_EVENT_ENCRYPT=0

# Encryption key (required if DATA_EVENT_ENCRYPT=1)
DATA_EVENT_KEY=your-secret-key
```

### Docker Configuration
- Backend: 4GB memory limit, port 8105
- Frontend: port 3001
- Shared volume: `data_root` for `/data`
- Environment: `.env` mounted to backend

## Resource Limits

**REPL Execution:**
- Memory: 2GB per session
- Subprocesses: 8 max
- Timeout: 15 minutes
- Thread limits: 1 (PyArrow, OpenMP)

**Data Events:**
- Table rows: 2000 max
- Table cells: 200k max
- Chart rows: 200 max
- Chart series: 10 max

**Uploads:**
- File size: 400MB max
- Session exports: 400MB total
- Encoding: UTF-8, latin1, auto-detect

## Testing

### Backend Tests
```bash
# Syntax check
python -m py_compile backend/agent.py backend/main.py backend/session.py

# Unit tests
cd backend
pytest

# Coverage
pytest --cov=. --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm run build    # Build check
npm run lint     # Linter
npm run test     # Unit tests (if configured)
```

### Integration Testing
1. Upload CSV via `/upload`
2. Send query via `/chat` or `/chat/stream`
3. Verify tool calls in logs
4. Check data events in frontend
5. Validate token tracking
6. Test compaction at 100k tokens

## Best Practices

### For Model Prompting
- Request specific analyses, not vague questions
- Specify output format (table, chart, summary)
- Ask for clarification when ambiguous
- Reference column names explicitly

### For Tool Usage
- Start with `list_datasets` if unsure
- Use `get_dataset_info` before loading
- Prefer sampling over full loads
- Use `load_dataset_columns` for wide data
- Emit tables/charts before final summary

### For Development
- Log all tool calls with session_id
- Sanitize tool outputs before returning
- Handle tool errors gracefully
- Test with large datasets (>100k rows)
- Monitor memory usage in Docker

## Documentation Links

- **Claude.md** - Comprehensive project reference
- **README.md** - Quick start guide
- **plans/** - Architectural plans and design docs
  - `agent-notes.md` - Technical notes on agent loop
  - `overhaul-plan.md` - Architecture refactoring
  - `tools-migration-plan.md` - Tool registry migration
  - `data-delivery-plan.md` - Event encryption strategy
  - `chart-plan.md` - Chart feature design

## Support

For issues or questions:
1. Check logs: `docker compose logs -f backend`
2. Review **Claude.md** for detailed reference
3. Check plans/ folder for design decisions
4. Test with small CSV first
5. Verify OpenAI API key and quota

---

*Last Updated: 2026-01-05*
