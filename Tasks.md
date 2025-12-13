Tasks to build the CSV-analysis agent (Python)
-----------------------------------------------

1) Requirements & design
   - Confirm user-facing flow: accept either an uploaded CSV file or filesystem path; respond with natural-language summary plus any tables/plots as needed.
   - Pick core model: `gpt-5.1` with `reasoning.effort="medium"`; prefer streaming responses.
   - Define stopping condition: final LLM turn returns text with no tool calls (decision call to exit loop).

2) Tooling contracts
   - Define `python_repl` tool (function calling) that executes code in a persistent per-session namespace; return tagged JSON (`kind: text|dataframe|object|error` with repr/head/columns).
   - Add helper tools:
       * `list_csv_files` to enumerate available datasets.
       * `load_csv_handle` to bind a logical CSV name to a DataFrame variable inside the REPL (safer than arbitrary paths).
       * Optional `web_search` tool if you want the agent to fetch context (use OpenAI web-search capability).
   - Decide tool exposure rules: use `allowed_tools` to restrict active tools per turn; require preamble/plan before tool calls.

3) Session memory
   - Implement per-user `session_id` that maps to both chat history window and a persistent Python session (globals).
   - Choose storage: in-memory for dev; SQLite or Redis for persistence (Agents SDK offers `SQLiteSession`).
   - Add TTL/size limits and a reset endpoint/command.

4) Agent loop
   - Compose system/developer prompts instructing the model to:
       * prefer `python_repl` for calculations,
       * keep outputs concise,
       * explain results in plain language,
       * exit when analysis is complete.
   - Implement loop: send messages + tools → if tool_calls → dispatch → append tool results → re-call model until final text (no tool_calls) or max_turns.
   - Add token accounting using OpenAI usage from each turn; enforce hard cap of 20,000 tokens combined input+output per conversation and terminate gracefully with a summary when reached.
   - Set reasoning effort to medium; temperature low (e.g., 0.1–0.3).

5) CSV handling
   - For uploads: save to a sandboxed `/data/{session_id}/` path; validate size and schema quickly (`pd.read_csv(..., nrows=50)` sampling).
   - For file paths: enforce allowlist/root directory to avoid arbitrary access.
   - Normalize encodings, handle gzip/zip, and report validation errors clearly.

6) Serialization of results
   - For DataFrames: return repr of head, columns list, row_count, dtypes; cap rows returned.
   - For plots: generate PNGs via matplotlib/plotly, save, and return file handles/URLs the frontend can render.
   - For text/objects: include both repr and JSON-friendly payload when possible.

7) Safety & sandboxing
   - Run `python_repl` in an isolated worker (container/vm); limit CPU/memory/time; block network.
   - Log executed code and tool calls for auditing; redact secrets in logs.

8) Testing & QA
   - Unit test tool serializers and session store.
   - Golden-path integration test: upload CSV → basic stats → filtered calc → final answer.
   - Regression test for decision-to-exit behavior (model stops when no further tools needed).

9) Deployment & ops
   - Package as FastAPI service with endpoints: `/analyze` (conversation turns), `/upload`, `/health`.
   - Add observability: request logs, latency, tool-call counts, error traces.
   - Document environment variables (OPENAI_API_KEY, sandbox paths, limits) and provide docker-compose for local run.

10) Frontend (Next.js + shadcn + Tailwind + Vite)
    - Build a simple two-step flow:
        * Home page: upload CSV or provide path, then start a session.
        * Chat page: user sends queries; show messages in chat layout.
    - Display agent status: show “thinking…”/current task while tool calls run; replace with final formatted answer when the loop returns text.
    - Render DataFrame heads as tables and plots/images when provided; keep transcript and allow reset/clear.
