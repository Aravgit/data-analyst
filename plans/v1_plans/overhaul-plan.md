# Agent Overhaul Plan (Python, aligned with codex-rs behaviors)

Goal: Rebuild the backend agent to match codex-rs’ architecture and the OpenAI Responses API best practices: correct function/tool calling, parallel calls, streaming/SSE, telemetry, robust session lifecycle, and safer execution.

## 1) Architecture & Process Model
- Introduce an internal dispatcher with two queues:
  - Submission Queue (SQ): receives Ops (e.g., UserMessage, UploadRegistered, Reset, ConfigUpdate).
  - Event Queue (EQ): emits Events (e.g., ModelStarted, ToolCalled, ToolResult, ModelReply, Error, TokenUsage).
- Run a long-lived async “agent loop” task that:
  1) Dequeues Ops, updates session state.
  2) Builds model requests (prompt + tools + state).
  3) Handles streaming responses and tool calls.
  4) Pushes Events to EQ.
- HTTP endpoints become thin shims that enqueue Ops and optionally stream Events (Server-Sent Events / websockets).

## 2) Model Integration (Responses API)
- Use `responses.create` with:
  - `input`: system prompt + conversation history + last tool outputs.
  - `tools`: function schemas (flattened).
  - `parallel_tool_calls`: true (default).
  - `tool_choice`: auto unless overridden.
  - `max_output_tokens`, temperature, top_p from config.
- Streaming:
  - Enable streaming to surface partial outputs and tool call headers to clients via SSE.
  - Accumulate chunks to reconstruct `response.output`.
- Function calling loop:
  1) Call model → get `response.output` list.
  2) Collect all items with `type == "function_call"` (each has name, arguments, call_id).
  3) If none: finalize assistant message, emit ModelReply event, end turn.
  4) If present: execute tools (parallel where safe), produce `function_call_output` items (one per call_id).
  5) Append those outputs to the conversation and call `responses.create` again (same turn), repeating until no more calls.
- Robust parsing: always `model_dump()` Pydantic items; tolerate empty/None content; JSON-parse arguments with good errors.

## 3) Tools & Execution
- Tool schemas:
  - python_repl(code: string)
  - list_datasets()
  - load_dataset_full(name: string, df_name: string)
  - (future) shell, apply_patch, plan if approved.
- Execution policy:
  - Python REPL is stateful; serialize REPL calls (no parallel within a session).
  - Stateless tools may run in parallel (use asyncio.gather).
  - Resource limits remain: 1 GB mem, max 5 subprocesses, 15-min wall timeout.
  - Clear, concise descriptions in schemas; optional examples in system prompt.
- Error handling:
  - Wrap each tool call; return structured error text in `function_call_output` (do not crash loop).
  - Log traceback server-side; sanitize outputs.

## 4) Session & State
- SessionState holds: messages, tool outputs, REPL globals, token totals, config knobs.
- SessionStore: thread-safe (async lock); support eviction or TTL later.
- Token accounting: use `usage` attributes; accumulate; enforce TOKEN_LIMIT with early and post-checks.
- Reset: clear messages, REPL, dataset registry (parquet), token count.

## 5) Telemetry & Logging
- Structured logs with session_id, turn, attempt, tool_name, call_id, tokens.
- Emit telemetry events (EQ) for: turn_start, model_call_start/stop, tool_call_start/stop/result, token_usage, errors.
- Optional metrics sink (Prometheus-style counters/latencies) behind a feature flag.

## 6) Streaming to Clients (SSE/WebSocket)
- Provide `/chat/stream` endpoint:
  - Accepts a message, enqueues Op, streams Events from EQ as SSE.
  - Event types: status, partial tokens, tool_call_started, tool_call_output, final_reply, error.
- Keep `/chat` synchronous endpoint by consuming the same Events internally and returning final reply.

## 7) Prompts & Guidance
- System prompt updates:
  - Explain when to use tools, how to format tool calls, and when to stop.
  - Ask model to minimize tokens, batch tool calls when possible, and avoid unnecessary calls.
  - Include failure-handling instruction: on tool error, summarize and continue or ask for clarification.
- Developer notes (hidden) describing available tools and constraints.

## 8) Configuration
- Config precedence: env vars → config file → request overrides.
- Expose: model, temperature/top_p, max_output_tokens, TOKEN_LIMIT, parallel_tool_calls toggle, streaming toggle, log level.

## 9) HTTP API Adjustments
- `/upload`: convert CSV to parquet; emit Event when dataset registered.
- `/chat`: enqueue Op, block until final Event; return reply, tokens, status.
- `/chat/stream`: SSE; forward Events.
- `/health`: keep simple OK.

## 10) Testing Plan
- Unit: tool dispatcher, argument parsing, content extraction, empty output handling, token accounting.
- Integration: mock `responses.create` to emit (a) message only, (b) single function_call, (c) multiple parallel function_call, (d) malformed args, (e) empty output.
- Concurrency: ensure REPL calls are serialized; parallel stateless tools tested with asyncio.gather.
- End-to-end: upload + chat happy path; token-limit path; tool error path.

## 11) Migration Steps
1) Introduce Event/Op types and async dispatcher; keep old sync path temporarily.
2) Implement new model-call loop with full function_call/function_call_output handling.
3) Add streaming endpoint; adapt sync `/chat` to consume same events.  
   - Status: placeholder SSE endpoint `/chat/stream` added (emits start, reply, status, done). Full event queue wiring still pending.
4) Refine system prompt and tool schemas.
5) Add telemetry hooks and structured logging.
6) Expand tests; remove old legacy parsing.

## 12) Open Questions
- Do we need shell/apply_patch tools now or defer?
- Persistence of sessions across process restarts (Redis?) or keep in-memory?
- Approval/permission model for dangerous tools?
- Metrics backend choice (Prometheus vs. logs only).
