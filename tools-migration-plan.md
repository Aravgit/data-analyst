# Tool Registry Migration Plan (Codex-rs Style)

## Goals
- Replace inline `if/elif` tool dispatch with a registry/handler pattern similar to codex-rs.
- Centralize schemas, execution, logging, and safe serialization.
- Prevent crashes from non-serializable tool outputs (Timestamp, ndarray, DataFrame).
- Keep resource limits and truncation safeguards.

## Tasks
1) Tool interface
   - Create `backend/tools.py` with a `Tool` protocol/class: `name`, `schema` (JSON schema/dict for OpenAI tool), `run(session, args) -> dict`.
   - Standardize tool return shape: `{kind: "text"|"object"|"error", text?/json?/traceback?}`.
   - Add a `safe_serialize(result)` helper to handle Timestamp/ndarray/DataFrame/Decimal etc., fallback `default=str`, and truncate large payloads (set a uniform cap, e.g., 8 KB of text/JSON per tool output).

2) Implement tool classes
   - `PythonReplTool`: call `session.python.run(code)` (keeps resource limits via PythonSession).
   - `ListCsvFilesTool`: return registry keys.
   - `LoadCsvHandleTool`: wrap `load_csv_into_session`.
   - `UnknownTool`: structured error for missing handlers.

3) Build registry and specs
   - `TOOL_REGISTRY = {t.name: t, ...}` in `tools.py`.
   - `TOOLING_SPEC`: list of OpenAI `FunctionTool` definitions derived from each tool’s schema to pass in `responses.create`.

4) Wire agent loop
   - Import `TOOL_REGISTRY`, `TOOLING_SPEC`.
   - Replace inline dispatch with lookup; on miss, use `UnknownTool`.
   - Serialize tool outputs via `safe_serialize` before appending `function_call_output`.
   - Keep existing logging; optionally move common tool-call logging into a helper.

5) Safety practices (from codex-rs)
   - Preserve resource limits (memory/proc/time).
   - Truncate large tool outputs before storing/sending.
   - Ensure every `function_call` gets a `function_call_output`; drop/flag orphans if needed.
   - Structured errors instead of exceptions; no mid-turn compaction; retain recent tool outputs on compaction.

6) Tests
   - Unit: registry lookup, safe_serialize(Timestamp/ndarray/DataFrame), unknown tool error.
   - Integration: turn with python_repl returning Timestamp (no crash).
   - Verify token accounting unaffected.

7) Migration steps
   - Add `backend/tools.py` with classes, registry, specs.
   - Update agent to use registry/spec.
   - Remove inline `if/elif` dispatch.
   - Keep prompts unchanged.
