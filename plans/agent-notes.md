# Agent & Tool-Calling Notes

## Core agent loop
- Build a single agent first: model + clear system prompt + small, well-documented tools.
- Keep message history; prepend the system prompt every turn.
- Steps per user turn:  
  1) Append user message.  
  2) Call `responses.create` with `input=[system, history]`, `tools`, `tool_choice` (usually `auto`).  
  3) Inspect `response.output` items:  
     - `type == "message"`: contains assistant text blocks.  
     - `type == "function_call"`: contains `name`, `arguments`, `call_id`.  
  4) If any function calls: execute locally; send another `responses.create` with prior input plus `{"type":"function_call_output","call_id":..., "output": <string or json>}`. Repeat until no more calls.  
  5) When no calls remain, return the assistant text.
- Enforce turn/time/token limits to prevent runaway loops.

## Tool definitions
- Each tool entry (Responses API):  
  ```json
  {
    "type": "function",
    "name": "python_repl",
    "description": "Execute Python code in a persistent REPL.",
    "parameters": { "type": "object", "properties": { "code": {"type":"string"} }, "required":["code"] }
  }
  ```
- Provide concise, action-oriented descriptions; include constraints/expected outputs.
- Add examples in the system prompt if the tool is complex.
- All uploaded CSVs are converted to parquet; tools read parquet only.

### Strict tool schemas (OpenAI Responses)
- When `strict: true`, OpenAI requires the `required` array to list **every** property that appears in `properties`. If you omit any property from `required`, the schema is rejected with `"required is required to be supplied and to be an array including every key in properties"`.
- To keep optional fields, give them safe defaults and still include them in `required` (the model must send them; defaults are only for validation).
- Example (chart tool):
  - properties: `df_name, logical_name, chart_type, x_field, x_label, y_label, series, title, note`
  - required: all of the above, even if you treat some as optional in logic.
  - defaults: `x_label`, `y_label`, `title`, `note` set to `""` so the model can send empty strings.
  - Schema snippet:
    ```json
    {
      "type": "object",
      "properties": {
        "df_name": {"type":"string"},
        "logical_name": {"type":"string","default":""},
        "chart_type": {"type":"string","enum":["bar","column","stacked_column","line","scatter"]},
        "x_field": {"type":"string"},
        "x_label": {"type":"string","default":""},
        "y_label": {"type":"string","default":""},
        "series": {"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"y_field":{"type":"string"}},"required":["name","y_field"],"additionalProperties":false},"minItems":1,"maxItems":10},
        "title": {"type":"string","default":""},
        "note": {"type":"string","default":""}
      },
      "required": ["df_name","logical_name","chart_type","x_field","x_label","y_label","series","title","note"],
      "additionalProperties": false
    }
    ```
- Update the system prompt/instructions so the model knows to always supply those fields (empty string is fine for optional ones).

## Handling function calls (Responses API)
- Detect calls: iterate all `response.output` items; collect any with `type == "function_call"`.
- Execute functions safely; validate/parse `arguments` JSON; on parse error, return a friendly error string to the model.
- Return outputs as `function_call_output` items referencing the original `call_id`.
- Keep `tool_choice: "auto"` by default; set `"none"` to force no tools or `{"type":"function","function":{...}}` to force one tool.

### Parallel tool calling
- Responses API may return **multiple** `function_call` items in one `response.output` (the flag is `parallel_tool_calls: true`).
- Handle them independently in the same turn:
  1) Collect all calls (`call_id`, `name`, `arguments`).
  2) Execute each tool; gather results in any order.
  3) Send a follow-up `responses.create` with *all* corresponding `function_call_output` items (one per `call_id`) appended to the input.
- Do not serialize/queue them unless the tools conflict; default is to run them concurrently/asynchronously where safe.
- If one tool fails, return an error string for that `call_id` so the model can recover; still send outputs for the others.

## System prompt guidance
- Spell out: role, when to use tools, when to stop, format of final answer, and safety/constraints.
- Include completion criteria (e.g., “If answer is sufficient, reply concisely without further tool calls.”).

## Guardrails & limits
- Resource limits: memory, subprocess count, wall-clock timeout for tool code.
- File/IO limits: cap upload size; restrict file paths if exposing server paths.
- Token budget: track input+output; bail out with a clear message when exceeded.
- Logging: log model attempts, tool calls, errors, token usage; redact secrets.

## Testing & debugging tips
- Log the raw `response.output` shape when integrating a new model/version.
- Add unit tests that mock `responses.create` outputs (message only, function_call, empty output).
- Fuzz-test argument parsing for tools to avoid crashes on malformed JSON.
