You can build this with **one core tool** (`python_repl`) plus optional helpers. I’ll sketch:

* The **tool schema** (JSON / OpenAPI-ish).
* A **minimal Python backend** that:

  * Maintains a **persistent REPL per session**.
  * Executes arbitrary Python.
  * Returns a **tagged JSON result** (`kind: dataframe | text | object | error`).
* How many tools + what “memory” you actually need.
* How this becomes a full agent loop.

If you want, I can later turn this into a FastAPI microservice or an “agent SDK”-style package.

---

## 1. Tool / function schema for `python_repl`

### a) OpenAI tools schema (for Chat/Responses API)

Based on the current tools/function-calling docs, tools are defined with JSON Schema under a `tools` array.

Here’s a schema for your REPL:

```jsonc
[
  {
    "type": "function",
    "function": {
      "name": "python_repl",
      "description": "Execute Python code in a persistent, sandboxed REPL attached to this conversation. Use this for all CSV/data analysis.",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Python code to execute. Assume a persistent session, you can reuse variables defined in earlier calls."
          }
        },
        "required": ["code"],
        "additionalProperties": false
      }
    }
  }
]
```

### b) Expected output shape (the “multiple outputs” trick)

The protocol only allows **one return object**, so we encode multiple shapes via a `kind` field:

```ts
type PythonResult =
  | { kind: "text"; text: string }
  | {
      kind: "dataframe";
      repr: string;
      head: any[][];
      columns: string[];
      row_count: number;
      meta?: any;
    }
  | { kind: "object"; repr: string; json?: any }
  | { kind: "error"; traceback: string };
```

Your backend will **actually return** JSON like:

```json
{
  "kind": "dataframe",
  "repr": "   col1  col2\n0    1     2\n1    3     4",
  "head": [[1, 2], [3, 4]],
  "columns": ["col1", "col2"],
  "row_count": 1000
}
```

or

```json
{
  "kind": "error",
  "traceback": "Traceback (most recent call last):\n ..."
}
```

The model learns how to branch on `kind`.

If you want to be strict, you can also supply a **Structured Output schema** via the newer APIs, but we can layer that later.

---

## 2. Minimal Python backend for the REPL

I’ll give you a **single-file, minimal but complete** backend that:

* Uses `openai` Python SDK (modern `OpenAI` client).
* Keeps a **PythonSession** object per `session_id`.
* Executes code, inspects the **last expression**, and builds the `PythonResult` JSON.
* Orchestrates a simple **tool-calling loop**.

You can paste this into `agent.py` and run it as a script (no web framework yet).

> If you’d rather not see full code like this going forward, tell me and I’ll just show the key functions next time.

```python
import ast
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
from openai import OpenAI

# ---------- OpenAI client ----------

client = OpenAI()  # expects OPENAI_API_KEY in env

MODEL = "gpt-5.1-thinking"  # adjust to what you actually have


# ---------- Python REPL session ----------

@dataclass
class PythonSession:
    """
    Very simple, not-secure REPL environment.
    You MUST isolate this (e.g., container) in production.
    """
    globals: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Preload some safe-ish builtins and libraries
        base_globals = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,  # tighten this in prod
        }
        # Import common DS stack
        base_globals["pd"] = pd
        try:
            import numpy as np
            base_globals["np"] = np
        except ImportError:
            pass

        self.globals.update(base_globals)

    def run(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code and return a PythonResult-like dict:
        { kind: "dataframe" | "text" | "object" | "error", ... }
        """
        try:
            # Parse so we can treat last expression specially
            parsed = ast.parse(code, mode="exec")

            # If last statement is an Expr, we want its value
            last_value_node = None
            body = parsed.body
            if body and isinstance(body[-1], ast.Expr):
                last_value_node = body[-1].value
                body = body[:-1]  # rest of the code

            # Exec everything except last expression
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]),
                             filename="<repl>",
                             mode="exec"),
                     self.globals,
                     self.globals)

            value = None
            if last_value_node is not None:
                # Eval the last expression to capture the result
                value = eval(compile(ast.Expression(last_value_node),
                                     filename="<repl>",
                                     mode="eval"),
                             self.globals,
                             self.globals)

            # Decide how to serialize the result
            return self._serialize_value(value)

        except Exception:
            tb = traceback.format_exc()
            return {
                "kind": "error",
                "traceback": tb,
            }

    def _serialize_value(self, value: Any) -> Dict[str, Any]:
        """
        Turn Python object into JSON-compatible dict that the model can reason over.
        """
        # Nothing returned or last line was a statement
        if value is None:
            return {
                "kind": "text",
                "text": "Code executed. No return value."
            }

        # DataFrame case
        if isinstance(value, pd.DataFrame):
            head_df = value.head(10)
            head_records = head_df.to_dict(orient="records")
            return {
                "kind": "dataframe",
                "repr": repr(head_df),
                "head": head_records,
                "columns": list(value.columns),
                "row_count": int(len(value)),
                "meta": {
                    "dtypes": {c: str(t) for c, t in value.dtypes.items()}
                }
            }

        # Generic Python objects: try to JSON-ify, else just repr
        try:
            import json
            json_value = json.loads(json.dumps(value, default=str))
        except Exception:
            json_value = None

        return {
            "kind": "object",
            "repr": repr(value),
            "json": json_value,
        }


# ---------- Session store (super minimal) ----------

_sessions: Dict[str, PythonSession] = {}


def get_session(session_id: str) -> PythonSession:
    if session_id not in _sessions:
        _sessions[session_id] = PythonSession()
    return _sessions[session_id]


# ---------- Tool schema in Python ----------

PYTHON_REPL_TOOL = {
    "type": "function",
    "function": {
        "name": "python_repl",
        "description": (
            "Execute Python code in a persistent REPL for CSV/data analysis. "
            "You can reuse variables from previous calls in this conversation. "
            "Prefer concise, idiomatic pandas and numpy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "The Python code to run. "
                        "Assume any CSVs are accessible in the /data directory. "
                        "Return the final result expression instead of printing when possible."
                    )
                }
            },
            "required": ["code"],
            "additionalProperties": False
        }
    }
}


# ---------- Agent loop (single turn helper) ----------

def agent_step(user_message: str,
               session_id: Optional[str] = None,
               messages: Optional[list] = None) -> str:
    """
    One step of the agent:
      - Send messages + tools to OpenAI
      - If it asks to call python_repl, run it
      - Feed tool result back
      - Return final model reply text
    """

    if session_id is None:
        session_id = str(uuid.uuid4())

    if messages is None:
        messages = []

    # System prompt: tell the model HOW to use the tool
    system_message = {
        "role": "system",
        "content": (
            "You are a data analysis assistant. "
            "You are allowed to execute Python code via the python_repl tool. "
            "Use pandas for CSV analysis, avoid loading entire huge files into memory, "
            "and always explain your results in plain language after running code."
        )
    }

    # Append existing convo + new user message
    convo = [system_message] + messages + [
        {"role": "user", "content": user_message}
    ]

    # 1) Call the model with tools enabled
    response = client.chat.completions.create(
        model=MODEL,
        messages=convo,
        tools=[PYTHON_REPL_TOOL],
        tool_choice="auto",
        temperature=0.1,
    )

    msg = response.choices[0].message

    # 2) If no tool calls, just return its text
    if not msg.tool_calls:
        return msg.content or ""

    # 3) Handle tool call(s) – we only expect python_repl here
    tool_messages = []
    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments  # JSON string; SDK may already parse this in newer versions
        import json
        args = json.loads(tool_args)

        if tool_name == "python_repl":
            session = get_session(session_id)
            result = session.run(args["code"])

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result)
            })

    # 4) Call the model again with tool outputs
    convo_with_tools = convo + [msg] + tool_messages

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=convo_with_tools,
        temperature=0.1,
    )

    final_msg = final_response.choices[0].message
    return final_msg.content or ""


if __name__ == "__main__":
    # Tiny manual REPL to test the agent
    sess_id = str(uuid.uuid4())
    history = []
    print("CSV Agent (type 'exit' to quit)")
    while True:
        user_in = input("You: ")
        if user_in.strip().lower() == "exit":
            break
        reply = agent_step(user_in, session_id=sess_id, messages=history)
        history.append({"role": "user", "content": user_in})
        history.append({"role": "assistant", "content": reply})
        print("Agent:", reply)
```

This is intentionally barebones so you can adapt it.

---

## 3. What’s *required* vs *optional* (tools, memory, etc.)

### Tools: how many do you actually need?

**Strict minimum:**

1. `python_repl` (above)

   * Can do *all* CSV work: loading files, analysis, plotting, etc., if you give it access to the filesystem (or an abstraction).

**Very useful optional tools:**

2. `list_csv_files`

   * Lets the model discover what’s available instead of guessing filenames.
   * Example schema:

     ```json
     {
       "type": "function",
       "function": {
         "name": "list_csv_files",
         "description": "List available CSV files for this user/session.",
         "parameters": {
           "type": "object",
           "properties": {},
           "additionalProperties": false
         }
       }
     }
     ```

3. `load_csv_handle`

   * Instead of reading arbitrary paths, you give the model short IDs.

     ```json
     {
       "type": "function",
       "function": {
         "name": "load_csv_handle",
         "description": "Load a CSV by logical name and inject it into the REPL as a pandas DataFrame.",
         "parameters": {
           "type": "object",
           "properties": {
             "name": {
               "type": "string",
               "description": "Logical name of the CSV file (e.g. 'sales_2024')."
             },
             "df_name": {
               "type": "string",
               "description": "Name of the DataFrame variable in the REPL, e.g. 'df_sales'."
             }
           },
           "required": ["name", "df_name"],
           "additionalProperties": false
         }
       }
     }
     ```

   Internally, this tool just does `session.globals[df_name] = pd.read_csv(path)` and returns a small confirmation JSON.

You *can* get away with **only `python_repl`** if you’re okay exposing filesystem paths; otherwise I’d add at least `list_csv_files` to make the UX nicer.

---

### Memory: what you actually need

There are **two layers of memory**:

1. **Model-side memory** = chat history

   * You already have this: just keep passing previous messages (or a windowed subset).

2. **Tool-side memory** = REPL state

   * This is crucial for your use case.
   * Persist **one `PythonSession` per logical “agent session”**:

     * Key: `session_id` from your app / user.
     * Value: `PythonSession(globals=...)`.
   * In production, this should live in:

     * A long-lived worker process, or
     * A small in-memory store keyed by session id, or
     * An external container / microservice you talk to over RPC.

If you later want **long-term memory** (e.g., vector DB of past analyses), that’s a separate tool (`search_notes`, `save_note`).

---

## 4. How to build the agent end-to-end

Rough implementation plan:

1. **Define tools**

   * Start with `python_repl`.
   * Optionally add `list_csv_files` and `load_csv_handle`.

2. **Implement REPL backend** (above):

   * Ensure it can:

     * Import pandas.
     * See a `/data` directory or mapping of logical CSV names → paths.
     * Return structured JSON based on the result type.

3. **Orchestrate tool calling**:

   * Wrap the OpenAI API call logic into a function like `agent_step`.
   * Loop:

     1. Send `messages` + `tools`.
     2. If `tool_calls` present: dispatch to Python backend.
     3. Append tool outputs as `tool` role messages.
     4. Call model again to get final text.

4. **Attach your CSVs**:

   * Decide how you pass “location of my CSV files”:

     * Option A: mount a directory and tell the model: “All CSVs are in `/data/`.”
     * Option B: maintain a registry `{logical_name -> path}` and expose it via `list_csv_files` / `load_csv_handle`.

5. **Harden it**:

   * Security: run REPL in **a real sandbox** (Docker, firejail, gVisor, etc.). Don’t run arbitrary `exec` on your app host.
   * Resource limits: kill sessions after N minutes, limit memory, etc.
   * Logging: log executed code and results for debugging, but watch for sensitive data.

---

## 5. Quick recap

* You **don’t** need a special protocol for “multiple outputs” – the tool just returns **one JSON object** with a `kind` field and whatever extra keys you want.
* You can start with **one tool** (`python_repl`) and a **persistent REPL per session**.
* The minimal backend above is enough to:

  * Take a user question.
  * Let GPT-5 decide when to run Python.
  * Execute Python.
  * Return structured results (df/text/object/error).
  * Let the model explain results back to the user.
