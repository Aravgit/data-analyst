import json
import logging
from typing import Dict, Any, Optional, List, Iterable
import os
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import Response

from session import SessionState
from pydantic import BaseModel
from tools import TOOL_REGISTRY, TOOLING_SPEC, safe_tool_output
from data_events import make_data_frame_event, encrypt_payload

MODEL = "gpt-5.1"
REASONING_EFFORT = "medium"
TOKEN_LIMIT = 100000  # combined input + output (budget before auto-compaction)
MAX_API_RETRIES = 3
# Streaming of partial chunks is temporarily disabled; we emit only final replies.
USE_STREAMING = bool(int(os.getenv("ENABLE_RESPONSES_STREAMING", "0")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data")).resolve()

logger = logging.getLogger("csv_agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_history(session: SessionState) -> str:
    """
    Lightweight compaction: capture the last few user/assistant turns into a short text summary.
    This avoids sending full history when token budget is hit.
    Output is Markdown-friendly.
    """
    tail = [m for m in session.messages if m.get("role") in ("user", "assistant")]
    tail = tail[-6:]
    parts = []
    for m in tail:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        parts.append(f"- **{role}**: {content}")
    summary = "## Conversation Summary\n" + "\n".join(parts)
    return summary[:1200]




def build_model_input(session: SessionState) -> List[Dict[str, Any]]:
    """
    Normalize history for the Responses API:
    - allow role/content messages
    - allow function_call and function_call_output pairs
    - drop any telemetry/status artifacts
    """
    normalized: List[Dict[str, Any]] = []
    for item in session.messages:
        if item.get("type") == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output", "")
            if call_id:
                normalized.append({"type": "function_call_output", "call_id": call_id, "output": output})
        elif item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            arguments = item.get("arguments", "{}")
            if call_id and name:
                normalized.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
        else:
            role = item.get("role")
            if role in ("user", "assistant", "system"):
                normalized.append({"role": role, "content": item.get("content", "")})
    return normalized


def compact_session(
    session: SessionState, pending_user: Optional[Dict[str, Any]] = None, keep_tail: int = 2
) -> str:
    """
    Summarize recent conversation and reset token/state.
    - Keeps a short summary.
    - Retains the most recent user/assistant exchanges (default last 2 role messages) so continuity remains.
    - Retains the most recent tool call/output pairs to preserve computed evidence.
    - Optionally appends the pending user message (for pre-turn compaction).
    """
    summary = summarize_history(session)
    # Collect the last few user/assistant role messages
    tail_msgs: List[Dict[str, Any]] = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in session.messages
        if m.get("role") in ("user", "assistant")
    ][-keep_tail:]

    # Collect the last tool call/output pair (if any) to keep recent computation context
    tool_tail: List[Dict[str, Any]] = []
    for m in reversed(session.messages):
        if m.get("type") in ("function_call_output", "function_call"):
            tool_tail.append(m)
            if len(tool_tail) >= 4:  # capture up to two call/output pairs
                break
    tool_tail = list(reversed(tool_tail))

    new_messages = [{"role": "assistant", "content": summary}]
    new_messages.extend(tool_tail)
    new_messages.extend(tail_msgs)
    if pending_user:
        new_messages.append(pending_user)

    session.messages = new_messages
    session.total_tokens = 0
    session.summary = summary
    return summary


def _extract_data_event(result: Any, session: SessionState) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    if result.get("kind") != "dataframe":
        return None

    df_name = result.get("df_name", "df")
    logical = result.get("logical_name") or df_name
    df = session.python.globals.get(df_name)
    if df is not None:
        try:
            evts, _ = make_data_frame_event(df, df_name, session.session_id, logical, DATA_ROOT)
            return evts[0] if evts else None
        except Exception:
            return None

    head = result.get("head") or []
    payload = {
        "df_name": df_name,
        "logical_name": logical,
        "columns": result.get("columns") or [],
        "rows": head,
        "row_count": result.get("row_count"),
        "dtypes": result.get("dtypes") or {},
    }
    return {"type": "data_frame", "payload": encrypt_payload(payload)}


def run_agent_turn(
    session: SessionState, user_message: str, max_turns: int = 12
) -> Dict[str, Any]:
    """
    Execute the agent loop for one user message, respecting the token budget.
    Returns: {"reply": str, "total_tokens": int, "status": "ok"|"token_limit"}
    """
    if session.total_tokens >= TOKEN_LIMIT:
        compact_session(session, pending_user={"role": "user", "content": user_message})
        logger.info("token budget reached, pre-turn compact", extra={"session_id": session.session_id})
    else:
        session.messages.append({"role": "user", "content": user_message})
    logger.info(
        "user message received",
        extra={"session_id": session.session_id, "turn": len(session.messages), "content_preview": user_message[:200]},
    )
    turns = 0
    reply_text = ""
    reply_sent = False

    over_limit_after_turn = False
    reply_tokens = 0
    turn_input_tokens = 0
    turn_output_tokens = 0
    last_data_event: Optional[Dict[str, Any]] = None

    while turns < max_turns:
        turns += 1
        response = None
        last_error: Optional[Exception] = None
        filtered_inputs = build_model_input(session)

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                logger.info(
                    "calling model",
                    extra={
                        "session_id": session.session_id,
                        "attempt": attempt,
                        "turn": turns,
                        "input_items": len(filtered_inputs),
                        "msg_count": len(session.messages),
                        "streaming": False,
                    },
                )
                response = client.responses.create(
                    model=MODEL,
                    reasoning={"effort": REASONING_EFFORT},
                    input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                    tools=TOOLING_SPEC,
                    max_output_tokens=2000,
                )
                # print(response)
                # print(response)
                break
            except Exception as exc:  # network/transient failures
                last_error = exc
                logger.exception(
                    "model call failed",
                    extra={"session_id": session.session_id, "attempt": attempt, "turn": turns},
                )
                if attempt == MAX_API_RETRIES:
                    reply_text = f"Upstream model call failed after {MAX_API_RETRIES} attempts: {exc}"  # surfaced to user for visibility
                    session.messages.append({"role": "assistant", "content": reply_text})
                    return {
                        "reply": reply_text,
                        "total_tokens": session.total_tokens,
                        "status": "error",
                    }
                continue

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        turn_input_tokens += input_tokens
        turn_output_tokens += output_tokens
        session.total_tokens += input_tokens + output_tokens
        logger.info(
            "model response received",
            extra={
                "session_id": session.session_id,
                "turn": turns,
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": session.total_tokens,
            },
        )

        if session.total_tokens >= TOKEN_LIMIT:
            over_limit_after_turn = True
        reply_tokens = session.total_tokens

        outputs = response.output or []

        # log raw model output for debugging (truncated)
        try:
            raw_out = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
            logger.info(
                "model output",
                extra={
                    "session_id": session.session_id,
                    "turn": turns,
                    "raw_output": raw_out[:2000],
                },
            )
        except Exception:
            pass

        text_parts: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        reasoning_only = False

        for item in outputs:
            if hasattr(item, "model_dump"):
                itm = item.model_dump()
            elif isinstance(item, dict):
                itm = item
            else:
                try:
                    itm = json.loads(str(item))
                except Exception:
                    itm = {}

            item_type = itm.get("type")
            if item_type == "message":
                content_blocks = itm.get("content") or []
                if not isinstance(content_blocks, list):
                    content_blocks = [content_blocks]
                for block in content_blocks:
                    if block.get("type") in ("text", "output_text"):
                        text_parts.append(block.get("text", ""))
            elif item_type == "function_call":
                function_calls.append(itm)
            elif item_type == "reasoning":
                reasoning_only = True

        if function_calls:
            # Keep the function calls in history so the next model call has call ids to match outputs
            for fc in function_calls:
                logger.info(
                    "function_call received",
                    extra={
                        "session_id": session.session_id,
                        "turn": turns,
                        "call_id": fc.get("call_id") or fc.get("id"),
                        "name": fc.get("name"),
                    },
                )
                session.messages.append(
                    {
                        "type": "function_call",
                        "call_id": fc.get("call_id") or fc.get("id"),
                        "name": fc.get("name"),
                        "arguments": fc.get("arguments", "{}"),
                    }
                )
            function_results: List[Dict[str, Any]] = []
            for fc in function_calls:
                call_id = fc.get("call_id") or fc.get("id")
                name = fc.get("name")
                args_raw = fc.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception as exc:
                    args = {}
                    logger.exception("argument parse failed", extra={"session_id": session.session_id, "call_id": call_id})
                    result = {"kind": "error", "traceback": f"Bad arguments: {exc}"}
                else:
                    logger.info(
                        "executing tool",
                        extra={
                            "session_id": session.session_id,
                            "turn": turns,
                            "tool": name,
                            "call_id": call_id,
                            "args_preview": str(args)[:200],
                        },
                    )
                    tool = TOOL_REGISTRY.get(name)
                    if tool is None:
                        result = {"kind": "error", "traceback": f"Unknown tool {name}"}
                        logger.error(
                            "unknown tool requested",
                            extra={"session_id": session.session_id, "turn": turns, "tool": name},
                        )
                    else:
                        try:
                            result = tool.run(session, args)
                        except Exception as exc:
                            logger.exception(
                                "tool execution failed",
                                extra={"session_id": session.session_id, "turn": turns, "tool": name},
                            )
                            result = {"kind": "error", "traceback": repr(exc)}

                serialized = safe_tool_output(result)

                function_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": serialized,
                    }
                )

                try:
                    evt = _extract_data_event(result, session)
                    if evt:
                        last_data_event = evt
                except Exception:
                    logger.exception("data event extraction failed", extra={"session_id": session.session_id})

                logger.info(
                    "tool result",
                    extra={
                        "session_id": session.session_id,
                        "turn": turns,
                        "tool": name,
                        "call_id": call_id,
                        "result_kind": result.get("kind") if isinstance(result, dict) else None,
                    },
                )

            # Add tool outputs to history and continue loop for another model call
            session.messages.extend(function_results)
            continue

        # If we only got reasoning and no message or tool calls, ask the model again.
        if reasoning_only and not text_parts:
            logger.info(
                "reasoning-only output, retrying for final message",
                extra={"session_id": session.session_id, "turn": turns},
            )
            continue

        # Final response (no function calls)
        message_text = "\n".join([p for p in text_parts if p])
        reply_text = message_text if message_text.strip() else reply_text
        if not reply_text.strip():
            reply_text = "No content returned."
        session.messages.append({"role": "assistant", "content": reply_text})
        logger.info(
            "assistant reply ready",
            extra={"session_id": session.session_id, "turn": turns, "reply_preview": reply_text[:200]},
        )
        break

    # If we exhausted the turn loop without a reply, run one last model call disallowing tools.
    if not reply_text.strip():
        try:
            filtered_inputs = build_model_input(session)
            force_response = client.responses.create(
                model=MODEL,
                reasoning={"effort": REASONING_EFFORT},
                input=[{"role": "system", "content": system_prompt() + "\nProvide the final answer now based on prior tool results. Do not call tools."}, *filtered_inputs],
                tools=[],  # disable tool calls for the final attempt
                max_output_tokens=800,
            )
            parts = []
            for out in force_response.output or []:
                itm = out.model_dump() if hasattr(out, "model_dump") else (out if isinstance(out, dict) else {})
                if itm.get("type") == "message":
                    for blk in itm.get("content") or []:
                        if blk.get("type") in ("text", "output_text"):
                            parts.append(blk.get("text", ""))
            merged = "\n".join([p for p in parts if p]).strip()
            if merged:
                reply_text = merged
                session.messages.append({"role": "assistant", "content": reply_text})
        except Exception:
            pass

    if not reply_text.strip():
        reply_text = "I wasn't able to produce a final answer after tool calls. Please ask again or try a simpler request."

    # Post-turn compaction so the answered reply is preserved; applies only after the turn completes.
    # We keep the UI clean by not surfacing token-limit status; just log it.
    if over_limit_after_turn:
        logger.warning("token budget hit; compacting post-turn", extra={"session_id": session.session_id})
        compact_session(session)
        # after compaction, surface input/output as 0 but keep total as the pre-compaction tally
        turn_input_tokens = 0
        turn_output_tokens = 0
    status = "ok"

    return {
        "reply": reply_text,
        "total_tokens": reply_tokens,
        "input_tokens": turn_input_tokens,
        "output_tokens": turn_output_tokens,
        "status": status,
        "data_events": [last_data_event] if last_data_event else [],
    }


async def run_agent_turn_async(
    session: SessionState, user_message: str, max_turns: int = 12
) -> Iterable[Dict[str, Any]]:
    """
    Async generator version that yields events for streaming consumers.
    Implements a loop over Responses API with function_call/function_call_output handling.
    Event types:
      status: {stage: start|model_call|tool_call|done}
      partial: {text: "..."}  (currently from final message; TODO: wire real streaming chunks)
      tool_call: {call_id, name, arguments}
      tool_result: {call_id, result_kind}
      token_usage: {input_tokens, output_tokens, total_tokens}
      reply: {text, status, total_tokens}
      error: {message}
    """
    if session.total_tokens >= TOKEN_LIMIT:
        summary = compact_session(session, pending_user={"role": "user", "content": user_message})
        logger.warning("token budget hit pre-turn; compacted", extra={"session_id": session.session_id})
    else:
        session.messages.append({"role": "user", "content": user_message})
    yield {"type": "status", "stage": "start"}

    turns = 0
    reply_text = ""
    reply_sent = False
    over_limit_after_turn = False
    reply_tokens = 0
    last_data_event: Optional[Dict[str, Any]] = None

    while turns < max_turns:
        turns += 1
        yield {"type": "status", "stage": "model_call"}
        try:
            filtered_inputs = build_model_input(session)
            if USE_STREAMING:
                stream = await async_client.responses.create(
                    model=MODEL,
                    reasoning={"effort": REASONING_EFFORT},
                    input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                    tools=TOOLING_SPEC,
                    max_output_tokens=2000,
                    stream=True,
                )
                # print(stream)
                logger.info(
                    "calling model (stream)",
                    extra={
                        "session_id": session.session_id,
                        "turn": turns,
                        "input_items": len(filtered_inputs),
                        "msg_count": len(session.messages),
                        "streaming": True,
                    },
                )
                async for chunk in stream:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else {}
                    outputs = chunk_dict.get("output", []) or []
                    for out in outputs:
                        if out.get("type") == "message":
                            for blk in out.get("content") or []:
                                if blk.get("type") in ("text", "output_text") and blk.get("text"):
                                    yield {"type": "partial", "text": blk["text"]}
                # After streaming partials, fetch a full response (with usage) non-streaming
                response: Response = await async_client.responses.create(
                    model=MODEL,
                    reasoning={"effort": REASONING_EFFORT},
                    input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                    tools=TOOLING_SPEC,
                    max_output_tokens=2000,
                )
                # print(response)
            else:
                response: Response = await async_client.responses.create(
                    model=MODEL,
                    reasoning={"effort": REASONING_EFFORT},
                    input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                    tools=TOOLING_SPEC,
                    max_output_tokens=2000,
                )
                # print(response)
        except Exception as exc:
            yield {"type": "error", "message": f"Model call failed: {exc}"}
            break

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        session.total_tokens += input_tokens + output_tokens
        yield {
            "type": "status",
            "stage": "model_call_complete",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": session.total_tokens,
        }
        yield {
            "type": "token_usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": session.total_tokens,
        }

        if session.total_tokens >= TOKEN_LIMIT:
            over_limit_after_turn = True
        reply_tokens = session.total_tokens

        outputs: List[Any] = response.output or []
        # log raw model output for debugging (truncated)
        try:
            raw_out = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
            logger.info(
                "model output",
                extra={"session_id": session.session_id, "turn": turns, "raw_output": raw_out[:2000]},
            )
        except Exception:
            pass

        normalized_outputs: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        reasoning_only = False

        for item in outputs:
            if hasattr(item, "model_dump"):
                itm = item.model_dump()
            elif isinstance(item, dict):
                itm = item
            else:
                try:
                    itm = json.loads(str(item))
                except Exception:
                    itm = {}
            normalized_outputs.append(itm)

            item_type = itm.get("type")
            if item_type == "message":
                for block in itm.get("content") or []:
                    if block.get("type") in ("text", "output_text"):
                        text_parts.append(block.get("text", ""))
            elif item_type == "function_call":
                function_calls.append(itm)
            elif item_type == "reasoning":
                reasoning_only = True

        # Persist only message/function artifacts; drop reasoning-only items from history
        for itm in normalized_outputs:
            if itm.get("type") in ("function_call", "function_call_output"):
                session.messages.append(itm)
            elif itm.get("role") in ("user", "assistant", "system"):
                session.messages.append({"role": itm.get("role"), "content": itm.get("content", "")})

        if function_calls:
            yield {"type": "status", "stage": "tool_call"}
            function_results: List[Dict[str, Any]] = []
            for fc in function_calls:
                call_id = fc.get("call_id") or fc.get("id")
                name = fc.get("name")
                args_raw = fc.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception as exc:
                    args = {}
                    result = {"kind": "error", "traceback": f"Bad arguments: {exc}"}
                    yield {"type": "tool_call", "call_id": call_id, "name": name, "arguments": args_raw}
                    yield {"type": "tool_result", "call_id": call_id, "result_kind": "error"}
                else:
                    yield {"type": "tool_call", "call_id": call_id, "name": name, "arguments": args}
                    tool = TOOL_REGISTRY.get(name)
                    if tool is None:
                        result = {"kind": "error", "traceback": f"Unknown tool {name}"}
                    else:
                        try:
                            result = tool.run(session, args)
                        except Exception as exc:
                            result = {"kind": "error", "traceback": repr(exc)}
                serialized = safe_tool_output(result)

                function_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": serialized,
                    }
                )
                yield {"type": "tool_result", "call_id": call_id, "result_kind": result.get("kind") if isinstance(result, dict) else None}

                try:
                    evt = _extract_data_event(result, session)
                    if evt:
                        last_data_event = evt
                except Exception:
                    logger.exception("data event extraction failed", extra={"session_id": session.session_id})

            session.messages.extend(function_results)
            continue

        # If we only got reasoning and no message/tool call, request another turn.
        if reasoning_only and not text_parts:
            yield {"type": "status", "stage": "reasoning_only"}
            continue

        # Emit partials for each text block (approximate streaming)
        for part in text_parts:
            if part:
                yield {"type": "partial", "text": part}

        message_text = "\n".join([p for p in text_parts if p])
        reply_text = message_text if message_text.strip() else reply_text
        if not reply_text.strip():
            reply_text = "No content returned."
        session.messages.append({"role": "assistant", "content": reply_text})
        if last_data_event:
            yield {"type": last_data_event.get("type"), "payload": last_data_event.get("payload")}
        yield {"type": "reply", "text": reply_text, "status": "ok", "total_tokens": reply_tokens}
        reply_sent = True
        break

    if not reply_sent:
        fallback = reply_text if reply_text.strip() else "No content returned."
        yield {"type": "reply", "text": fallback, "status": "ok", "total_tokens": reply_tokens}

    if over_limit_after_turn:
        logger.warning("token budget hit post-turn; compacted", extra={"session_id": session.session_id})
        compact_session(session)

    yield {"type": "status", "stage": "done"}


def system_prompt() -> str:
    return (
        "You are Codex, a coding/data agent, developed by Actalyst. Apply these rules:\n"
        "- Answer only questions related to the provided data; ask for refinement if unclear.\n"
        "- Use tools for computation/data access; prefer python_repl for analysis. Discover CSV names with list_csv_files. First inspect with load_csv_sample (nrows default 200). When ready, use load_csv_columns to load all rows of just the needed columns. Use load_csv_handle only if you must load the entire table.\n"
        "- Tables are delivered to the UI via data_frame/data_download events. Never paste large tables in your final message. Include a Markdown table only when the result has 20 rows or fewer; otherwise summarize with aggregates and say the data is too large to inline, suggesting helpful aggregations/filters.\n"
        "- Keep messages concise; highlight findings and aggregates, not code.\n"
        "- When a tool fails, report briefly and propose a fix.\n"
        "- Stop calling tools once you have the answer.\n"
        "- Default to ASCII; avoid non-essential Unicode.\n"
        "- Ask for missing info or column names when needed; suggest the most relevant column.\n"
        "- Derived columns from computations are allowed; never fabricate data.\n"
        "- For each question, assess required columns from current context; do not reuse old columns unless relevant or loaded.\n"
        "- In final summaries use Markdown with headings/bullets; include a small table only when <=20 rows; otherwise omit tables and point to the Data tab or suggest grouping/filtering to reduce size.\n"
        "   - The summary should essentially capture and answer the user question or clarify requirements etc. Do'nt say The full table is available in the Data tab as 'OCT-25-26 cost by business unit', etc."
        "- The data tab shows full results via events(but don't say that in summary); the text summary is reserved for small outputs. Before replying, emit the final analysis dataframe via data_frame (use send_data_to_ui_as_df on your final df). If a result exceeds 20 rows, avoid inlining and recommend an aggregation (e.g., by category or time) to shrink it."
    )
