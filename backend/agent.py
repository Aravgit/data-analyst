import gc
import json
import logging
import random
import time
from typing import Dict, Any, Optional, List, Iterable
import os
import asyncio
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import Response

from session import SessionState, PythonSession
from pydantic import BaseModel
from tools import TOOL_REGISTRY, TOOLING_SPEC, safe_tool_output
from data_events import make_data_frame_event, make_chart_event, encrypt_payload

MODEL = "gpt-5.1"
REASONING_EFFORT = "medium"
TOKEN_LIMIT = 100000  # combined input + output (hard budget)
TOKEN_WATERMARK = int(TOKEN_LIMIT * 0.8)  # 80k - trigger compaction before hitting limit
MAX_API_RETRIES = 3
# Streaming of partial chunks is temporarily disabled; we emit only final replies.
USE_STREAMING = bool(int(os.getenv("ENABLE_RESPONSES_STREAMING", "0")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data")).resolve()

logger = logging.getLogger("csv_agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _reset_repl_after_turn(session: SessionState) -> None:
    """Reset REPL after each turn, ensuring memory is freed."""
    # Explicitly clear old session's memory before creating new one
    try:
        session.python._reset_globals()
    except Exception:
        pass
    # Create fresh session
    session.python = PythonSession()
    # Force immediate garbage collection to free memory
    gc.collect()
    logger.info("repl reset after turn", extra={"session_id": session.session_id})


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


def _extract_data_events(result: Any, session: SessionState) -> List[Dict[str, Any]]:
    """
    Extract data-related events (tables, charts) from a tool result.
    """
    events: List[Dict[str, Any]] = []
    if not isinstance(result, dict):
        return events

    # DataFrame event
    if result.get("kind") == "dataframe":
        df_name = result.get("df_name", "df")
        logical = result.get("logical_name") or df_name
        df = session.python.globals.get(df_name)
        if df is not None:
            try:
                evts, _ = make_data_frame_event(df, df_name, session.session_id, logical, DATA_ROOT)
                events.extend(evts)
            except Exception:
                pass
        else:
            head = result.get("head") or []
            payload = {
                "df_name": df_name,
                "logical_name": logical,
                "columns": result.get("columns") or [],
                "rows": head,
                "row_count": result.get("row_count"),
                "dtypes": result.get("dtypes") or {},
            }
            events.append({"type": "data_frame", "payload": encrypt_payload(payload)})

    # Chart event if provided
    chart_spec = result.get("chart") or result.get("chart_spec")
    if chart_spec:
        df_for_chart = None
        df_name = result.get("df_name") or (chart_spec.get("df_name") if isinstance(chart_spec, dict) else None)
        if df_name:
            df_for_chart = session.python.globals.get(df_name)
        if df_for_chart is None and isinstance(result.get("head"), list):
            try:
                import pandas as pd

                df_for_chart = pd.DataFrame(result.get("head"))
            except Exception:
                df_for_chart = None
        if df_for_chart is not None:
            try:
                chart_events, reject_reason = make_chart_event(df_for_chart, chart_spec, session.session_id, result.get("logical_name"))
                if chart_events:
                    events.extend(chart_events)
                elif reject_reason:
                    events.append({"type": "chart_rejected", "payload": encrypt_payload({"reason": reject_reason})})
            except Exception:
                pass

    # Direct chart-only result that already has payload
    if result.get("kind") == "chart" and result.get("payload"):
        events.append({"type": "chart", "payload": result.get("payload")})

    return events


def _make_data_frame_events_from_result(result: Dict[str, Any], session: SessionState) -> List[Dict[str, Any]]:
    if not isinstance(result, dict) or result.get("kind") != "dataframe":
        return []

    df_name = result.get("df_name", "df")
    logical = result.get("logical_name") or df_name
    df = session.python.globals.get(df_name)
    if df is None:
        df = session.python.consume_handoff_df(df_name)
    if df is not None:
        try:
            evts, _ = make_data_frame_event(df, df_name, session.session_id, logical, DATA_ROOT)
            return evts
        except Exception:
            return []

    head = result.get("head") or []
    payload = {
        "df_name": df_name,
        "logical_name": logical,
        "columns": result.get("columns") or [],
        "rows": head,
        "row_count": result.get("row_count"),
        "dtypes": result.get("dtypes") or {},
    }
    return [{"type": "data_frame", "payload": encrypt_payload(payload)}]


def _safe_item_type(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _safe_item_attr(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _safe_content_blocks(item: Any) -> List[Any]:
    content = _safe_item_attr(item, "content") or []
    if isinstance(content, list):
        return content
    return [content]


def _safe_output_items(obj: Any) -> List[Any]:
    if isinstance(obj, dict):
        return obj.get("output") or []
    output = getattr(obj, "output", None)
    return output or []


def run_agent_turn(
    session: SessionState, user_message: str, max_turns: int = 12
) -> Dict[str, Any]:
    """
    Execute the agent loop for one user message, respecting the token budget.
    Returns: {"reply": str, "total_tokens": int, "status": "ok"|"token_limit"}
    """
    try:
        if session.pending_compaction or session.total_tokens >= TOKEN_WATERMARK:
            compact_session(session)
            session.pending_compaction = False
            logger.info(
                "token watermark reached, pre-turn compact",
                extra={"session_id": session.session_id, "tokens": session.total_tokens, "watermark": TOKEN_WATERMARK},
            )
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
        data_events: List[Dict[str, Any]] = []
        last_df_result: Optional[Dict[str, Any]] = None

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
                    break
                except Exception as exc:  # network/transient failures
                    last_error = exc
                    logger.exception(
                        "model call failed",
                        extra={"session_id": session.session_id, "attempt": attempt, "turn": turns},
                    )
                    if attempt == MAX_API_RETRIES:
                        # Remove failed user message from history to prevent corruption
                        if session.messages and session.messages[-1].get("role") == "user":
                            session.messages.pop()
                        reply_text = f"Upstream model call failed after {MAX_API_RETRIES} attempts: {exc}"
                        return {
                            "reply": reply_text,
                            "total_tokens": session.total_tokens,
                            "status": "error",
                            "data_events": [],
                        }
                    # Exponential backoff with jitter before retry
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"retrying after {backoff:.2f}s", extra={"session_id": session.session_id})
                    time.sleep(backoff)

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

            if session.total_tokens >= TOKEN_WATERMARK:
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
                item_type = _safe_item_type(item)
                if item_type == "message":
                    for block in _safe_content_blocks(item):
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            block_text = block.get("text", "")
                        else:
                            block_type = getattr(block, "type", None)
                            block_text = getattr(block, "text", "")
                        if block_type in ("text", "output_text"):
                            text_parts.append(block_text)
                elif item_type == "function_call":
                    function_calls.append(
                        {
                            "type": "function_call",
                            "call_id": _safe_item_attr(item, "call_id") or _safe_item_attr(item, "id"),
                            "id": _safe_item_attr(item, "id"),
                            "name": _safe_item_attr(item, "name"),
                            "arguments": _safe_item_attr(item, "arguments") or "{}",
                        }
                    )
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

                    # Extract data events from ALL tool results (not just the last)
                    if isinstance(result, dict):
                        tool_events = _extract_data_events(result, session)
                        if tool_events:
                            data_events.extend(tool_events)
                            logger.info(
                                "data events extracted",
                                extra={
                                    "session_id": session.session_id,
                                    "tool": name,
                                    "event_count": len(tool_events),
                                },
                            )
                        # Also track last df result for fallback event generation
                        if result.get("kind") == "dataframe" and name in ("python_repl", "send_data_to_ui_as_df"):
                            last_df_result = result

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
                    item_type = _safe_item_type(out)
                    if item_type == "message":
                        for blk in _safe_content_blocks(out):
                            if isinstance(blk, dict):
                                blk_type = blk.get("type")
                                blk_text = blk.get("text", "")
                            else:
                                blk_type = getattr(blk, "type", None)
                                blk_text = getattr(blk, "text", "")
                            if blk_type in ("text", "output_text"):
                                parts.append(blk_text)
                merged = "\n".join([p for p in parts if p]).strip()
                if merged:
                    reply_text = merged
                    session.messages.append({"role": "assistant", "content": reply_text})
            except Exception:
                pass

        if not reply_text.strip():
            reply_text = "I wasn't able to produce a final answer after tool calls. Please ask again or try a simpler request."

        if over_limit_after_turn:
            logger.info(
                "token watermark reached; compacting next turn",
                extra={"session_id": session.session_id, "tokens": session.total_tokens},
            )
            session.pending_compaction = True
        status = "ok"

        # Fallback: if no data events were extracted during tool execution,
        # try to generate from the last df result
        if not data_events and last_df_result:
            final_events = _make_data_frame_events_from_result(last_df_result, session)
            if final_events:
                data_events = final_events

        return {
            "reply": reply_text,
            "total_tokens": reply_tokens,
            "input_tokens": turn_input_tokens,
            "output_tokens": turn_output_tokens,
            "status": status,
            "data_events": data_events,
        }
    finally:
        _reset_repl_after_turn(session)


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
    if session.pending_compaction or session.total_tokens >= TOKEN_WATERMARK:
        compact_session(session)
        session.pending_compaction = False
        logger.info(
            "token watermark reached pre-turn; compacted",
            extra={"session_id": session.session_id, "tokens": session.total_tokens, "watermark": TOKEN_WATERMARK},
        )
    session.messages.append({"role": "user", "content": user_message})
    yield {"type": "status", "stage": "start"}

    turns = 0
    reply_text = ""
    reply_sent = False
    over_limit_after_turn = False
    reply_tokens = 0
    data_events: List[Dict[str, Any]] = []
    last_df_result: Optional[Dict[str, Any]] = None

    while turns < max_turns:
        turns += 1
        yield {"type": "status", "stage": "model_call"}
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
                        "streaming": USE_STREAMING,
                    },
                )
                if USE_STREAMING:
                    stream = await async_client.responses.create(
                        model=MODEL,
                        reasoning={"effort": REASONING_EFFORT},
                        input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                        tools=TOOLING_SPEC,
                        max_output_tokens=2000,
                        stream=True,
                    )
                    async for chunk in stream:
                        outputs = _safe_output_items(chunk)
                        for out in outputs:
                            if _safe_item_type(out) == "message":
                                for blk in _safe_content_blocks(out):
                                    if isinstance(blk, dict):
                                        blk_type = blk.get("type")
                                        blk_text = blk.get("text", "")
                                    else:
                                        blk_type = getattr(blk, "type", None)
                                        blk_text = getattr(blk, "text", "")
                                    if blk_type in ("text", "output_text") and blk_text:
                                        yield {"type": "partial", "text": blk_text}
                    # After streaming partials, fetch a full response (with usage) non-streaming
                    response = await async_client.responses.create(
                        model=MODEL,
                        reasoning={"effort": REASONING_EFFORT},
                        input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                        tools=TOOLING_SPEC,
                        max_output_tokens=2000,
                    )
                else:
                    response = await async_client.responses.create(
                        model=MODEL,
                        reasoning={"effort": REASONING_EFFORT},
                        input=[{"role": "system", "content": system_prompt()}, *filtered_inputs],
                        tools=TOOLING_SPEC,
                        max_output_tokens=2000,
                    )
                break
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "model call failed",
                    extra={"session_id": session.session_id, "attempt": attempt, "turn": turns},
                )
                if attempt == MAX_API_RETRIES:
                    # Remove failed user message from history to prevent corruption
                    if session.messages and session.messages[-1].get("role") == "user":
                        session.messages.pop()
                    yield {"type": "error", "message": f"Model call failed after {MAX_API_RETRIES} attempts: {exc}"}
                    _reset_repl_after_turn(session)
                    return
                # Exponential backoff with jitter before retry
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"retrying after {backoff:.2f}s", extra={"session_id": session.session_id})
                await asyncio.sleep(backoff)

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

        if session.total_tokens >= TOKEN_WATERMARK:
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
            item_type = _safe_item_type(item)
            normalized_outputs.append(item if isinstance(item, dict) else {"type": item_type})
            if item_type == "message":
                for block in _safe_content_blocks(item):
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        block_text = block.get("text", "")
                    else:
                        block_type = getattr(block, "type", None)
                        block_text = getattr(block, "text", "")
                    if block_type in ("text", "output_text"):
                        text_parts.append(block_text)
            elif item_type == "function_call":
                function_calls.append(
                    {
                        "type": "function_call",
                        "call_id": _safe_item_attr(item, "call_id") or _safe_item_attr(item, "id"),
                        "id": _safe_item_attr(item, "id"),
                        "name": _safe_item_attr(item, "name"),
                        "arguments": _safe_item_attr(item, "arguments") or "{}",
                    }
                )
            elif item_type == "reasoning":
                reasoning_only = True

        if function_calls:
            yield {"type": "status", "stage": "tool_call"}
            function_results: List[Dict[str, Any]] = []
            for fc in function_calls:
                call_id = fc.get("call_id") or fc.get("id")
                name = fc.get("name")
                args_raw = fc.get("arguments", "{}")
                session.messages.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": args_raw,
                    }
                )
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

                # Extract data events from ALL tool results (not just the last)
                if isinstance(result, dict):
                    tool_events = _extract_data_events(result, session)
                    if tool_events:
                        data_events.extend(tool_events)
                    # Also track last df result for fallback
                    if result.get("kind") == "dataframe" and name in ("python_repl", "send_data_to_ui_as_df"):
                        last_df_result = result

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

        # Fallback: if no data events were extracted during tool execution,
        # try to generate from the last df result
        if not data_events and last_df_result:
            final_events = _make_data_frame_events_from_result(last_df_result, session)
            if final_events:
                data_events = final_events

        for evt in data_events:
            yield {"type": evt.get("type"), "payload": evt.get("payload")}
        yield {"type": "reply", "text": reply_text, "status": "ok", "total_tokens": reply_tokens}
        reply_sent = True
        break

    if not reply_sent:
        fallback = reply_text if reply_text.strip() else "No content returned."
        yield {"type": "reply", "text": fallback, "status": "ok", "total_tokens": reply_tokens}

    if over_limit_after_turn:
        logger.info(
            "token watermark reached; compacting next turn",
            extra={"session_id": session.session_id, "tokens": session.total_tokens},
        )
        session.pending_compaction = True

    _reset_repl_after_turn(session)
    yield {"type": "status", "stage": "done"}


def system_prompt() -> str:
    return """You are Codex, a data analysis agent by Actalyst.

DATA LOADING STRATEGY (follow this order):
1. get_dataset_info(name) - Get schema (columns, dtypes, row_count) first
2. load_dataset_sample(name, df_name, nrows=100) - For schema discovery / example values ONLY
3. load_dataset_columns(name, columns, df_name) - PRIMARY TOOL: Load ONLY the columns needed for analysis
4. load_dataset_full(name, df_name) - RARE: Only when ALL columns are genuinely required

COLUMN SELECTION (you decide automatically):
- Analyze the user's question to determine minimal columns needed
- Examples:
  - "total sales by region" -> load_dataset_columns(['region', 'sales'])
  - "filter by date and sum revenue" -> load_dataset_columns(['date', 'revenue'])
  - "show customers in APAC" -> load_dataset_columns(['customer_name', 'region'])
- When unsure, load fewer columns first; expand only if needed
- NEVER load all columns unless the analysis genuinely requires it

VALUE DISCOVERY:
- If user provides a specific value (ID, code, name), use find_value_columns to locate which column contains it
- Only ask the user if still ambiguous after inspection

EFFICIENCY AND ACCURACY:
- Minimize token usage to reduce cost, without sacrificing analytical accuracy.
- Ensure the user sees correct and accurate data for their question.

OUTPUT:
- Use send_data_to_ui_as_df for result tables (auto-truncated to 2000 rows)
- Use send_chart_to_ui for visualizations (bar/column/stacked_column/line/scatter, max 200 rows, max 10 series)
- Keep text responses concise; highlight insights and findings
- Never paste large tables in text; reference the Data tab instead
- Only describe a chart if you called send_chart_to_ui in this turn
- Include Markdown table only when result has <=20 rows

MEMORY: DataFrames are cleared after each response. Load only what you need, analyze, emit results.

OTHER RULES:
- Answer only questions related to the provided data
- Use python_repl for calculations after loading data
- When a tool fails, report briefly and propose a fix
- Stop calling tools once you have the answer
- Derived columns from computations are allowed; never fabricate data
- Ask clarifying questions only when the user ask is very vague or unclear, and treat this as a last resort
"""
