import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from agent import run_agent_turn, run_agent_turn_async
from session import SessionStore
from dispatcher import Dispatcher, Op, Event
import shutil

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

store = SessionStore()
dispatcher = Dispatcher(handler=None)  # set later
app = FastAPI(title="CSV Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


def sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.post("/upload")
async def upload_csv(file: UploadFile = File(None), path: str = Form(None), session_id: str = Form(None)):
    if not file and not path:
        raise HTTPException(status_code=400, detail="Provide a file or a path.")

    session_reset = False
    if not session_id:
        session = store.new()
    else:
        # Reset session to ensure a clean chat when replacing data
        store.reset(session_id)
        session = store.get(session_id)
        session_reset = True

    max_bytes = 400 * 1024 * 1024  # 400 MB limit

    if file:
        dest_dir = DATA_ROOT / session.session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.filename

        def _save():
            bytes_written = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB
            with dest_path.open("wb") as out_f:
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise HTTPException(status_code=413, detail="CSV too large; max 400 MB.")
                    out_f.write(chunk)

        try:
            await run_in_threadpool(_save)
        except HTTPException:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            raise

        logical_name = Path(file.filename).stem
    else:
        dest_path = Path(path).expanduser().resolve()
        if not dest_path.exists():
            raise HTTPException(status_code=404, detail="Path not found.")
        if dest_path.stat().st_size > max_bytes:
            raise HTTPException(status_code=413, detail="CSV too large; max 400 MB.")
        logical_name = dest_path.stem

    session.csv_registry.clear()
    session.csv_registry[logical_name] = str(dest_path)
    # Hint to the model about newly available data
    session.messages = [
        {
            "role": "assistant",
            "content": f"CSV '{logical_name}' registered and ready. Use load_csv_handle('{logical_name}', '<df_name>') to load it for analysis.",
        }
    ]
    session.total_tokens = 0
    session.summary = ""
    return {
        "session_id": session.session_id,
        "csv_name": logical_name,
        "path": str(dest_path),
        "session_reset": session_reset,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    session = store.get(req.session_id) if req.session_id else store.new()
    result = run_agent_turn(session, req.message)
    return {
        "session_id": session.session_id,
        "reply": result["reply"],
        "total_tokens": result["total_tokens"],
        "status": result["status"],
        "data_events": result.get("data_events", []),
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    SSE stream: send a simple start -> reply -> done sequence (no partial streaming).
    """
    session = store.get(req.session_id) if req.session_id else store.new()

    async def event_generator():
        yield sse_event("status", "start")
        # Use synchronous agent to avoid partial-stream issues
        result = run_agent_turn(session, req.message)
        for evt in result.get("data_events", []):
            yield sse_event(evt.get("type", "data_frame"), json.dumps({"payload": evt.get("payload")}))
        yield sse_event(
            "token_usage",
            json.dumps(
                {
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "total_tokens": result.get("total_tokens"),
                }
            ),
        )
        yield sse_event("reply", json.dumps({"text": result["reply"], "status": result["status"], "total_tokens": result["total_tokens"]}))
        yield sse_event("done", json.dumps({"status": result.get("status", "ok")}))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reset")
async def reset(session_id: str):
    store.reset(session_id)
    exports_dir = DATA_ROOT / session_id / "exports"
    if exports_dir.exists():
        shutil.rmtree(exports_dir, ignore_errors=True)
    return {"session_id": session_id, "reset": True}


@app.get("/health")
async def health():
    return {"ok": True}


# Dispatcher handler that wraps run_agent_turn synchronously for now
async def handle_op(op: Op):
    events: List[Event] = []
    if op.kind == "user_message":
        session = store.get(op.session_id) if op.session_id else store.new()
        async for ev in run_agent_turn_async(session, op.payload.get("message", "")):
            etype = ev.get("type")
            if etype == "reply":
                events.append(Event(session.session_id, "reply", {"text": ev["text"], "status": ev["status"]}))
                events.append(Event(session.session_id, "done", {"status": ev["status"], "total_tokens": ev["total_tokens"]}))
            elif etype == "status":
                events.append(Event(session.session_id, "status", {"stage": ev.get("stage")}))
            elif etype == "tool_call":
                events.append(Event(session.session_id, "tool_call", {"call_id": ev.get("call_id"), "name": ev.get("name"), "arguments": ev.get("arguments")}))
            elif etype == "tool_result":
                events.append(Event(session.session_id, "tool_result", {"call_id": ev.get("call_id"), "result_kind": ev.get("result_kind")}))
            elif etype == "token_usage":
                events.append(Event(session.session_id, "token_usage", {"input_tokens": ev.get("input_tokens"), "output_tokens": ev.get("output_tokens"), "total_tokens": ev.get("total_tokens")}))
            elif etype == "partial":
                events.append(Event(session.session_id, "partial", {"text": ev.get("text")}))
            elif etype == "error":
                events.append(Event(session.session_id, "error", {"message": ev["message"]}))
            elif etype in ("data_frame", "data_download", "chart", "chart_rejected"):
                events.append(Event(session.session_id, etype, {"payload": ev.get("payload")}))
    return events


dispatcher.handler = handle_op


@app.on_event("startup")
async def on_startup():
    await dispatcher.start()


@app.on_event("shutdown")
async def on_shutdown():
    await dispatcher.stop()
