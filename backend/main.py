import os
import multiprocessing as mp

# Reduce thread usage in constrained containers to avoid pthread_create failures.
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("ARROW_IO_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("PANDAS_USE_NUMEXPR", "0")
import faulthandler
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
faulthandler.enable()
try:
    mp.set_start_method("spawn", force=True)
except Exception:
    pass

# CORS configuration - allow_credentials=True with allow_origins=["*"] is invalid
# Use specific origins in production or disable credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


def sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _convert_csv_to_parquet(src_path: Path, parquet_path: Path) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    try:
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except Exception:
        pass

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    compression = os.getenv("PARQUET_COMPRESSION", "zstd")
    tmp_path = parquet_path.with_suffix(".parquet.tmp")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        row_count = 0
        schema = None
        writer = None
        success = False
        try:
            read_opts = pacsv.ReadOptions(use_threads=False)
            reader = pacsv.open_csv(src_path, read_options=read_opts)
            for batch in reader:
                if writer is None:
                    schema = batch.schema
                    writer = pq.ParquetWriter(tmp_path, schema, compression=compression)
                writer.write_table(pa.Table.from_batches([batch]))
                row_count += batch.num_rows
            if writer is not None:
                writer.close()
                writer = None
            if schema is not None:
                success = True
        except Exception as exc:
            last_error = exc
            # Fallback to pandas for formats pyarrow can't parse
            try:
                import pandas as pd
                import pyarrow as pa
                import pyarrow.parquet as pq

                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    writer = None
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

                row_count = 0
                schema = None
                chunk_rows = int(os.getenv("CSV_PANDAS_CHUNK_ROWS", "100000"))
                engine = os.getenv("PANDAS_CSV_ENGINE", "python")
                # low_memory is not supported with python engine
                read_kwargs = {"chunksize": chunk_rows, "engine": engine}
                if engine != "python":
                    read_kwargs["low_memory"] = False
                reader = pd.read_csv(src_path, **read_kwargs)
                for chunk in reader:
                    if schema is None:
                        schema = pa.Schema.from_pandas(chunk, preserve_index=False)
                        writer = pq.ParquetWriter(tmp_path, schema, compression=compression)
                    try:
                        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                    except Exception:
                        table = pa.Table.from_pandas(chunk, preserve_index=False)
                        if table.schema != schema:
                            table = table.cast(schema, safe=False)
                    writer.write_table(table)
                    row_count += int(len(chunk))

                if writer is not None:
                    writer.close()
                    writer = None
                if schema is not None:
                    success = True
                    last_error = None
            except Exception as pandas_exc:
                last_error = pandas_exc
        finally:
            if writer is not None:
                writer.close()

        if not success:
            last_error = last_error or RuntimeError("Parquet conversion failed with empty schema.")
        else:
            tmp_path.replace(parquet_path)
            columns = list(schema.names)
            dtypes = {name: str(schema.field(name).type) for name in schema.names}
            return {"row_count": row_count, "columns": columns, "dtypes": dtypes}

    raise RuntimeError(f"Parquet conversion failed after 3 attempts: {last_error}")


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
        session = store.get_or_create(session_id)
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
        # Path traversal validation: ensure path is within DATA_ROOT or is an absolute path
        # that the user explicitly provided (not trying to access system files)
        try:
            # Check if path exists and is a regular file
            if not dest_path.exists():
                raise HTTPException(status_code=404, detail="Path not found.")
            if not dest_path.is_file():
                raise HTTPException(status_code=400, detail="Path must be a regular file.")
            # Reject paths that try to access sensitive system directories
            sensitive_dirs = ["/etc", "/var", "/usr", "/bin", "/sbin", "/root", "/home"]
            path_str = str(dest_path)
            for sensitive in sensitive_dirs:
                if path_str.startswith(sensitive + "/") or path_str == sensitive:
                    raise HTTPException(status_code=403, detail="Access to this path is not allowed.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid path: {e}")
        if dest_path.stat().st_size > max_bytes:
            raise HTTPException(status_code=413, detail="CSV too large; max 400 MB.")
        logical_name = dest_path.stem

    parquet_dir = DATA_ROOT / session.session_id
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{logical_name}.parquet"

    try:
        meta = await run_in_threadpool(_convert_csv_to_parquet, dest_path, parquet_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to convert CSV to parquet: {exc}")

    session.csv_registry.clear()
    csv_bytes = dest_path.stat().st_size if dest_path.exists() else None
    parquet_bytes = parquet_path.stat().st_size if parquet_path.exists() else None

    session.csv_registry[logical_name] = {
        "csv_path": str(dest_path),
        "parquet_path": str(parquet_path),
        "row_count": meta.get("row_count"),
        "columns": meta.get("columns"),
        "dtypes": meta.get("dtypes"),
        "csv_bytes": csv_bytes,
        "parquet_bytes": parquet_bytes,
    }
    # Hint to the model about newly available data
    session.messages = [
        {
            "role": "assistant",
            "content": f"Dataset '{logical_name}' registered and ready. It has been converted to parquet for full-data analysis. Use load_dataset_full('{logical_name}', '<df_name>') if you need the entire table.",
        }
    ]
    session.total_tokens = 0
    session.summary = ""
    return {
        "session_id": session.session_id,
        "csv_name": logical_name,
        "path": str(dest_path),
        "parquet_path": str(parquet_path),
        "csv_bytes": csv_bytes,
        "parquet_bytes": parquet_bytes,
        "session_reset": session_reset,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id:
        session = store.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    else:
        session = store.new()
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
    if req.session_id:
        session = store.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    else:
        session = store.new()

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
    return {"ok": True, "sessions": store.stats()}


# Dispatcher handler that wraps run_agent_turn synchronously for now
async def handle_op(op: Op):
    events: List[Event] = []
    if op.kind == "user_message":
        if op.session_id:
            session = store.get(op.session_id)
            if session is None:
                events.append(Event(op.session_id, "error", {"message": "Session not found"}))
                return events
        else:
            session = store.new()
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
