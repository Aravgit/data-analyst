import signal
import os
import gc
import shutil
import time
import uuid
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MAX_MEMORY_BYTES = 2_000_000_000  # 2 GB
MAX_SUBPROCESSES = 8
EXEC_TIMEOUT_SECONDS = 10 * 60
PREVIEW_ROWS = int(os.getenv("TOOL_DF_PREVIEW_ROWS", "20"))
PREVIEW_COLS = int(os.getenv("TOOL_DF_PREVIEW_COLS", "200"))
MAX_REPL_DF_BYTES = int(os.getenv("REPL_DF_BYTES", "500000000"))  # 500 MB cap for in-turn dataframe retention

# Session management limits
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))  # Max concurrent sessions
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour default
DATA_ROOT = os.getenv("DATA_ROOT", "./data")


@dataclass
class PythonSession:
    """Lightweight, persistent Python REPL per user session."""

    globals: Dict[str, Any] = field(default_factory=dict)
    _base_globals: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    handoff_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    handoff_df_name: str | None = field(default=None, init=False, repr=False)
    _df_counter: int = field(default=0, init=False, repr=False)
    _df_meta: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        base = {
            "__name__": "__repl__",
            "__builtins__": __builtins__,
            "pd": pd,
        }
        try:
            import numpy as np
            base["np"] = np
        except Exception:
            pass
        try:
            pd.set_option("compute.use_numexpr", False)
        except Exception:
            pass
        self._base_globals = dict(base)
        self.globals.update(base)

    def _reset_globals(self) -> None:
        """Clear all user-defined variables, explicitly freeing DataFrames."""
        # Pop and delete each user object to break reference cycles
        for key in list(self.globals.keys()):
            if key not in self._base_globals:
                obj = self.globals.pop(key, None)
                # Explicit delete to help GC
                del obj
        self.globals.clear()
        self.globals.update(self._base_globals)
        self._df_meta.clear()
        # Clear handoff DataFrame
        if self.handoff_df is not None:
            del self.handoff_df
        self.handoff_df = None
        self.handoff_df_name = None
        # Force garbage collection to free memory immediately
        gc.collect()

    def set_handoff_df(self, df: pd.DataFrame, name: str) -> None:
        self.handoff_df = df
        self.handoff_df_name = name

    def consume_handoff_df(self, name: str | None = None) -> pd.DataFrame | None:
        if self.handoff_df is None:
            return None
        # "analysis_df" is a special name that always matches the handoff result
        if name and name != "analysis_df" and self.handoff_df_name and name != self.handoff_df_name:
            return None
        df = self.handoff_df
        self.handoff_df = None
        self.handoff_df_name = None
        return df

    def _estimate_df_bytes(self, df: pd.DataFrame) -> int:
        try:
            return int(df.memory_usage(deep=True).sum())
        except Exception:
            return 0

    def _refresh_df_meta(self) -> None:
        for key, value in list(self.globals.items()):
            if not isinstance(value, pd.DataFrame):
                continue
            df_id = id(value)
            meta = self._df_meta.get(key)
            if meta is None or meta.get("id") != df_id:
                self._df_counter += 1
                self._df_meta[key] = {"id": df_id, "ts": self._df_counter, "bytes": self._estimate_df_bytes(value)}
            else:
                meta["bytes"] = self._estimate_df_bytes(value)

        # Drop metadata for removed globals
        for key in list(self._df_meta.keys()):
            if key not in self.globals or not isinstance(self.globals.get(key), pd.DataFrame):
                self._df_meta.pop(key, None)

    def prune_dataframes(self, keep_names: set[str] | None = None) -> None:
        if MAX_REPL_DF_BYTES <= 0:
            return
        self._refresh_df_meta()
        total = sum(m.get("bytes", 0) for m in self._df_meta.values())
        if total <= MAX_REPL_DF_BYTES:
            return
        protected = keep_names or set()
        # Evict oldest dataframes first, skipping protected names
        candidates = sorted(
            ((name, meta) for name, meta in self._df_meta.items() if name not in protected),
            key=lambda item: item[1].get("ts", 0),
        )
        for name, _meta in candidates:
            if total <= MAX_REPL_DF_BYTES:
                break
            df = self.globals.pop(name, None)
            if isinstance(df, pd.DataFrame):
                total -= self._estimate_df_bytes(df)
            self._df_meta.pop(name, None)

    def run(self, code: str) -> Dict[str, Any]:
        import ast
        import traceback
        from time import monotonic

        try:
            parsed = ast.parse(code, mode="exec")
            body = parsed.body
            last_expr = None
            if body and isinstance(body[-1], ast.Expr):
                last_expr = body[-1].value
                body = body[:-1]

            with apply_resource_limits():
                start = monotonic()
                with execution_timeout(EXEC_TIMEOUT_SECONDS):
                    if body:
                        exec(
                            compile(ast.Module(body=body, type_ignores=[]), "<repl>", "exec"),
                            self.globals,
                            self.globals,
                        )

                    value = None
                    if last_expr is not None:
                        value = eval(
                            compile(ast.Expression(last_expr), "<repl>", "eval"),
                            self.globals,
                            self.globals,
                        )

                elapsed = monotonic() - start

            if isinstance(value, pd.DataFrame):
                self.set_handoff_df(value, "analysis_df")
            result = serialize_value(value)
            if isinstance(result, dict) and result.get("kind") == "dataframe" and "df_name" not in result:
                result["df_name"] = "analysis_df"
            result.setdefault("meta", {})
            result["meta"]["execution_seconds"] = elapsed
            # Keep dataframes around for this turn, but cap total retained memory.
            self.prune_dataframes(keep_names={"analysis_df"})
            return result
        except TimeoutError:
            return {"kind": "error", "traceback": "Execution timed out after 15 minutes."}
        except Exception:
            return {"kind": "error", "traceback": traceback.format_exc()}


def serialize_value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"kind": "text", "text": "Code executed. No return value."}

    if isinstance(value, pd.DataFrame):
        max_cols = PREVIEW_COLS if PREVIEW_COLS > 0 else len(value.columns)
        head_df = value.iloc[:, :max_cols].head(PREVIEW_ROWS if PREVIEW_ROWS > 0 else len(value))
        head_records = head_df.to_dict(orient="records")
        return {
            "kind": "dataframe",
            "repr": repr(head_df),
            "head": head_records,
            "columns": list(value.columns[:max_cols]),
            "row_count": int(len(value)),
            "meta": {
                "dtypes": {c: str(t) for c, t in value.dtypes.items()},
                "total_columns": int(len(value.columns)),
                "preview_columns": int(len(value.columns[:max_cols])),
                "preview_rows": int(len(head_df)),
            },
        }

    if isinstance(value, pd.Series):
        df = value.reset_index()
        df.columns = ["index", "value"]
        return serialize_value(df)

    if isinstance(value, dict):
        df = pd.DataFrame(list(value.items()), columns=["key", "value"])
        return serialize_value(df)

    if isinstance(value, (list, tuple)):
        df = pd.DataFrame(value)
        return serialize_value(df)

    # Fallback: wrap scalar into a 1-row dataframe for consistent table output
    try:
        df = pd.DataFrame([{"value": value}])
        return serialize_value(df)
    except Exception:
        pass

    try:
        import json
        json_value = json.loads(json.dumps(value, default=str))
    except Exception:
        json_value = None

    return {"kind": "object", "repr": repr(value), "json": json_value}


def _apply_limit(resource_name: str, limits: Tuple[int, int]):
    import resource

    res = getattr(resource, resource_name, None)
    if res is None:
        return None
    try:
        old = resource.getrlimit(res)
        resource.setrlimit(res, limits)
        return (res, old)
    except Exception:
        return None


@contextmanager
def apply_resource_limits():
    """Temporarily cap memory and subprocess creation for the REPL run."""
    import resource

    old_limits = []
    mem = _apply_limit("RLIMIT_AS", (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    if mem:
        old_limits.append(mem)
    procs = _apply_limit("RLIMIT_NPROC", (MAX_SUBPROCESSES, MAX_SUBPROCESSES))
    if procs:
        old_limits.append(procs)
    try:
        yield
    finally:
        for res, old in reversed(old_limits):
            try:
                resource.setrlimit(res, old)
            except Exception:
                pass


@contextmanager
def execution_timeout(seconds: int):
    """Raise TimeoutError if the block exceeds the given wall time."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum, _frame):
        raise TimeoutError()

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Each item is an input element for the Responses API (dicts only, never Pydantic objects)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    python: PythonSession = field(default_factory=PythonSession)
    total_tokens: int = 0
    pending_compaction: bool = False
    csv_registry: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class SessionStore:
    """Session store with TTL expiration and LRU eviction to prevent memory leaks."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_seconds: int = SESSION_TTL_SECONDS):
        self._store: Dict[str, SessionState] = {}
        self._access_times: Dict[str, float] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def get(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID. Returns None if not found or expired."""
        self._evict_expired()
        if session_id not in self._store:
            return None
        # Update access time
        self._access_times[session_id] = time.time()
        return self._store[session_id]

    def get_or_create(self, session_id: str) -> SessionState:
        """Get existing session or create a new one with the given ID."""
        self._evict_expired()
        if session_id not in self._store:
            self._store[session_id] = SessionState(session_id=session_id)
        self._access_times[session_id] = time.time()
        return self._store[session_id]

    def new(self) -> SessionState:
        """Create a new session with auto-generated ID."""
        self._evict_expired()
        state = SessionState()
        self._store[state.session_id] = state
        self._access_times[state.session_id] = time.time()
        return state

    def reset(self, session_id: str) -> None:
        """Reset a session, clearing history and REPL state."""
        if session_id in self._store:
            old_session = self._store[session_id]
            # Clean up old session's memory
            old_session.python._reset_globals()
            # Create fresh session
            self._store[session_id] = SessionState(session_id=session_id)
            self._access_times[session_id] = time.time()

    def _evict_expired(self) -> None:
        """Remove expired sessions and enforce max session limit."""
        now = time.time()

        # First, remove expired sessions (TTL exceeded)
        expired = [
            sid for sid, last_access in self._access_times.items()
            if now - last_access > self._ttl_seconds
        ]
        for sid in expired:
            self._cleanup_session(sid)
            logger.info(f"Session {sid[:8]}... expired after TTL")

        # Then, evict oldest if over limit (LRU)
        while len(self._store) > self._max_sessions:
            if not self._access_times:
                break
            # Find oldest accessed session
            oldest_sid = min(self._access_times, key=self._access_times.get)
            self._cleanup_session(oldest_sid)
            logger.info(f"Session {oldest_sid[:8]}... evicted (over limit)")

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up a session: free memory, delete files, remove from store."""
        if session_id in self._store:
            session = self._store[session_id]
            # Free REPL memory
            try:
                session.python._reset_globals()
            except Exception as e:
                logger.warning(f"Error resetting globals for {session_id[:8]}...: {e}")

            # Delete parquet files for this session
            session_data_dir = os.path.join(DATA_ROOT, session_id)
            if os.path.exists(session_data_dir):
                try:
                    shutil.rmtree(session_data_dir)
                    logger.debug(f"Deleted data directory for session {session_id[:8]}...")
                except Exception as e:
                    logger.warning(f"Error deleting data for {session_id[:8]}...: {e}")

            del self._store[session_id]

        self._access_times.pop(session_id, None)

        # Force garbage collection after cleanup
        gc.collect()

    def active_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._store)

    def stats(self) -> Dict[str, Any]:
        """Return session store statistics."""
        now = time.time()
        return {
            "active_sessions": len(self._store),
            "max_sessions": self._max_sessions,
            "ttl_seconds": self._ttl_seconds,
            "oldest_session_age": max(
                (now - t for t in self._access_times.values()),
                default=0
            ),
        }
