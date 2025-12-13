import signal
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import pandas as pd

MAX_MEMORY_BYTES = 1_000_000_000  # 1 GB
MAX_SUBPROCESSES = 5
EXEC_TIMEOUT_SECONDS = 10 * 60


@dataclass
class PythonSession:
    """Lightweight, persistent Python REPL per user session."""

    globals: Dict[str, Any] = field(default_factory=dict)

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
        self.globals.update(base)

    def run(self, code: str) -> Dict[str, Any]:
        import ast
        import traceback
        from time import monotonic

        try:
            print("Python code is \n\n",code,"\n***********\n")
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

            result = serialize_value(value)
            result.setdefault("meta", {})
            result["meta"]["execution_seconds"] = elapsed
            return result
        except TimeoutError:
            return {"kind": "error", "traceback": "Execution timed out after 15 minutes."}
        except Exception:
            return {"kind": "error", "traceback": traceback.format_exc()}


def serialize_value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"kind": "text", "text": "Code executed. No return value."}

    if isinstance(value, pd.DataFrame):
        head_df = value.head(20)
        head_records = head_df.to_dict(orient="records")
        return {
            "kind": "dataframe",
            "repr": repr(head_df),
            "head": head_records,
            "columns": list(value.columns),
            "row_count": int(len(value)),
            "meta": {"dtypes": {c: str(t) for c, t in value.dtypes.items()}},
        }

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
    csv_registry: Dict[str, str] = field(default_factory=dict)
    summary: str = ""


class SessionStore:
    def __init__(self):
        self._store: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._store:
            self._store[session_id] = SessionState(session_id=session_id)
        return self._store[session_id]

    def new(self) -> SessionState:
        state = SessionState()
        self._store[state.session_id] = state
        return state

    def reset(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id] = SessionState(session_id=session_id)
