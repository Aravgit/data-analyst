import json
import datetime
import decimal
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import os

import numpy as np
import pandas as pd

from session import SessionState

MAX_TOOL_OUTPUT_CHARS = 8_192


def _default_serializer(obj: Any):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.DataFrame):
        return obj.head(50).to_dict(orient="records")
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def safe_tool_output(result: Any) -> str:
    try:
        raw = json.dumps(result, default=_default_serializer)
    except Exception as exc:
        raw = json.dumps({"kind": "error", "traceback": f"Serialization failed: {exc}"})
    if len(raw) > MAX_TOOL_OUTPUT_CHARS:
        raw = raw[:MAX_TOOL_OUTPUT_CHARS] + "...(truncated)"
    return raw


def python_repl_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "python_repl",
        "description": "Execute Python code in the session REPL. Use for calculations, pandas analysis, and plotting preparation. Avoid long-running loops.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run; can reuse prior variables.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def list_csv_files_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_csv_files",
        "description": "List logical CSV names uploaded/registered in this session.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }


def load_csv_handle_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_csv_handle",
        "description": "Load a registered CSV into pandas DataFrame in the REPL. Use before analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical CSV name"},
                "df_name": {"type": "string", "description": "Variable name in REPL"},
            },
            "required": ["name", "df_name"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def load_csv_sample_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_csv_sample",
        "description": "Load only the first N rows of a registered CSV into a pandas DataFrame for quick inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical CSV name"},
                "df_name": {"type": "string", "description": "Variable name in REPL", "default": "df_sample"},
                "nrows": {
                    "type": "integer",
                    "description": "Number of rows to load from the top of the file (default 200).",
                    "minimum": 1,
                    "maximum": 5000,
                },
            },
            "required": ["name", "df_name", "nrows"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def load_csv_columns_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_csv_columns",
        "description": "Load specific columns from a registered CSV into a pandas DataFrame (all rows).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical CSV name"},
                "df_name": {"type": "string", "description": "Variable name in REPL", "default": "df"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of columns to load",
                    "minItems": 1,
                },
            },
            # OpenAI function tool schemas require `required` to list every property when `strict` is true.
            "required": ["name", "df_name", "columns"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def send_chart_to_ui_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "send_chart_to_ui",
        "description": "Emit a chart spec for an existing DataFrame to the UI (bar/line/area/pie). Use after computing aggregates.",
        "parameters": {
            "type": "object",
            "properties": {
                "df_name": {"type": "string", "description": "Name of DataFrame in the REPL"},
                "logical_name": {"type": "string", "description": "Label to show in UI"},
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "area", "pie"],
                    "description": "Chart type (prefer bar/line; pie only for few categories)",
                },
                "x_field": {"type": "string", "description": "Column for x-axis or category"},
                "series": {
                    "type": "array",
                    "description": "List of series (y values) to plot",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Display name (defaults to y_field)"},
                            "y_field": {"type": "string", "description": "Numeric column for values"},
                        },
                        "required": ["name", "y_field"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "title": {"type": "string", "description": "Optional title"},
                "note": {"type": "string", "description": "Optional short note/footnote"},
            },
            "required": ["df_name", "chart_type", "x_field", "series"],
            "additionalProperties": False,
        },
        "strict": False,
    }


def send_df_to_ui_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "send_data_to_ui_as_df",
        "description": "Emit an existing DataFrame (by name) to the UI as the final data table.",
        "parameters": {
            "type": "object",
            "properties": {
                "df_name": {"type": "string", "description": "Name of DataFrame variable in the REPL"},
                "logical_name": {"type": "string", "description": "Label to show in UI", "default": "df"},
            },
            # Strict schemas must list every property in required
            "required": ["df_name", "logical_name"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def load_csv_into_session(session: SessionState, name: str, df_name: str) -> Dict[str, Any]:
    path = session.csv_registry.get(name)
    if not path:
        return {"kind": "error", "traceback": f"CSV '{name}' not registered."}
    try:
        try:
            df = pd.read_csv(path, low_memory=True)
        except MemoryError as exc:
            return {"kind": "error", "traceback": f"MemoryError loading CSV (sandbox ~1GB): {exc}"}
        session.python.globals[df_name] = df
        return {
            "kind": "dataframe",
            "text": f"Loaded {name} into {df_name} ({len(df)} rows).",
            "df_name": df_name,
            "logical_name": name,
            "head": df.head(50).to_dict(orient="records"),
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


def send_df_to_ui_as_df(session: SessionState, df_name: str, logical_name: str = "df") -> Dict[str, Any]:
    df = session.python.globals.get(df_name)
    if df is None:
        return {"kind": "error", "traceback": f"DataFrame '{df_name}' not found."}
    try:
        head = df.head(50).to_dict(orient="records")
        return {
            "kind": "dataframe",
            "text": f"Sent dataframe {df_name} to UI as {logical_name}. Rows: {len(df)}.",
            "df_name": df_name,
            "logical_name": logical_name,
            "head": head,
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


def load_csv_sample(session: SessionState, name: str, df_name: str = "df_sample", nrows: int = 200) -> Dict[str, Any]:
    path = session.csv_registry.get(name)
    if not path:
        return {"kind": "error", "traceback": f"CSV '{name}' not registered."}
    n = nrows if nrows and nrows > 0 else 200
    try:
        df = pd.read_csv(path, nrows=n, low_memory=True)
        session.python.globals[df_name] = df
        return {
            "kind": "dataframe",
            "text": f"Loaded sample of {n} rows from {name} into {df_name}.",
            "df_name": df_name,
            "logical_name": name,
            "head": df.head(50).to_dict(orient="records"),
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


def load_csv_columns(
    session: SessionState, name: str, columns: List[str], df_name: str = "df"
) -> Dict[str, Any]:
    path = session.csv_registry.get(name)
    if not path:
        return {"kind": "error", "traceback": f"CSV '{name}' not registered."}
    if not columns:
        return {"kind": "error", "traceback": "No columns provided."}
    try:
        df = pd.read_csv(path, usecols=columns, low_memory=True)
        session.python.globals[df_name] = df
        return {
            "kind": "dataframe",
            "text": f"Loaded columns {columns} from {name} into {df_name}. Rows loaded: {len(df)}.",
            "df_name": df_name,
            "logical_name": name,
            "head": df.head(50).to_dict(orient="records"),
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


@dataclass
class Tool:
    name: str
    schema: Dict[str, Any]
    run: Callable[[SessionState, Dict[str, Any]], Dict[str, Any]]


def _python_repl_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    return session.python.run(args.get("code", ""))


def _list_csv_run(session: SessionState, _args: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "object", "json": list(session.csv_registry.keys())}


def _load_csv_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    return load_csv_into_session(session, args.get("name", ""), args.get("df_name", "df"))


def _send_chart_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    df_name = args.get("df_name", "")
    df = session.python.globals.get(df_name)
    if df is None:
        return {"kind": "error", "traceback": f"DataFrame '{df_name}' not found."}
    return {
        "kind": "chart",
        "df_name": df_name,
        "logical_name": args.get("logical_name") or df_name,
        "chart_spec": {
            "title": args.get("title"),
            "chart_type": args.get("chart_type"),
            "x_field": args.get("x_field"),
            "series": args.get("series") or [],
            "note": args.get("note"),
            "df_name": df_name,
        },
    }


TOOL_REGISTRY: Dict[str, Tool] = {
    "python_repl": Tool("python_repl", python_repl_schema(), _python_repl_run),
    "list_csv_files": Tool("list_csv_files", list_csv_files_schema(), _list_csv_run),
    "load_csv_handle": Tool("load_csv_handle", load_csv_handle_schema(), _load_csv_run),
    "load_csv_sample": Tool("load_csv_sample", load_csv_sample_schema(), lambda s, a: load_csv_sample(s, a.get("name", ""), a.get("df_name", "df_sample"), a.get("nrows", 200))),
    "load_csv_columns": Tool(
        "load_csv_columns",
        load_csv_columns_schema(),
        lambda s, a: load_csv_columns(s, a.get("name", ""), a.get("columns", []), a.get("df_name", "df")),
    ),
    "send_data_to_ui_as_df": Tool(
        "send_data_to_ui_as_df",
        send_df_to_ui_schema(),
        lambda s, a: send_df_to_ui_as_df(s, a.get("df_name", ""), a.get("logical_name", "df")),
    ),
    "send_chart_to_ui": Tool(
        "send_chart_to_ui",
        send_chart_to_ui_schema(),
        _send_chart_run,
    ),
}

TOOLING_SPEC: List[Dict[str, Any]] = [tool.schema for tool in TOOL_REGISTRY.values()]
