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

MAX_TOOL_OUTPUT_CHARS = 32_768  # 32KB limit for tool output

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


def _read_proc_status_kb() -> Dict[str, int]:
    stats: Dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSize:", "VmData:", "VmStk:", "VmExe:", "VmLib:", "VmSwap:")):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        stats[parts[0].rstrip(":")] = int(parts[1])
    except Exception:
        pass
    return stats


def _read_mem_available_kb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except Exception:
        pass
    return None


def _log_mem(stage: str) -> None:
    stats = _read_proc_status_kb()
    if not stats:
        return
    parts = [f"{k}={v}kB" for k, v in stats.items()]
    print(f"[mem] {stage}: " + " ".join(parts))


def _log_df_mem(stage: str, df: pd.DataFrame) -> None:
    try:
        bytes_used = int(df.memory_usage(deep=True).sum())
        cols = len(df.columns)
        rows = len(df)
        print(f"[mem] {stage}: df_rows={rows} df_cols={cols} df_bytes={bytes_used}")
    except Exception:
        pass


def _configure_pyarrow_threads() -> None:
    try:
        import pyarrow as pa

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except Exception:
        pass


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
    """
    Serialize tool output to JSON, with smart truncation to maintain validity.
    For dataframe results, reduces head size instead of breaking JSON.
    """
    try:
        raw = json.dumps(result, default=_default_serializer)
    except Exception as exc:
        raw = json.dumps({"kind": "error", "traceback": f"Serialization failed: {exc}"})

    if len(raw) <= MAX_TOOL_OUTPUT_CHARS:
        return raw

    # Try smart truncation for dataframe results
    if isinstance(result, dict) and result.get("kind") == "dataframe" and "head" in result:
        # Progressively reduce head size to fit within limit
        for head_size in [10, 5, 3, 1]:
            truncated = result.copy()
            head_data = result.get("head", [])
            if len(head_data) > head_size:
                truncated["head"] = head_data[:head_size]
                truncated["truncated"] = True
                truncated["truncated_from"] = len(head_data)
            try:
                raw = json.dumps(truncated, default=_default_serializer)
                if len(raw) <= MAX_TOOL_OUTPUT_CHARS:
                    return raw
            except Exception:
                pass
        # If still too large, remove head entirely
        truncated = result.copy()
        truncated["head"] = []
        truncated["truncated"] = True
        truncated["truncated_reason"] = "data too large to serialize"
        try:
            raw = json.dumps(truncated, default=_default_serializer)
            if len(raw) <= MAX_TOOL_OUTPUT_CHARS:
                return raw
        except Exception:
            pass

    # Fallback: create a valid JSON with truncation notice
    # This ensures the model always gets parseable JSON
    try:
        truncated_result = {
            "kind": result.get("kind", "unknown") if isinstance(result, dict) else "truncated",
            "truncated": True,
            "truncated_reason": f"Output exceeded {MAX_TOOL_OUTPUT_CHARS} chars",
            "row_count": result.get("row_count") if isinstance(result, dict) else None,
            "columns": result.get("columns") if isinstance(result, dict) else None,
        }
        # Remove None values
        truncated_result = {k: v for k, v in truncated_result.items() if v is not None}
        return json.dumps(truncated_result, default=_default_serializer)
    except Exception:
        return json.dumps({"kind": "error", "truncated": True, "traceback": "Output too large to serialize"})


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


def list_datasets_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_datasets",
        "description": "List logical dataset names registered in this session (parquet-backed).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }


def get_dataset_info_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "get_dataset_info",
        "description": "Get dataset schema info (row count, columns, dtypes) from the session registry without loading parquet.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical dataset name"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def load_dataset_full_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_dataset_full",
        "description": "Load a registered dataset into pandas DataFrame in the REPL (parquet-backed). Use only if you need the full table.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical dataset name"},
                "df_name": {"type": "string", "description": "Variable name in REPL"},
            },
            "required": ["name", "df_name"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def load_dataset_sample_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_dataset_sample",
        "description": "Load only the first N rows of a registered dataset into a pandas DataFrame for quick inspection.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical dataset name"},
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


def load_dataset_columns_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "load_dataset_columns",
        "description": "Load specific columns from a registered dataset into a pandas DataFrame (all rows).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical dataset name"},
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


def find_value_columns_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "find_value_columns",
        "description": "Scan a dataset to find which columns contain the given value. Returns match counts per column.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Logical dataset name"},
                "value": {"type": "string", "description": "Value to search for (exact match)"},
            },
            "required": ["name", "value"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def send_chart_to_ui_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "send_chart_to_ui",
        "description": "Emit a chart spec for an existing DataFrame to the UI (bar/column/stacked_column/line/scatter). Use only when a chart helps the user understand the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "df_name": {"type": "string", "description": "Name of DataFrame in the REPL"},
                "logical_name": {
                    "type": "string",
                    "description": "Label to show in UI (optional, defaults to df_name)",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "column", "stacked_column", "line", "scatter"],
                    "description": "Chart type (bar/column/stacked_column/line/scatter only)",
                },
                "x_field": {"type": "string", "description": "Column for x-axis or category"},
                "x_label": {"type": "string", "description": "Optional label for the x-axis"},
                "y_label": {"type": "string", "description": "Optional label for the y-axis"},
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
                "title": {"type": "string", "description": "Optional chart title"},
                "note": {"type": "string", "description": "Optional short note/footnote"},
            },
            "required": ["df_name", "chart_type", "x_field", "series"],
            "additionalProperties": False,
        },
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


def _registry_entry(session: SessionState, name: str) -> Optional[Dict[str, Any]]:
    entry = session.csv_registry.get(name)
    if entry is None:
        return None
    if isinstance(entry, str):
        return {"csv_path": entry, "parquet_path": "", "row_count": None, "columns": None, "dtypes": None}
    return entry


def _read_parquet_table(parquet_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    import pyarrow as pa

    if not parquet_path:
        raise FileNotFoundError("parquet_path missing")
    _configure_pyarrow_threads()
    _log_mem("before_read_parquet_table")
    try:
        pf = pq.ParquetFile(parquet_path, memory_map=False)
        meta = pf.metadata
        total_rows = meta.num_rows if meta else None
        total_row_groups = meta.num_row_groups if meta else None
        total_uncompressed = 0
        if meta:
            for i in range(meta.num_row_groups):
                rg = meta.row_group(i)
                total_uncompressed += sum(rg.column(j).total_uncompressed_size for j in range(rg.num_columns))
        mem_avail = _read_mem_available_kb()
        print(
            "[parquet] stats:",
            f"rows={total_rows}",
            f"row_groups={total_row_groups}",
            f"uncompressed_bytes={total_uncompressed}",
            f"mem_available_kb={mem_avail}",
            f"columns={columns if columns else 'ALL'}",
        )
    except Exception:
        pass
    # memory_map can fail inside constrained containers; default to non-mmap for reliability
    try:
        table = pq.read_table(parquet_path, columns=columns, use_threads=False, memory_map=False)
        df = table.to_pandas(use_threads=False, self_destruct=True, split_blocks=True)
        _log_mem("after_read_parquet_table")
        _log_df_mem("after_read_parquet_table", df)
        return df
    except Exception as exc:
        msg = str(exc).lower()
        if "resource temporarily unavailable" not in msg and "pthread" not in msg and "bad_alloc" not in msg and "memoryerror" not in msg:
            raise
        # Fallback: stream in batches without threads and assemble in pandas.
        pf = pq.ParquetFile(parquet_path, memory_map=False)
        batches = []
        for batch in pf.iter_batches(columns=columns, use_threads=False):
            batches.append(batch.to_pandas(use_threads=False, self_destruct=True, split_blocks=True))
        if not batches:
            return pd.DataFrame()
        df = pd.concat(batches, ignore_index=True)
        _log_mem("after_read_parquet_table_fallback")
        _log_df_mem("after_read_parquet_table_fallback", df)
        return df


def _read_parquet_sample(parquet_path: str, columns: Optional[List[str]], nrows: int) -> pd.DataFrame:
    import pyarrow.parquet as pq
    import pyarrow as pa

    if not parquet_path:
        raise FileNotFoundError("parquet_path missing")
    _configure_pyarrow_threads()
    _log_mem("before_read_parquet_sample")
    # memory_map can fail inside constrained containers; default to non-mmap for reliability
    pf = pq.ParquetFile(parquet_path, memory_map=False)
    batches = []
    rows = 0
    for batch in pf.iter_batches(batch_size=nrows, columns=columns, use_threads=False):
        batches.append(batch)
        rows += batch.num_rows
        if rows >= nrows:
            break
    if not batches:
        return pd.DataFrame()
    table = pa.Table.from_batches(batches)
    df = table.to_pandas(use_threads=False, self_destruct=True, split_blocks=True).head(nrows)
    _log_mem("after_read_parquet_sample")
    _log_df_mem("after_read_parquet_sample", df)
    return df


def _load_dataframe(entry: Dict[str, Any], columns: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    parquet_path = entry.get("parquet_path") or ""
    if not parquet_path:
        raise FileNotFoundError("parquet_path missing")
    _log_mem("before_load_dataframe")
    if nrows is not None:
        return _read_parquet_sample(parquet_path, columns, nrows)
    return _read_parquet_table(parquet_path, columns=columns)


def load_dataset_into_session(session: SessionState, name: str, df_name: str) -> Dict[str, Any]:
    entry = _registry_entry(session, name)
    if not entry:
        return {"kind": "error", "traceback": f"Dataset '{name}' not registered."}
    try:
        df = _load_dataframe(entry)
        session.python.globals[df_name] = df
        session.python.prune_dataframes(keep_names={df_name})
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
    except MemoryError as exc:
        return {"kind": "error", "traceback": f"MemoryError loading data (sandbox ~2GB): {exc}"}
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


def send_df_to_ui_as_df(session: SessionState, df_name: str, logical_name: str = "df") -> Dict[str, Any]:
    df = session.python.globals.get(df_name)
    if df is None:
        df = session.python.consume_handoff_df(df_name)
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


def load_dataset_sample(session: SessionState, name: str, df_name: str = "df_sample", nrows: int = 200) -> Dict[str, Any]:
    entry = _registry_entry(session, name)
    if not entry:
        return {"kind": "error", "traceback": f"Dataset '{name}' not registered."}
    n = nrows if nrows and nrows > 0 else 200
    try:
        df = _load_dataframe(entry, nrows=n)
        session.python.globals[df_name] = df
        session.python.prune_dataframes(keep_names={df_name})
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


def load_dataset_columns(
    session: SessionState, name: str, columns: List[str], df_name: str = "df"
) -> Dict[str, Any]:
    entry = _registry_entry(session, name)
    if not entry:
        return {"kind": "error", "traceback": f"Dataset '{name}' not registered."}
    if not columns:
        return {"kind": "error", "traceback": "No columns provided."}
    try:
        df = _load_dataframe(entry, columns=columns)
        session.python.globals[df_name] = df
        session.python.prune_dataframes(keep_names={df_name})
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


def _get_dataset_info_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name", "")
    if not name:
        return {"kind": "error", "traceback": "Dataset name required."}
    entry = session.csv_registry.get(name)
    if not entry:
        return {"kind": "error", "traceback": f"Dataset '{name}' not found."}
    return {
        "kind": "object",
        "json": {
            "name": name,
            "row_count": entry.get("row_count"),
            "columns": entry.get("columns") or [],
            "dtypes": entry.get("dtypes") or {},
            "csv_bytes": entry.get("csv_bytes"),
            "parquet_bytes": entry.get("parquet_bytes"),
        },
    }


def _load_dataset_full_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    return load_dataset_into_session(session, args.get("name", ""), args.get("df_name", "df"))


def _send_chart_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    df_name = args.get("df_name", "")
    df = session.python.globals.get(df_name)
    # Also check handoff_df (result from python_repl)
    if df is None and session.python.handoff_df is not None:
        if df_name == session.python.handoff_df_name or df_name == "analysis_df":
            df = session.python.handoff_df
    if df is None:
        return {"kind": "error", "traceback": f"DataFrame '{df_name}' not found. Available: {list(session.python.globals.keys())}"}
    return {
        "kind": "chart",
        "df_name": df_name,
        "logical_name": args.get("logical_name") or df_name,
        "chart_spec": {
            "title": args.get("title"),
            "chart_type": args.get("chart_type"),
            "x_field": args.get("x_field"),
            "x_label": args.get("x_label"),
            "y_label": args.get("y_label"),
            "series": args.get("series") or [],
            "note": args.get("note"),
            "df_name": df_name,
        },
    }


def _parse_scalar_value(value: str, arrow_type) -> Any:
    import pyarrow as pa
    import pandas as pd

    if value is None:
        return None
    if pa.types.is_boolean(arrow_type):
        v = value.strip().lower()
        if v in ("true", "t", "1", "yes", "y"):
            return True
        if v in ("false", "f", "0", "no", "n"):
            return False
        return None
    if pa.types.is_integer(arrow_type):
        try:
            return int(value)
        except Exception:
            return None
    if pa.types.is_floating(arrow_type):
        try:
            return float(value)
        except Exception:
            return None
    if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
        try:
            return pd.to_datetime(value)
        except Exception:
            return None
    # string-like and other types: compare as string
    return str(value)


def _count_matches_for_batch(arr, arrow_type, raw_value: str) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc

    if raw_value is None:
        mask = pc.is_null(arr)
        return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)

    coerced = _parse_scalar_value(raw_value, arrow_type)
    if coerced is None:
        return 0

    try:
        is_string_like = pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_dictionary(arrow_type)
        if isinstance(coerced, str) and not is_string_like:
            # For non-string columns, skip string comparisons unless it can be coerced
            return 0
        if isinstance(coerced, str) and is_string_like:
            mask = pc.equal(arr.cast(pa.string()), coerced)
        else:
            mask = pc.equal(arr, pa.scalar(coerced, type=arrow_type))
        return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)
    except Exception:
        return 0


def _find_value_columns_run(session: SessionState, args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name", "")
    value = args.get("value", "")
    if not name:
        return {"kind": "error", "traceback": "Dataset name required."}
    entry = session.csv_registry.get(name)
    if not entry:
        return {"kind": "error", "traceback": f"Dataset '{name}' not found."}
    parquet_path = entry.get("parquet_path")
    if not parquet_path or not os.path.exists(parquet_path):
        return {"kind": "error", "traceback": "Parquet path not found."}

    try:
        import pyarrow.parquet as pq

        _configure_pyarrow_threads()
        pf = pq.ParquetFile(parquet_path, memory_map=False)
        schema = pf.schema_arrow
        columns = [field.name for field in schema]
        match_counts: Dict[str, int] = {c: 0 for c in columns}
        scanned_rows = 0

        for batch in pf.iter_batches(use_threads=False):
            scanned_rows += batch.num_rows
            for i, col_name in enumerate(batch.schema.names):
                arr = batch.column(i)
                match_counts[col_name] += _count_matches_for_batch(arr, arr.type, value)

        matches = [
            {"column": col, "match_count": int(cnt), "dtype": str(schema.field(col).type)}
            for col, cnt in match_counts.items()
            if cnt > 0
        ]
        matches.sort(key=lambda x: x["match_count"], reverse=True)
        return {
            "kind": "object",
            "json": {
                "name": name,
                "value": value,
                "scanned_rows": int(scanned_rows),
                "matches": matches,
                "columns_scanned": len(columns),
            },
        }
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}


TOOL_REGISTRY: Dict[str, Tool] = {
    "python_repl": Tool("python_repl", python_repl_schema(), _python_repl_run),
    "list_datasets": Tool("list_datasets", list_datasets_schema(), _list_csv_run),
    "get_dataset_info": Tool("get_dataset_info", get_dataset_info_schema(), _get_dataset_info_run),
    "load_dataset_full": Tool("load_dataset_full", load_dataset_full_schema(), _load_dataset_full_run),
    "load_dataset_sample": Tool("load_dataset_sample", load_dataset_sample_schema(), lambda s, a: load_dataset_sample(s, a.get("name", ""), a.get("df_name", "df_sample"), a.get("nrows", 200))),
    "load_dataset_columns": Tool(
        "load_dataset_columns",
        load_dataset_columns_schema(),
        lambda s, a: load_dataset_columns(s, a.get("name", ""), a.get("columns", []), a.get("df_name", "df")),
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
    "find_value_columns": Tool(
        "find_value_columns",
        find_value_columns_schema(),
        _find_value_columns_run,
    ),
}

TOOLING_SPEC: List[Dict[str, Any]] = [tool.schema for tool in TOOL_REGISTRY.values()]
