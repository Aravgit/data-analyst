import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CELL_THRESHOLD = int(os.getenv("DATA_EVENT_CELL_THRESHOLD", "10000"))  # cells = rows * cols
ROW_THRESHOLD = int(os.getenv("DATA_EVENT_ROW_THRESHOLD", "5000"))
INLINE_ROW_LIMIT = int(os.getenv("DATA_EVENT_INLINE_ROWS", "200"))
EXPORT_DIR_NAME = "exports"

ALLOWED_CHART_TYPES = {"bar", "column", "stacked_column", "line", "scatter"}
MAX_CHART_ROWS = 200
MAX_CHART_SERIES = 10


def _fernet_from_key(secret: Optional[str]):
    if not secret:
        return None
    try:
        from cryptography.fernet import Fernet
    except Exception:
        return None
    key = hashlib.sha256(secret.encode("utf-8")).digest()
    fkey = base64.urlsafe_b64encode(key)
    return Fernet(fkey)


def encrypt_payload(obj: Dict[str, Any]) -> str:
    """
    Encrypt JSON payload; fall back to base64 if encryption is disabled or unavailable.
    Disable by setting DATA_EVENT_ENCRYPT=0 (default).
    """
    data = json.dumps(obj, default=str).encode("utf-8")
    if os.getenv("DATA_EVENT_ENCRYPT", "0") in ("0", "false", "False", "", None):
        return base64.urlsafe_b64encode(data).decode("utf-8")

    secret = os.getenv("DATA_EVENT_KEY")
    f = _fernet_from_key(secret)
    if f:
        try:
            return f.encrypt(data).decode("utf-8")
        except Exception:
            pass
    return base64.urlsafe_b64encode(data).decode("utf-8")


def make_data_frame_event(
    df: pd.DataFrame,
    df_name: str,
    session_id: str,
    logical_name: str,
    base_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Build data_frame event; optionally create a download and return download path.
    Returns (events, download_path_or_None).
    """
    cols = list(df.columns)
    row_count = int(len(df))
    head_rows = df.head(INLINE_ROW_LIMIT)
    dtypes = {c: str(t) for c, t in df.dtypes.items()}

    events: List[Dict[str, Any]] = []
    download_path: Optional[str] = None
    cells = row_count * max(len(cols), 1)

    payload = {
        "df_name": df_name,
        "logical_name": logical_name,
        "columns": cols,
        "rows": head_rows.to_dict(orient="records"),
        "row_count": row_count,
        "dtypes": dtypes,
    }
    events.append({"type": "data_frame", "payload": encrypt_payload(payload)})

    if base_dir is not None and (row_count > ROW_THRESHOLD or cells > CELL_THRESHOLD):
        exports_dir = base_dir / session_id / EXPORT_DIR_NAME
        exports_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{logical_name or df_name}.csv"
        out_path = exports_dir / file_name
        df.to_csv(out_path, index=False)
        download_payload = {
            "df_name": df_name,
            "logical_name": logical_name,
            "rows": row_count,
            "columns": len(cols),
            "path": str(out_path),
        }
        events.append({"type": "data_download", "payload": encrypt_payload(download_payload)})
        download_path = str(out_path)

    return events, download_path


def make_chart_event(
    df: pd.DataFrame,
    chart_spec: Dict[str, Any],
    session_id: str,
    logical_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Build chart event from a dataframe + spec.
    Returns (events, rejection_reason) where rejection_reason is None on success.
    """
    if not isinstance(chart_spec, dict):
        return [], "chart_spec must be an object"

    chart_type = str(chart_spec.get("chart_type", "")).lower()
    x_field = chart_spec.get("x_field")
    series = chart_spec.get("series") or []
    data_rows = chart_spec.get("data")

    if chart_type not in ALLOWED_CHART_TYPES:
        return [], f"unsupported chart_type '{chart_type}'"
    if not isinstance(x_field, str) or not x_field:
        return [], "x_field required"
    if not isinstance(series, list) or len(series) == 0:
        return [], "series required"
    if len(series) > MAX_CHART_SERIES:
        return [], f"too many series ({len(series)})"
    if chart_type == "scatter" and len(series) != 1:
        return [], "scatter charts require exactly one series"

    for s in series:
        if not isinstance(s, dict) or not s.get("y_field"):
            return [], "each series needs y_field"

    # choose data source
    if data_rows is not None and isinstance(data_rows, list):
        src_df = pd.DataFrame(data_rows)
    else:
        src_df = df.copy()

    if x_field not in src_df.columns:
        return [], f"x_field '{x_field}' missing"

    needed_cols = [x_field] + [s["y_field"] for s in series if isinstance(s, dict) and s.get("y_field")]
    missing = [c for c in needed_cols if c not in src_df.columns]
    if missing:
        return [], f"missing columns {missing}"

    # limit rows
    src_df = src_df.head(MAX_CHART_ROWS)

    # validation: numeric y for cartesian charts
    if chart_type in ("bar", "column", "stacked_column", "line", "scatter"):
        for s in series:
            y = s.get("y_field")
            try:
                pd.to_numeric(src_df[y])
            except Exception:
                return [], f"y_field '{y}' not numeric"

    # scatter requires numeric x as well
    if chart_type == "scatter":
        try:
            pd.to_numeric(src_df[x_field])
        except Exception:
            return [], f"x_field '{x_field}' not numeric for scatter"

    x_label = chart_spec.get("x_label")
    y_label = chart_spec.get("y_label")

    payload = {
        "title": chart_spec.get("title"),
        "chart_type": chart_type,
        "x_field": x_field,
        "x_label": x_label,
        "y_label": y_label,
        "series": [
            {
                "name": s.get("name") or s.get("y_field"),
                "y_field": s.get("y_field"),
                "color": s.get("color"),
            }
            for s in series
        ],
        "data": src_df[needed_cols].to_dict(orient="records"),
        "note": chart_spec.get("note"),
        "df_name": chart_spec.get("df_name"),
        "logical_name": logical_name,
    }
    return [{"type": "chart", "payload": encrypt_payload(payload)}], None
