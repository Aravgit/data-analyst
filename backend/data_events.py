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
