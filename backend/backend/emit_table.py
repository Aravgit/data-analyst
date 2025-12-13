from typing import Any, Dict, List

from session import SessionState
from tools import safe_tool_output
from data_events import make_data_frame_event


def emit_table(session: SessionState, df_name: str, logical_name: str = "df") -> Dict[str, Any]:
    df = session.python.globals.get(df_name)
    if df is None:
        return {"kind": "error", "traceback": f"DataFrame '{df_name}' not found."}
    try:
        events, _ = make_data_frame_event(df, df_name, session.session_id, logical_name, None)
        if not events:
            return {"kind": "error", "traceback": "No event generated."}
        return {"kind": "dataframe", "text": f"Emitted table {logical_name} from {df_name}.", "event": events[0]}
    except Exception as exc:
        return {"kind": "error", "traceback": repr(exc)}
