## Plan: Always emit a table per turn

Goal: Every answered turn delivers one `data_frame` (and optional `data_download`) event for the final result. Text summaries stay small; Data tab shows the table.

### Backend steps
1) **String-only payload contract**
   - Ensure all `data_frame` / `data_download` events carry `payload` as a base64-encoded JSON string (never an object).

2) **Wrap all python_repl tabular outputs**
   - When `python_repl` returns a DataFrame (or Series/list/dict convertible to rows), return `kind: "dataframe"` with `df_name`, `head`, `row_count`, `columns`, `dtypes`, plus a short `text`.
   - For non-tabular scalars, coerce to a 1-row, 2-column frame: `key`, `value`.

3) **Final-table emission**
   - Track the last dataframe result in a turn; after the tool loop, emit exactly one `data_frame` (and download if large) before the final reply.
   - Add a simple tool `emit_table(df_name, logical_name)` so the model can package an existing DataFrame explicitly if needed.

4) **Prompt update**
   - Instruct: “Every answer must produce one data_frame event for the final result (or a 1-row table if scalar). Summaries reference it; do not inline >20 rows.”

5) **Front-end safety**
   - Decode only when `typeof payload.payload === "string"`; ignore otherwise. Paginate tables (already in place).

### Testing
- Turn with python_repl aggregation → data_frame emitted, Data tab updates, summary small.
- Scalar result → 1-row table emitted.
- Large result (>threshold) → download link + head rows; summary says too large to inline.
- Multiple tool calls → only the last dataframe is sent.
