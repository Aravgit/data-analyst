## Data Delivery & Summary Plan

Goal: deliver raw tabular data to the UI (encrypted), while the agent returns only a concise summary/aggregates. Tables should appear immediately; summaries follow when ready.

### Backend changes
1) **Structured tool outputs**
   - Extend tool return shapes (`load_csv_sample`, `load_csv_columns`, `load_csv_handle`, `python_repl`) to include `kind: "dataframe"` with `head`, `row_count`, `columns`, `dtypes`, `df_name`.
   - Preserve current serialized string for the model, but also emit a structured payload to the UI channel.

2) **Encryption helper**
   - Add a utility `encrypt_payload(obj) -> str` using a symmetric key from env (e.g., `DATA_EVENT_KEY`); fallback to no encryption if unset for dev.
   - Emit encrypted JSON blobs; UI will decrypt.

3) **New event types**
   - In sync `/chat`: include `data_frames` and `downloads` arrays (encrypted) alongside `reply`.
   - In streaming `/chat/stream` and dispatcher SSE: emit `data_frame` and `data_download` events before the model’s final reply.

4) **Export for large datasets**
   - When `row_count * len(columns) > CELL_THRESHOLD` (e.g., 10k) or `row_count > ROW_THRESHOLD` (e.g., 5000), write CSV to `DATA_ROOT/<session_id>/exports/<name>.csv`.
   - Emit `data_download` event with encrypted metadata + relative URL.
   - Inline `data_frame.rows` capped at 200 rows.

5) **Session lifecycle**
   - On `/reset`, delete the session’s `exports` directory.
   - Respect existing 400 MB cap for exports; emit user-friendly error if exceeded.

6) **System prompt updates**
   - Instruct the model: “Tables are emitted via data_frame/data_download events; keep textual reply to summary/aggregates. Include small Markdown tables only if under the inline threshold.”
   - Remind to stop calling tools after obtaining needed results.

### Frontend changes
1) **Event handling**
   - Chat client listens for `data_frame` → decrypt → render a paginated table component immediately.
   - Listen for `data_download` → show Download button with URL.
   - Keep text reply area for summary/aggregates only.

2) **Decryption**
   - Add shared decrypt helper using same key (from env).

3) **UI behavior**
   - Show “Data received” panel with rows/columns count; paginate/virtualize for speed.
   - When summary arrives, render below the table; avoid re-rendering the table.

### Testing checklist
- Small result (<10k cells): inline table shown; summary includes brief Markdown table.
- Large result: download event emitted; no inline bulk data; summary is brief.
- Missing column / bad args: error surfaced without breaking event flow.
- Multiple sessions: exports and events stay session-scoped.
