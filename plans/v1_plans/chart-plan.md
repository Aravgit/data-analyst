## Chart Feature Plan

### Goals
- Let the agent surface a lightweight chart alongside summaries so users see trends faster than tables alone.
- Keep tables/downloads as fallback; reject unsafe or oversized payloads before render.

### Event & Payload Schema
- New SSE/sync type: `chart`.
- Payload (encrypted via `encrypt_payload`):  
  `{ title?: string, chart_type: "bar"|"line"|"area"|"pie", x_field: string, series: [{ name: string, y_field: string, color?: string }], data: Row[], note?: string, df_name?: string, logical_name?: string }`
- Limits: max 200 rows, max 10 series, require `x_field` present on all rows, numeric `y_field` for bar/line/area, pie only if unique categories ≤ 8; trim long strings.
- Sync `/chat` response includes `charts: EncPayload[]` array parallel to `data_frames`/`downloads`; decrypt client-side.

### Backend Work
- Add `make_chart_event(df, spec, session_id)` helper that builds payload, validates rows/fields, encrypts, and returns a `chart` event list plus rejection reason if dropped.
- Emit hook: after aggregations/plots in tools (e.g., `load_csv_sample`, `python_repl` results), attempt `make_chart_event`; if rejected, log reason in activity stream; still emit `data_frame`/`data_download`.
- Stream dispatcher (`/chat/stream` and SSE) forwards `chart` events before the final `reply`; sync path attaches `charts` array (encrypted) alongside `data_frames`/`downloads`.
- Reuse existing encryption/env flags; keep row caps aligned with `DATA_EVENT_*` thresholds; respect export thresholds (large data → skip chart, send download note).

### Agent Prompting
- Update system prompt/tool docs: when data is small enough, propose one chart by supplying `chart` event fields (x_field, series, chart_type, optional title/note); prefer bar/line, allow pie only if categories ≤ 8 and values sum to meaningful whole; cap rows at 200. Keep textual reply to summary/aggregates.
- Stop tool calls once chart + summary ready; do not duplicate data in markdown tables if chart emitted.

### Frontend
- State: store ordered list of charts (most recent first) keyed by `title || logical_name || df_name`; keep selection index to prevent flicker.
- SSE handler: add `chart` event branch → decrypt → validate → push to list; on rejection, show activity entry and inline message in Charts tab.
- Sync `/chat` path: decrypt `charts` array and merge into chart list on initial load or refresh.
- Tabs: add `Charts` tab next to Summary/Data. When active, show selector (if >1), chart canvas, badges for rows/fields/source, and “View as table” link that switches to Data tab.
- Component: new `components/ui/chart-card.tsx` using Recharts primitives (`ResponsiveContainer`, `CartesianGrid`, `XAxis`, `YAxis`, `Tooltip`, `Legend`, `LineChart`, `BarChart`, `AreaChart`, `PieChart`). Load via `dynamic(() => import(...), { ssr: false })` to avoid Next SSR issues.
- Dependency: add `recharts` (and types if needed) to `frontend/package.json`; import only required primitives to keep bundle small.

### Validation & Safety
- Reject payloads missing required fields or exceeding limits; log to activity panel with reason.
- Clamp numeric parsing, coerce finite numbers, and sanitize colors to a safe palette fallback.
- Never crash Summary/Data tabs if chart parse fails.

### UX Copy
- If rejected: “Chart skipped (reason). View the data in the Data tab.”
- If download emitted instead: show note “Dataset large; chart not rendered. Download available.”

### Testing
- Unit: chart validator with malformed payloads; helper to coerce numeric fields.
- Integration (frontend): mock SSE chart event → chart renders; invalid event → rejection path.
- Manual: upload sample CSV, ask “bar chart of sales by region”; verify chart tab, summary, and table coexist.

### Open Questions
- Allowed chart set beyond bar/line/area? (pie default off unless categories < 8)
- Palette/branding preferences? Use Tailwind slate/emerald fallback until decided.
- Should charts be generated only from tool-produced aggregates (safer) vs. model-specified raw rows (faster)?
