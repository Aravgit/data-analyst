# Subplan: Multi-CSV Upload Support (Only)

Date: 2026-02-16
Status: Draft for discussion

## Goal

Enable one session to hold and query multiple CSV datasets at once, while keeping the current agent/tooling model intact.

This subplan intentionally excludes:
- XLSX/XLS/XLSB ingestion
- Layout parsing
- New semantic retrieval/indexing architecture

## Why this first

- The backend tools already support dataset names (`name` argument), so multi-dataset reasoning is mostly blocked by upload/session behavior.
- This gives immediate product value with low architectural risk.
- It sets up the foundation for later XLSX and semantic routing work.

## Current blockers in code

- Upload clears existing registry every time.
- Upload seeds `session.messages` in a way that assumes one active dataset.
- Frontend tracks one file label (`csvName`) and does not show dataset inventory.

## Target behavior

1. A session can contain N CSV datasets.
2. User can upload one or many CSV files into the same session.
3. Existing datasets are preserved by default.
4. Name collisions are resolved deterministically.
5. Agent can list/select datasets naturally using existing tools.
6. User can optionally replace all datasets in a session.

## API design

### Option A (recommended first): Keep `/upload`, add upload mode

`POST /upload` adds fields:
- `mode`: `append` | `replace_session` (default `append`)
- `dataset_name` (optional override for logical name)

Behavior:
- `append`: keep existing registry, add new dataset.
- `replace_session`: clear registry + messages + REPL, then add uploaded dataset(s).

Response:
- `datasets`: full list of registered dataset names after upload.
- `added_dataset`: logical name created/updated.

### Option B (phase 2 UX): Add `/upload/batch`

`POST /upload/batch` with multiple files in one request.

Behavior:
- Uses same `mode`.
- Returns:
  - `added`: names added
  - `skipped`: files skipped
  - `errors`: per-file errors
  - `datasets`: final dataset list

Recommendation: ship Option A first, then Option B.

## Naming and collision policy

Rules:
- Base logical name from filename stem unless `dataset_name` is provided.
- If collision on `append`, add suffix: `_2`, `_3`, ...
- Keep a stable mapping `{logical_name -> parquet metadata}`.
- Return final resolved logical name in response.

This avoids silent overwrite and keeps user mental model clear.

## Backend implementation details

### 1) Session model

No schema rewrite needed immediately. Keep `csv_registry` shape, but allow many entries.

Add helper methods:
- `register_dataset(session, logical_name, meta, collision_mode)`
- `list_dataset_names(session)`

### 2) Upload flow (`backend/main.py`)

Changes:
- Stop unconditional `session.csv_registry.clear()` for default flow.
- Add mode-driven clear/replace behavior.
- Preserve existing datasets when `append`.
- Keep per-session storage under `DATA_ROOT/{session_id}/`.

For replacements:
- If replacing a specific dataset, delete old parquet file if present.
- Keep cleanup defensive to avoid deleting unrelated session files.

### 3) Agent context seeding

Current seeding overwrites message history on upload.

Update to:
- Build a compact dataset inventory string and inject it into the system prompt per turn.
  - Example: `"Datasets: sales_2024(cols: date, region, revenue), ops_daily(cols: date, unit, generation_mw)"`
- Do not rely on assistant chat messages for inventory context (can be compacted away).
- Do not wipe chat history in `append` mode.
- Only reset history in `replace_session`.

### 4) Tooling impact

Existing tools already support multi-dataset through `name` argument:
- `list_datasets`
- `get_dataset_info`
- `load_dataset_full`
- `load_dataset_sample`
- `load_dataset_columns`
- `find_value_columns`

Required prompt tweak:
- Instruct model to call `list_datasets` when multiple datasets may exist.
- Instruct model to confirm target dataset when user query is ambiguous.

### 5) Session cleanup and quotas

Add guardrails:
- Per-file upload size remains 400MB.
- Add per-session cumulative cap: default `2GB` (`SESSION_DATA_BYTES_LIMIT=2147483648`).
- On cap exceed: return `413` with structured detail (`{"code":"session_limit_exceeded", ...}`).
- Enforce TTL cleanup already present in session store.

## Frontend changes

### 1) Dataset inventory panel

Show all datasets in current session:
- Fetch from upload response immediately.
- Add required `GET /session/{id}/datasets` endpoint for page refresh/hydration.

### 2) Upload UX

Phase 1:
- Keep single file picker, allow repeated uploads into same session.
- Add mode selector: `Append` or `Replace all`.

Phase 2:
- Enable `<input multiple>` and upload queue.

### 3) Chat affordance

Add a small hint in chat input area:
- `"Active datasets: sales_2024, ops_daily, metal_cost"`

This reduces ambiguity before semantic routing is added.

### 4) URL and routing update

- Replace `/chat?session={id}&csv={name}` with `/chat?session={id}`.
- `csv` query param is single-file oriented and should be removed for multi-file mode.

## Migration and compatibility

- Keep existing endpoint and request format backward compatible.
- Default mode is `append`; old behavior remains available via `mode=replace_session`.

## Testing plan

### Backend tests

1. Append mode:
- Upload A then B in same session.
- Assert registry has 2 datasets.

2. Collision mode:
- Upload same filename twice.
- Assert suffix naming and both datasets accessible.

3. Replace session:
- Upload A + B, then upload C with `replace_session`.
- Assert only C remains.

4. Tool compatibility:
- `list_datasets` returns all names.
- `get_dataset_info` works for each dataset.

### Frontend tests

1. Repeated uploads update visible dataset list.
2. Replace mode clears list and shows only new dataset(s).
3. Chat continues in same session after append uploads.

## Rollout sequence

Phase 1:
- Backend append mode + collision handling + tests.
- Minimal frontend dataset list.

Phase 2:
- Batch upload endpoint + multiple file picker.
- Better upload progress and partial-failure handling.

## Risks and mitigations

Risk:
- User asks question without specifying dataset.

Mitigation:
- Prompt policy: list datasets and ask one short clarification when ambiguous.

Risk:
- Session storage growth from many uploads.

Mitigation:
- Per-session byte cap + clear replacement modes + existing TTL eviction.

Risk:
- Hidden behavior change for current users.

Mitigation:
- Keep explicit upload mode selector in UI and preserve old behavior via `replace_session`.

## Acceptance criteria

1. Same session can retain and query at least 10 CSV files.
2. No existing tool contracts break.
3. Upload modes behave deterministically.
4. Agent can accurately choose dataset when user names it.
5. Ambiguous queries trigger clarification instead of wrong dataset execution.
