# Roadblock Comments: Multi-File + Semantic Retrieval + Messy XLSX

Date: 2026-02-16

This note responds directly to the four blockers and proposes implementable approaches that fit this codebase.

## 1) Only CSV, one file at a time

Current bottleneck is not just file type, it is the data model:
- Session currently treats upload as a single active dataset and clears previous registry.
- Agent tools assume one logical table per workflow.

What to change:

1. Move from `single dataset` to `artifact graph` per session.
- A session should contain many artifacts.
- Artifact types:
  - `table` (rectangular data, queryable)
  - `kv_block` (label-value report fragments)
  - `matrix_block` (cross-tab / KPI boards)
  - `text_block` (notes, disclaimers)
- Each artifact keeps provenance:
  - `file_id`, `sheet_name`, `cell_range`, `parser_version`, `confidence`.

2. Upload should append by default, not replace.
- Add explicit actions:
  - `append` (default)
  - `replace_session` (clear all)
  - `replace_file` (remove only one file’s artifacts)

3. Add cross-file query planning.
- Query router should decide:
  - single artifact answer
  - union comparable artifacts
  - join artifacts by inferred keys/date dimensions.

Result: user can ask one question over many uploads without re-upload churn.

## 2) No semantic similarity for categorical routing (and no external backend DB)

You can solve this at runtime without an external service by building a **session-local hybrid index**.

### Proposed runtime index (no external DB dependency)

1. Build index at upload parse time.
- Store in-memory + optional local disk cache under `DATA_ROOT/{session_id}/index/`.
- Destroy on session expiry.

2. Index units:
- `field node`: one per column/metric/kv label.
- `value node`: sampled or sketch representation of categorical values.
- `artifact node`: summary of each table/block.

3. Hybrid retrieval score:
- Lexical score:
  - normalized token match
  - char n-gram fuzzy match (handles misspellings and abbreviations)
  - unit match bonus (`rs/kwh`, `mt`, `%`, etc.)
- Semantic score:
  - embedding similarity on field descriptions and label text.
- Data evidence score:
  - value-hit score from approximate categorical lookup.
- Final score = weighted blend + confidence threshold.

4. Categorical lookup that scales:
- For low cardinality columns:
  - keep full normalized dictionary.
- For high cardinality:
  - keep sketches:
    - top-k frequent values
    - minhash signatures
    - bloom filter or trigram inverted index.
- This allows fast candidate column discovery from user terms without scanning full tables every turn.

5. Ambiguity handling:
- If top-2 scores are close, the agent should ask a targeted disambiguation question.
- If confidence is high, auto-route without asking.

### Why this is better than only embeddings

Embeddings alone fail on:
- unit-heavy industrial metrics,
- short coded labels (`FHP-1`, `LME-S`, `P&B`),
- domain abbreviations.

Hybrid retrieval with lexical + semantic + value evidence is much more reliable for your spreadsheet/report domain.

## 3) XLSX support is missing

Treat XLSX ingestion as a **layout understanding problem**, not file reading.

Do not model workbook as one table. Model it as many candidate regions with different semantics.

## 3.1 Non-tabular report-style XLSX (core hard problem)

Your sample images are not classic tables. They are dashboard-like sheets with merged cells, section headers, and mixed granularities.

### Proposed parser: Layout-to-Facts pipeline

Stage A: Sheet decomposition
- Build a cell graph per sheet:
  - nodes: non-empty cells
  - edges: adjacency, merged relations, style similarity, border continuity.
- Segment into rectangular regions using:
  - blank gaps,
  - border boxes,
  - style boundaries,
  - merged-cell header spans.

Stage B: Region classification
- Classify each region as:
  - table candidate,
  - kv block,
  - matrix KPI block,
  - decorative/text.
- Use cheap heuristics first; model fallback only on uncertain regions.

Stage C: Canonicalization
- `table`:
  - detect multi-row headers
  - flatten hierarchical headers (`COST > MTD > value` style)
  - produce tidy columns with units and dimensions.
- `kv_block`:
  - extract `(metric_label, value, unit, date_context)`.
- `matrix_block`:
  - unpivot into long format:
    - dimensions from row headers + column headers
    - measure value cell.

Stage D: Quality scoring
- Compute extraction confidence:
  - header confidence,
  - numeric consistency,
  - sparsity sanity,
  - merged-cell propagation validity.
- Keep low-confidence regions but tag them for cautious agent usage.

Stage E: Artifact registration
- Register all extracted artifacts in the same session registry and index them for retrieval.

### Key idea: Structural fingerprints

Many business reports recur daily/weekly with same layout.

Create a `sheet_fingerprint`:
- merged-cell pattern,
- border grid signature,
- anchor labels.

If fingerprint matches a known template:
- apply stored extraction map directly,
- skip expensive inference,
- get stable, deterministic parsing.

This is a major accuracy and speed win for plant/ops reports.

## Recommended architecture changes in this repo

1. Replace `csv_registry` with `artifact_registry`.
- Backward compatibility layer can expose current dataset tools.

2. Add ingestion service modules:
- `ingest/file_router.py` (csv/xlsx/xls/xlsb dispatch)
- `ingest/xlsx_layout_parser.py` (region segmentation + classification)
- `ingest/canonicalize.py` (table/kv/matrix normalization)
- `ingest/fingerprint.py` (template matching)

3. Add session-local index module:
- `backend/semantic_index.py`
- APIs:
  - `index_artifact(artifact)`
  - `search_fields(query, top_k)`
  - `search_values(query, top_k)`
  - `resolve_query_targets(query)`.

4. Add agent tools for routing:
- `list_artifacts()`
- `get_artifact_info(artifact_id)`
- `search_semantic_targets(query)`
- `load_artifact(artifact_id, projection?, limit?)`.

5. Keep runtime-only by default.
- No external vector DB required.
- Optional local on-disk cache within session dir for speed.

## Practical rollout plan (risk-controlled)

Phase 1 (fast impact):
- multi-file append uploads.
- runtime hybrid index over current CSV/parquet columns + sampled values.
- semantic target routing tool.

Phase 2:
- basic XLSX extraction:
  - detect rectangular tables + kv pairs.
- artifacts registered with provenance and confidence.

Phase 3:
- advanced report parser:
  - matrix unpivot,
  - multi-row header flattening,
  - structural fingerprint templates.

Phase 4:
- query planner across artifacts/files with join inference and confidence-aware clarifications.

## What not to do

- Do not rely on "read_excel then hope first row is header".
- Do not force all sheets into one DataFrame.
- Do not use embeddings-only routing for domain abbreviations and units.
- Do not introduce mandatory external DB infra at this stage.

## Open decisions to finalize

1. Runtime embeddings provider:
- local model (fully offline) vs API embeddings (better quality, external dependency).

2. Session persistence policy:
- ephemeral only vs optional reusable template/fingerprint store across sessions.

3. Confidence policy:
- strict (ask user often) vs aggressive auto-routing (faster but riskier).

4. UI representation:
- show extracted artifacts + confidence badges so users can pick/confirm source blocks.

