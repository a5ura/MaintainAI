# MaintainAI project context

This document describes the implementation inspected at commit `33e5dc7`. It is shared technical context, not a claim that live model inference, cloud connectivity, or production readiness has been verified.

See [README.md](../README.md) for installation, launch commands, seeding, imports, environment settings, offline operation, and manual verification. See [AGENTS.md](../AGENTS.md) for development rules and [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for historical checkpoints and workflow work.

## Purpose and technologies

MaintainAI is an open-source local industrial equipment maintenance tracker and advisory troubleshooting prototype. Bundled records are synthetic portfolio examples, not manufacturer guidance. AI output requires qualified human verification using approved procedures.

The project requires Python 3.11+. `requirements.txt` declares Streamlit, pandas, NumPy, Sentence Transformers, PyTorch, requests, and pytest with bounded version ranges. SQLite and CSV parsing use Python's standard library. There is no dependency lockfile or external database requirement.

Streamlit provides the UI; pandas supports tables/charts; NumPy supports vector operations. Optional generation uses HTTP chat completions through requests, without a vendor SDK. `.streamlit/config.toml` binds the server to `127.0.0.1`, disables usage statistics, and limits uploads.

## Architecture and important modules

`app.py` opens a SQLite connection, initializes the schema, constructs `MaintenanceStore`, and calls `ui.render()`. Each Streamlit run closes its connection afterward; connections are not cached across users.

| Area | Modules and responsibility |
| --- | --- |
| Persistence | `maintainai/database.py`: schema, connections, default path. |
| Maintenance domain | `maintainai/service.py`: validation, equipment/issues/log operations, filtering, history, statistics; independent of Streamlit and pandas. |
| Main UI | `maintainai/ui.py`: navigation, Home, dashboard, equipment, issue creation/review, history, and equipment/issue CSV import. |
| Imports and fixtures | `maintainai/csv_import.py`, `log_import.py`, `seed.py`; `scripts/seed.py` exposes the seed CLI. |
| Local search | `maintainai/embeddings.py`, `semantic_search.py`, `semantic_ui.py`: model adapter, derived SQLite index, freshness, cosine search, and Similar Issues page. |
| Provider and shared RAG | `maintainai/llm.py`, `llm_ui.py`, `rag.py`, `rag_ui.py`: optional transport, safe configuration status, prompts, response validation, standalone assistant. |
| Fault workflows | `maintainai/fault_detection.py`, `fault_matching.py`, `fault_ui.py`, `fault_rag.py`, `fault_rag_ui.py`: observed temporal patterns, historical matching, and optional advisory generation. |
| Guided workflows | `maintainai/guided_matching.py`, `guided_rag.py`, `guided_ui.py`: description/log query construction, bounded evidence, session workflow, exact-message preview and authorization. |
| Verification | `tests/`: maintenance, import, search, provider, RAG, fault, guided, and Streamlit interaction tests. |

The service uses parameterized SQL. Batch imports are transactional. UI/session orchestration owns freshness and authorization checks; pure evidence builders do not independently verify database state or authorize transmission.

## SQLite data model

The default database is `data/maintainai.db`, resolved from the project source location. `MAINTAINAI_DB_PATH` can override it. Connections enable foreign keys and use a 10-second timeout; initialization enables WAL mode. Generated database files and sidecars are ignored by Git.

| Table | Contents |
| --- | --- |
| `equipment` | Unique case-insensitive asset tag, identifying/location fields, condition, creation timestamp. SQLite's built-in `NOCASE` folds ASCII letters. |
| `issues` | Equipment reference, description, priority, status, reporter, UTC timestamps, investigation/fix fields, optional unique demo seed key. |
| `equipment_logs` | Equipment reference, UTC event timestamp, severity, event code, message, source, import timestamp. |
| `issue_embeddings` | Derived issue vector BLOB, equipment reference, content hash, and dimension. |
| `semantic_index_metadata` | Single index record containing model key, source signature, dimension, issue count, and build timestamp. |

Schema initialization uses `CREATE ... IF NOT EXISTS`; it is not a general schema migration framework.

## Implemented maintenance and log functionality

- Equipment registration, listing/viewing, and condition updates. General equipment-field editing and deletion are not implemented.
- Issue creation, equipment/status/priority filtering, literal description/reporter text search, status/findings updates, equipment history, and dashboard counts/charts.
- Equipment condition is separate from issue status. Open totals include Open and In Progress; closed totals include Resolved and Closed.
- UTC timestamps; issue creation time is preserved during updates. Omitted findings are retained; explicit empty text clears them. Findings are editable current fields, not an append-only audit log.
- Strict UTF-8 equipment/issue/log CSV imports with header and row validation, 5 MiB and 10,000-row limits, and all-or-nothing insertion. Imports append records rather than updating them; repeated issue/log imports can create additional records.
- Repeatable equipment/issue demo seeding preserves edits and existing records. Stable issue seed keys depend on fixture row position, so existing rows must retain their order.
- `sample_data/` contains 5 equipment records, 20 issues, and 7 log events. The seed command loads equipment/issues only; logs require separate import. Seeding does not build embeddings.
- Stored-log browsing supports equipment and severity filters, newest event first. Log timestamps require an explicit timezone and are normalized to UTC.

## Deterministic patterns and historical evidence

Fault detection has two temporal rules: repeated ERROR/CRITICAL events with the same equipment and event code, with adjacent failures at most five minutes apart; and a WARNING strictly preceding an ERROR/CRITICAL event for the same equipment by at most two minutes. Overlapping matches sharing evidence are coalesced per rule, so an episode can span longer than an individual window.

These are deterministic observed correlations, not causal diagnoses. An absence of detected patterns does not establish equipment health. Fault historical matching uses cited log observations to query existing issues across equipment; it does not generate hypotheses or rebuild the index.

## Local embeddings and retrieval

`embeddings.py` lazily loads `sentence-transformers/all-MiniLM-L6-v2` at the revision pinned in source. Inference uses CPU and 384-dimensional embeddings. Model files are cached in `.cache/sentence-transformers`; initial loading may download files. `MAINTAINAI_MODEL_OFFLINE=1` requires cached files. Ordinary maintenance pages do not load the model.

Each populated description, troubleshooting-notes, root-cause, and corrective-action field is encoded with equipment-name context. Normalized field vectors are averaged and normalized into one issue vector. SQLite stores float32 vectors; NumPy scans them for cosine similarity. Retrieval returns up to five issues; equipment filtering, when requested, precedes ranking and limiting.

The index is explicitly rebuilt from the corpus. Source fingerprints, model identity, counts, content hashes, and vector validation detect stale or invalid indexes. Concurrent changes can abort index publication or invalidate retrieval. There is no automatic background indexing, relevance cutoff, reranker, or external vector database. Scores are similarity measures, not calibrated diagnostic confidence.

## Optional cloud generation and consent

`llm.py` supports an explicitly configured OpenAI-compatible chat-completions endpoint. There is no runtime default endpoint, model, or key. `.env.example` contains a credential-free Gemini-compatible example; provider/model availability must be verified separately.

Seven `MAINTAINAI_LLM_*` settings configure endpoint, model, key, timeout, token limit, JSON mode, and token-parameter name. The optional project-root `.env` reader is literal and does not export values; explicit process settings take precedence. Configuration status validates local settings without testing connectivity or granting consent. Never copy private configuration into shared documentation.

Transport validates endpoints, bounds requests/responses, disables redirects and inherited proxy/netrc settings, sanitizes failures, and does not automatically retry. Provider tool/function calls are unsupported.

The three UI flows have distinct authorization behavior:

- The standalone assistant defaults to retrieval-only mode. Cloud mode requires an opt-in checkbox and explicit submission.
- Fault troubleshooting previews the evidence envelope, ties consent to evidence fingerprints, and checks log/history freshness before and after generation.
- Guided troubleshooting saves a session draft, optionally analyzes stored logs on an explicit action, and retrieves history on another explicit action. Retrieval uses cached-only embeddings and scopes to selected equipment, or searches globally when none is selected. It previews exact system/user messages and requires consent plus an explicit generation click. Authorization includes full evidence snapshots, database/index identity, and revisions. Refreshes or changed evidence invalidate prior consent; detected changes during a request discard its response.

The shared generation function requires literal `use_cloud=True` and historical records before initializing a provider. Guided/fault adapters preserve their additional evidence requirements. Previewing, consenting, or navigating alone does not generate a response. Failures retain local evidence without displaying rejected generated advice. Fault/guided results live in the Streamlit session; standalone results are not cached across reruns.

## Provenance and safety

Technician descriptions are reports. Stored logs and historical tickets are recorded evidence that may be inaccurate or malicious. Temporal findings are deterministic correlations. AI-generated causes/checks/next steps are unverified hypotheses. Keep these categories separate.

`rag.py` validates structured JSON, allowed fields, actual supplied citation IDs, reference consistency, text limits, uncertainty, and selected unsafe instruction patterns. Log-aware responses also require log citations and stricter advisory language. Evidence is allowlisted and bounded; truncation and omissions are disclosed. Guided context and serialized messages are capped at 64 KiB each.

Citations establish provenance, not logical support or a verified current diagnosis. Safety filters are incomplete and are not a certified industrial safety system. The application implements no autonomous agents, MCP/tool execution, machine control, or AI-driven ticket/equipment changes.

## Verification boundaries and limitations

Routine pytest coverage uses temporary databases, deterministic fake embeddings, mock providers/transports, and Streamlit AppTest. `tests/conftest.py` isolates developer LLM settings and blocks real model loads and unmocked requests HTTP. These tests do not establish live embedding relevance, provider availability, or manual browser acceptance.

This is a local prototype without authentication, role permissions, attachments, an append-only audit trail, or production deployment hardening. Concurrent edits use last-write-wins behavior. Full index recomputation and in-memory scans target small local datasets; the general English model is not trained on this application's industrial data. Dependencies are not fully locked. Windows setup is documented; cross-platform readiness has not been established by this documentation work.

The audit did not verify current private database contents/index freshness, installed dependency versions, model-cache completeness, live cloud credentials/connectivity, or provider/model availability. The README architecture tree omits newer log/fault/guided modules; use the source and module inventory above when tracing those areas.
