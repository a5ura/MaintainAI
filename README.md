# MaintainAI

An open-source industrial equipment maintenance tracker with **v0.2B advisory RAG troubleshooting**, built on v0.2A semantic search and the v0.1 maintenance baseline. It runs with Python 3.11+, Streamlit, SQLite, pandas, NumPy, and Sentence Transformers. Local search, retrieval-only troubleshooting, and automated tests require no paid subscription, LLM API key, GPU, or external database. Optional cloud generation uses your chosen chat-completions endpoint and may incur provider charges. A public embedding model is downloaded on first use; embedding inference runs locally on CPU.

**Synthetic demo data:** all bundled assets, equipment models, locations, incidents, people, findings, and corrective actions are invented for this portfolio project. ABB and Zaber are example manufacturer names; this project is not affiliated with either company. Demo records are not manufacturer maintenance guidance.

## Features

- Equipment registry with asset tags, categories, manufacturers, models, locations, and condition. Includes an ABB robotic arm, vacuum gripper, Zaber gantry, conveyor, and machine vision station.
- Issue creation with equipment, description, priority, status, reporter, and automatic UTC timestamps.
- Issue filtering by equipment, priority, status, and literal text search across description and reporter.
- Updates to issue status, troubleshooting notes, root cause, and corrective action.
- Equipment maintenance history and dashboard counts, status charts, and category statistics.
- SQLite persistence, repeatable synthetic seeding, and validated, atomic equipment/issue CSV imports.
- **Similar Issues:** describe a maintenance problem and retrieve up to five related issues with cosine scores, equipment names, findings, corrective actions, and original issue references.
- Explicit index rebuild/refresh with stale-index detection after ticket creation, edits, or CSV imports.
- Twenty synthetic example tickets across five equipment types, including fifteen additional robotics maintenance cases in v0.2A.
- **AI Troubleshooting Assistant:** optional equipment-scoped retrieval, structured advisory hypotheses, server-validated source ticket IDs, and a no-key retrieval-only default.
- Configurable cloud chat-completions provider with environment-only credentials, bounded requests, and retrieval-only fallback on provider or validation errors.
- pytest coverage of core operations, persistence, import failures, semantic retrieval, index freshness, and Streamlit user flows without model downloads.

## Run on Windows PowerShell

Prerequisite: install Python 3.11 or newer with the Windows Python launcher. The commands below select Python 3.11 explicitly; use your installed version (for example `-3.12`) if needed. Replace the example path with your local checkout folder.

```powershell
Set-Location 'C:\path\to\MaintainAI'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.seed
.\.venv\Scripts\python.exe -m pytest -q --basetemp=./test_temp -p no:cacheprovider
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501** after the server starts. Stop it with **Ctrl+C**. The commands call the environment's Python directly, so PowerShell script activation and execution-policy changes are unnecessary. The included Streamlit configuration binds the server to this computer's loopback interface and disables Streamlit usage statistics.

The explicit PyTorch installation selects the [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/). The application also sets `device="cpu"` for model loading and encoding. Installation needs internet access and downloads larger packages than v0.1, but no CUDA toolkit or GPU is required. Running the app and using the original maintenance pages does not load or download the embedding model.

If `py -3.11` is unavailable and this working copy contains the optional project-local Python provisioned during development, you can use that interpreter instead:

```powershell
.\.tools\python\cpython-3.11-windows-x86_64-none\python.exe -m venv .venv
```

Then run the install, seed, test, and launch commands above. The `.tools` directory is a local development convenience and is ignored by Git. A fresh clone does not include this interpreter; install Python 3.11+ as described in the prerequisite.

## Run tests on Windows

From the project folder, after installing the requirements:

```powershell
.\.venv\Scripts\python.exe -m pytest -ra --basetemp=./test_temp -p no:cacheprovider
```

`--basetemp` keeps pytest fixture files in the ignored project-local `test_temp` directory. pytest clears this directory when needed, so reserve it for disposable test output. `-p no:cacheprovider` disables pytest's cache plugin. Run from your normal Windows account if the directory is inaccessible to a sandbox account; do not grant broad filesystem permissions. Tests use separate databases and do not modify `data/maintainai.db`. Semantic tests use deterministic fake embeddings or mocked model constructors; a fixture blocks real model loading. LLM tests use deterministic mock providers and fake HTTP transports, clear LLM environment configuration, and block unmocked HTTP. No cloud subscription, model download, real API key, or paid request is needed for routine tests. Streamlit interaction tests use AppTest; they do not constitute a manual browser review.

## Using the app

1. Load synthetic data with the seed command or the sidebar's **Load synthetic demo data** button. You can also start with an empty database.
2. Browse or register assets under **Equipment**. Equipment condition is maintained separately from issue status.
3. Use **Create issue** to report an issue, then **Issues** to filter, review, and record the investigation and fix.
4. Use **Equipment history** for all related issues, including troubleshooting notes and corrective actions.
5. Use **Dashboard** for overall totals. **Open issues** includes `Open` and `In Progress`; **Closed issues** includes `Resolved` and `Closed`. A fresh v0.2A seed has 5 assets and 20 issues: 3 open and 17 closed.
6. Use **Similar Issues** to rebuild the local search index and search by meaning rather than exact description text.
7. Use **AI Troubleshooting Assistant** for retrieval-only evidence or explicitly enabled cloud-assisted hypotheses. Read the advisory warning and review source tickets before drawing conclusions.

All timestamps are stored and displayed in UTC. `created_at` stays fixed when an issue is edited; `updated_at` changes on each save. Status changes are unrestricted, allowing an issue to be reopened. Notes are current editable fields, not an append-only audit log. At the service level, a status-only update preserves existing findings; explicitly passing empty text clears a field.

Equipment CRUD remains **partial**: registration, listing, viewing, and condition updates are implemented. Editing other equipment fields and deleting equipment are not implemented in this version.

## AI Troubleshooting Assistant (v0.2B)

**Advisory only. Recommendations require qualified human verification before any safety-critical maintenance or operation. Follow approved site and manufacturer procedures.** The assistant cannot change equipment states, close tickets, run shell commands, or execute physical robot commands. It has no agents, MCP integration, tool calling, or autonomous actions.

The assistant reuses the same local MiniLM encoder, SQLite index, and cosine search. It retrieves up to five tickets globally, or up to five within an optionally selected equipment ID. Filtering happens before ranking/limiting, so a selected asset is not padded with unrelated assets. The index must be current. Missing, new, or edited tickets require a refresh on **Similar Issues**. If records change during a cloud request, the response is discarded even if someone rebuilt the index meanwhile.

The page separates **retrieved historical facts**, quoted directly from actual tickets, from **AI-generated hypotheses**, which remain unverified. Responses have issue summary, possible causes, recommended diagnostic checks, suggested next steps, evidence references, and uncertainty/missing information. Retrieval-only mode displays the question and evidence without generating causes or operational recommendations. An empty result makes no provider request.

### Retrieval-only operation: no LLM key

Install the requirements, load demo data, and rebuild the semantic index. Open **AI Troubleshooting Assistant**, keep **Retrieval only (no cloud request)** selected, optionally choose equipment, and submit a problem description. This uses local embeddings only. First-time model downloads still need internet access; cached embedding operation can be forced offline as documented below. Do not confuse retrieval-only results with an AI-generated diagnosis.

### Optional cloud configuration in Windows PowerShell

Use an endpoint/account you are authorized to use. Set variables in the same PowerShell session that launches Streamlit. There is **no default cloud endpoint, model, or API key**. Choose a provider offering an OpenAI-style [chat-completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) with bearer authentication. The application uses `requests`; no vendor SDK is required.

```powershell
Set-Location 'C:\path\to\MaintainAI'
$env:MAINTAINAI_LLM_ENDPOINT = 'https://your-provider.example/v1/chat/completions'
$env:MAINTAINAI_LLM_MODEL = 'your-provider-model-id'
$env:MAINTAINAI_LLM_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Provider API key' -AsSecureString)).Password
$env:MAINTAINAI_LLM_TIMEOUT_SECONDS = '30'
$env:MAINTAINAI_LLM_MAX_TOKENS = '1000'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Replace the example endpoint/model with values from your provider. The hidden key prompt avoids putting the key literal in PowerShell history. Keys exist in the launched process environment and HTTP authorization header; they are never included in prompts or intentionally logged/persisted by this application. After stopping Streamlit, clear the key with `Remove-Item Env:MAINTAINAI_LLM_API_KEY -ErrorAction SilentlyContinue`. `.env` files are ignored by Git but are **not automatically loaded**; no dotenv dependency is added. Never paste credentials into tickets, questions, screenshots, source files, or commits.

| Variable | Behavior |
| --- | --- |
| `MAINTAINAI_LLM_ENDPOINT` | Full `/chat/completions` URL. HTTPS is required for remote hosts; HTTP is allowed only for loopback. No URL credentials, query parameters, or fragments. |
| `MAINTAINAI_LLM_MODEL` | Required provider-specific model ID; no hardcoded default. |
| `MAINTAINAI_LLM_API_KEY` | Required bearer credential for cloud mode. A local compatible server may accept a nonsecret placeholder if it does not authenticate. |
| `MAINTAINAI_LLM_TIMEOUT_SECONDS` | Read timeout, 1–120 seconds; default 30. Connect timeout is at most 5 seconds. This is a socket timeout, not a strict end-to-end deadline. |
| `MAINTAINAI_LLM_MAX_TOKENS` | Output-token request limit, 128–4096; default 1000. Not a monetary spending cap. |
| `MAINTAINAI_LLM_JSON_MODE` | `1` (default) sends `response_format: {"type":"json_object"}`. Set `0` if your endpoint does not support that option; local JSON/schema validation still applies. |
| `MAINTAINAI_LLM_TOKEN_PARAMETER` | `max_tokens` (default), or `max_completion_tokens` for compatible endpoints/models that require it. |

The thin provider supports a single nonstreaming text completion. Endpoints requiring different authentication, URL query parameters, tools, multimodal output, or vendor-specific fields need a future adapter. Redirects are rejected; TLS verification stays enabled. Environment proxies and netrc credentials are not inherited. There are no automatic retries or background requests. An intentional new submission is a new request and may be billable.

Cloud mode requires a visible opt-in to send the question, selected equipment metadata, and up to five ticket excerpts. These may contain sensitive information. Reporter names are not separately sent, but free-text tickets may still include personal or proprietary data. Questions are limited to 4,000 characters; each of the four retrieved text fields is limited to 1,200 characters per record in the prompt, with truncation marked. Review provider pricing, quotas, data retention, and data-use rules before enabling this mode. Free tiers may have limits; a free cloud service is not guaranteed. A ChatGPT subscription is not required for local tests or retrieval-only mode and does not configure an inference endpoint.

### Grounding, safeguards, and failure handling

- System instructions are separate from a JSON data envelope containing the question and retrieved records. Tickets and questions are explicitly untrusted data; embedded commands, role labels, or credential requests are not application instructions.
- Model output must contain exactly the documented structured fields. Every generated cause/check/next step needs nonempty integer `issue_ids` drawn from the retrieved set; evidence references must match the cited IDs. Unknown IDs, inline prose citations, duplicate JSON fields, malformed/oversized output, unsupported tool responses, and selected operational-command patterns are rejected.
- Historical facts are rendered from the server's actual retrieved records, not reconstructed by the model. Generated prose is rendered as plain text and labelled unverified. No output is executed or persisted as a ticket update.
- Missing/invalid configuration, timeout, authentication errors, rate limits, quota errors, unavailable providers, and invalid responses produce a sanitized message and retrieval-only evidence. Raw provider errors/bodies and rejected advice are not displayed. Empty search results skip generation. Stale evidence is discarded rather than shown as current.
- Prompt isolation and deterministic validation reduce risk; they cannot guarantee that every model ignores every injection or that a cited hypothesis is logically supported. Valid ticket IDs establish provenance, not factual entailment. Human verification is mandatory. Historical records themselves may be inaccurate or malicious. Operational-pattern checks are conservative and incomplete, not a certified industrial safety system.

### Manual troubleshooting verification

1. Start Streamlit, load the synthetic demo data, and rebuild the index on **Similar Issues**. Repeating the demo loader must not create duplicates.
2. Open **AI Troubleshooting Assistant**. Confirm the advisory warning, keep retrieval-only mode, and ask `Vacuum gripper loses suction while transferring a part`. Confirm up to five quoted tickets and all six report sections, with no AI-generated recommendations.
3. Select the Zaber gantry and ask about a homing limit signal. Confirm only that equipment's tickets appear (four records on a fresh demo seed).
4. With no cloud variables configured, choose cloud mode, explicitly allow sending excerpts, and submit. Confirm the missing-configuration message and retrieval-only fallback, with no crash.
5. To test a real cloud endpoint, configure it as above, restart Streamlit, choose cloud mode, opt in, and submit once. This can incur provider charges. Inspect hypotheses, uncertainty, source IDs, and the original tickets. Do not run any recommendations automatically.
6. Edit a ticket's findings in **Issues**, revisit the assistant, and confirm submission is disabled until the semantic index is rebuilt. An empty database should show a helpful no-history message without loading an LLM.

Automated tests cover mock generation, provider errors, prompt-injection text handling, response structure/citations, empty results, equipment filtering, nonmutation, and evidence changes during a response. Live cloud behavior is provider-dependent and is not exercised by routine tests.

## Semantic search (v0.2A)

The model is [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), an Apache-2.0 licensed English sentence encoder. It produces 384-dimensional embeddings. This implementation pins model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, uses safetensors weights, disables remote model code, and requires no Hugging Face account or token.

Each issue's description, troubleshooting notes, root cause, and corrective action are encoded separately when nonempty, including the equipment name for context. Their normalized vectors are averaged and normalized into one vector per issue. This ensures that a long description cannot displace every other field from the model input. Long individual fields and queries are still subject to the model's [256 word-piece input limit](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#intended-uses).

SQLite stores little-endian float32 vectors with the original issue ID, equipment ID, dimension, and content hash in `issue_embeddings`. `semantic_index_metadata` records the pinned model/pooling identity, ticket snapshot fingerprint, issue count, and rebuild timestamp. These are additive tables: the existing registry and issue schema/data are preserved. The embedding adapter, SQLite index storage, and retrieval service have separate boundaries for a future vector database. Search normalizes vectors and uses NumPy dot products to compute cosine similarity.

**Index freshness:** choose **Rebuild / refresh search index** after loading examples, creating/importing tickets, or editing tickets. All issue statuses are included. Each search checks the current ticket snapshot, stored row identities/content, and model identity. Missing, changed, or invalid embeddings block search until a rebuild. Changes during embedding generation abort publication of the new index; checks after query inference detect edits during model loading. Rebuilds atomically replace derived search data only and never rewrite tickets or equipment. Result lists are not cached across Streamlit reruns. As with any interactive app, changes made after a response is displayed become visible on the next interaction or page refresh.

### First model download and offline use

The first rebuild downloads model files from Hugging Face into `.cache/sentence-transformers/` inside the project (ignored by Git). Expect network transfer of roughly 90 MB of model weights plus small configuration/tokenizer files and a delay for initial loading. The initialized model is reused within the server process. Ticket text is passed to local CPU inference, not an external inference API. Network access is used to retrieve/check model files; Hugging Face telemetry is disabled by default by the adapter.

On Windows, Hugging Face may warn that symlinks are unavailable and store copies instead; this still works and can use more disk space. Administrator privileges are not required to run the application. An unauthenticated-download warning also does not mean a paid account or token is required.

After caching the model, force local-file-only loading before starting Streamlit:

```powershell
$env:MAINTAINAI_MODEL_OFFLINE = '1'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Remove this setting with `Remove-Item Env:MAINTAINAI_MODEL_OFFLINE` if you need to download the model again. Offline mode with an incomplete cache produces a helpful model-loading error. A missing dependency, failed download, or failed inference leaves existing maintenance data and the previous index intact.

### Manual verification

1. Start Streamlit, then select **Load synthetic demo data** in the sidebar. Loading it again should add no duplicate records. On a fresh database, the dashboard should show **5 equipment / 20 issues / 3 open / 17 closed**. An existing user database may have additional tickets or edited statuses.
2. Open **Similar Issues**, then choose **Rebuild / refresh search index**. Allow the first model download to finish. The index should report current with 20 indexed issues on a fresh seed.
3. Enter `Vacuum gripper loses suction and drops parts during transfer` and select **Find similar issues**. Inspect up to five ranked matches, their cosine scores, equipment names, descriptions, notes, root causes, corrective actions, and original issue IDs.
4. Try `Gantry cannot complete homing because the limit switch stays active`, `Conveyor motor overload with a jammed roller`, and `Camera images are blurry and out of focus`. Related equipment examples should rank near the top; exact rankings/scores are not acceptance guarantees for every environment.
5. In **Issues**, select a returned issue number and edit its root cause or corrective action. Reopen **Similar Issues**: a stale-index message should appear, search should be disabled, and the earlier result list should be absent. Rebuild, then search again and confirm the new text is displayed.
6. Create a new issue or import valid issue CSV rows. Confirm search again requires a refresh. Rebuild twice and verify dashboard totals and ticket IDs do not change.

### Limitations

- Cosine scores range from -1 to 1 and are similarity measures, not calibrated confidence, diagnostic certainty, or validated maintenance advice. The top five can include unrelated tickets when the dataset has no good match; there is no relevance cutoff or reranker.
- The general English model is not trained on this application's industrial data. Acronyms, uncommon fault codes, long fields, and subtle equipment differences can reduce relevance. Field averaging can dilute individual findings.
- A full refresh recomputes the corpus; vectors are scanned in memory with NumPy. This suits a local portfolio dataset, not a large shared production deployment.
- Model and dependency downloads require network access initially. The first inference can take time on CPU. Package ranges are bounded but not fully locked.
- Ticket edits use the existing last-write-wins behavior. Search freshness is a snapshot guarantee, not live background synchronization or automatic indexing.

## Data and seeding

By default the app creates `data/maintainai.db`, relative to the project folder, on first use. Foreign keys and database constraints enforce relationships and allowed values. Connections enable foreign keys, use a 10-second busy timeout, and the database uses SQLite WAL mode. Generated databases and sidecar files are ignored by Git.

Seed again safely at any time:

```powershell
.\.venv\Scripts\python.exe -m scripts.seed
```

The seed adds missing equipment by asset tag and missing demo issues by stable seed key. The first five CSV rows and their existing keys are unchanged from v0.1; fifteen new rows are appended. Seeding an already seeded v0.1 database adds only those missing examples. It preserves existing equipment, user-created issues, and edits to demo issues; it never resets the database. Seed keys derive from fixture row position, so keep existing example rows in their original order when extending the file. Asset tags use SQLite's `NOCASE` rules: ASCII letters are case-insensitive, while non-ASCII characters are compared exactly. CSV imports use the same matching rules. Both files are seeded in one transaction. Seeding does not download a model or build the search index.

To use another database, set the same environment variable for both the CLI and app:

```powershell
$env:MAINTAINAI_DB_PATH = Join-Path $PWD 'data\another_demo.db'
.\.venv\Scripts\python.exe -m scripts.seed
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also seed a specific path with `python -m scripts.seed --db data/another_demo.db`. The `--db` option affects that command only. Unset the override with `Remove-Item Env:MAINTAINAI_DB_PATH` when finished.

## CSV import

Use **CSV import** in the app. Download an example there, or use the files in `sample_data/`. These are synthetic examples with exact headers. Import `equipment.csv` before `issues.csv` into an empty database. If you already seeded, the sample equipment tags already exist and reimporting them will produce duplicate-tag errors.

| Type | Required columns | Optional columns |
| --- | --- | --- |
| Equipment | `asset_tag`, `name`, `category` | `manufacturer`, `model`, `location`, `operational_status`, `created_at` |
| Issues | `equipment_asset_tag`, `description`, `priority`, `status`, `reporter` | `created_at`, `troubleshooting_notes`, `root_cause`, `corrective_action` |

- Save as UTF-8 (a UTF-8 BOM is accepted). Headers are exact and case-sensitive; unknown, missing, or duplicate columns are rejected.
- Priorities: `Low`, `Medium`, `High`, `Critical`.
- Issue statuses: `Open`, `In Progress`, `Resolved`, `Closed`.
- Equipment conditions: `Operational`, `Maintenance`, `Offline`; omitted/blank condition defaults to `Operational`.
- Issue asset tags must refer to registered equipment. Leading/trailing whitespace is stripped. Blank required cells are rejected.
- Optional `created_at` must be ISO 8601 with an explicit timezone, e.g. `2026-01-15T09:00:00Z` or `2026-01-15T11:00:00+02:00`. Omitted/blank timestamps use the current UTC time; supplied timestamps are normalized to UTC and must remain within years 1 through 9999 after conversion.
- Quote values containing commas, quotes, or newlines using standard CSV rules.
- Maximum file size: 5 MiB; maximum records: 10,000. Validation reports row-specific errors (up to 25 at once); row numbers refer to logical CSV records, including the header. Malformed quoting reports the physical line.
- **All-or-nothing import:** validation or database failure adds no records from that file. Equipment tags must be unique. Imports add records and do not update existing rows. **Importing the same issue CSV again creates additional issues**; only the demo seeding command is idempotent.

## Architecture

```text
MaintainAI/
├── app.py                    # Streamlit entry point and connection lifecycle
├── maintainai/
│   ├── database.py            # Schema, SQLite connections, configurable DB path
│   ├── service.py             # Validation, registry, issues, filters, statistics
│   ├── csv_import.py          # CSV parsing, row errors, atomic batch import
│   ├── seed.py                # Repeatable synthetic demo loading
│   ├── embeddings.py          # Lazy CPU MiniLM adapter and embedder protocol
│   ├── semantic_search.py     # SQLite vector index, freshness, cosine retrieval
│   ├── semantic_ui.py         # Similar Issues page
│   ├── llm.py                 # Optional HTTP provider interface, config, test double
│   ├── rag.py                 # Grounded prompts, citation validation, advisory service
│   ├── rag_ui.py              # AI Troubleshooting Assistant page
│   └── ui.py                  # Streamlit pages and pandas tables/charts
├── scripts/seed.py            # Seed CLI
├── sample_data/               # Synthetic equipment and issue CSVs
├── tests/                    # Core, CSV, and Streamlit interaction tests
├── data/                     # Generated SQLite files (ignored)
├── requirements.txt
└── pyproject.toml            # Python requirement and pytest configuration
```

The UI calls a maintenance service that validates inputs and uses parameterized SQL. The service is independent of Streamlit and pandas. CSV parsing uses Python's standard library; pandas powers the displayed tables and charts. Each Streamlit run opens and closes its own database connection; no connection is cached across users. Batch imports use transactions. Tests use temporary databases and never modify the demo database. Dependencies use bounded compatible version ranges; this baseline does not include a lockfile.

## Repository contents

Commit the application source, tests, synthetic CSV examples, dependency/configuration files, documentation, and MIT license. `.streamlit/config.toml` is shared application configuration and belongs in Git.

The `.gitignore` excludes virtual environments, downloaded local tools, SQLite files and sidecars anywhere in the project, temporary test directories, caches, logs, local environment files, and common credential files. `.streamlit/secrets.toml` is excluded separately from the shared configuration. The database is generated by the application or seed command; it is not needed in a clone. Ignore rules do not remove files already tracked by Git or detect secrets embedded in source files, so inspect the staged file list and diff before committing.

## Scope

This is a local portfolio prototype without authentication, role permissions, attachments, a general migration framework, an audit trail, or production deployment hardening. Concurrent edits use last-write-wins behavior. Do not expose a shared deployment to untrusted users without adding access control. v0.2A provides local embeddings/retrieval; v0.2B adds optional, explicitly enabled cloud RAG generation. Retrieval-only mode does not send tickets to an inference provider. No agents, MCP, autonomous tools, or machine-control integration are implemented. Model-file checks/downloads can be disabled after caching; optional cloud mode requires network access to the configured endpoint.

## License

MIT. See [LICENSE](LICENSE).
