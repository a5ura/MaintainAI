# MaintainAI

An open-source industrial equipment maintenance tracker and the first baseline for a future AI intelligence platform. This version runs locally with Python 3.11+, Streamlit, SQLite, and pandas. It requires no API keys, paid services, GPU, or external database.

**Synthetic demo data:** all bundled assets, equipment models, locations, incidents, people, findings, and corrective actions are invented for this portfolio project. ABB and Zaber are example manufacturer names; this project is not affiliated with either company. Demo records are not manufacturer maintenance guidance.

## Features

- Equipment registry with asset tags, categories, manufacturers, models, locations, and condition. Includes an ABB robotic arm, vacuum gripper, Zaber gantry, conveyor, and machine vision station.
- Issue creation with equipment, description, priority, status, reporter, and automatic UTC timestamps.
- Issue filtering by equipment, priority, status, and literal text search across description and reporter.
- Updates to issue status, troubleshooting notes, root cause, and corrective action.
- Equipment maintenance history and dashboard counts, status charts, and category statistics.
- SQLite persistence, repeatable synthetic seeding, and validated, atomic equipment/issue CSV imports.
- pytest coverage of core operations, persistence, import failures, and Streamlit user flows.

## Run on Windows PowerShell

Prerequisite: install Python 3.11 or newer with the Windows Python launcher. The commands below select Python 3.11 explicitly; use your installed version (for example `-3.12`) if needed. Replace the example path with your local checkout folder.

```powershell
Set-Location 'C:\path\to\MaintainAI'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.seed
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .test-tmp\pytest
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501** after the server starts. Stop it with **Ctrl+C**. The commands call the environment's Python directly, so PowerShell script activation and execution-policy changes are unnecessary. The included Streamlit configuration binds the server to this computer's loopback interface and disables Streamlit usage statistics.

If `py -3.11` is unavailable and this working copy contains the optional project-local Python provisioned during development, you can use that interpreter instead:

```powershell
.\.tools\python\cpython-3.11-windows-x86_64-none\python.exe -m venv .venv
```

Then run the install, seed, test, and launch commands above. The `.tools` directory is a local development convenience and is ignored by Git. A fresh clone does not include this interpreter; install Python 3.11+ as described in the prerequisite.

## Run tests on Windows

From the project folder, after installing the requirements:

```powershell
.\.venv\Scripts\python.exe -m pytest -ra -p no:cacheprovider --basetemp .test-tmp\pytest
```

`--basetemp` keeps pytest fixture files in the ignored project-local `.test-tmp` directory. pytest clears the selected base directory when it needs it, so reserve `.test-tmp\pytest` for disposable test output. `-p no:cacheprovider` disables pytest's cache plugin. Tests use separate databases and do not modify `data/maintainai.db`. Streamlit interaction tests use AppTest; they do not constitute a manual browser review.

## Using the app

1. Load synthetic data with the seed command or the sidebar's **Load synthetic demo data** button. You can also start with an empty database.
2. Browse or register assets under **Equipment**. Equipment condition is maintained separately from issue status.
3. Use **Create issue** to report an issue, then **Issues** to filter, review, and record the investigation and fix.
4. Use **Equipment history** for all related issues, including troubleshooting notes and corrective actions.
5. Use **Dashboard** for overall totals. **Open issues** includes `Open` and `In Progress`; **Closed issues** includes `Resolved` and `Closed`. A fresh seed has 5 assets and 5 issues: 3 open and 2 closed.

All timestamps are stored and displayed in UTC. `created_at` stays fixed when an issue is edited; `updated_at` changes on each save. Status changes are unrestricted, allowing an issue to be reopened. Notes are current editable fields, not an append-only audit log. At the service level, a status-only update preserves existing findings; explicitly passing empty text clears a field.

Equipment CRUD is **partial**: registration, listing, viewing, and condition updates are implemented. Editing other equipment fields and deleting equipment are not implemented in v0.1.

## Data and seeding

By default the app creates `data/maintainai.db`, relative to the project folder, on first use. Foreign keys and database constraints enforce relationships and allowed values. Connections enable foreign keys, use a 10-second busy timeout, and the database uses SQLite WAL mode. Generated databases and sidecar files are ignored by Git.

Seed again safely at any time:

```powershell
.\.venv\Scripts\python.exe -m scripts.seed
```

The seed adds missing equipment by asset tag and missing demo issues by stable seed key. It preserves existing equipment, user-created issues, and edits to demo issues; it never resets the database. Asset tags use SQLite's `NOCASE` rules: ASCII letters are case-insensitive, while non-ASCII characters are compared exactly. CSV imports use the same matching rules. Both files are seeded in one transaction.

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

This is a local portfolio prototype without authentication, role permissions, attachments, migrations, an audit trail, or production deployment hardening. Concurrent edits use last-write-wins behavior. Do not expose a shared deployment to untrusted users without adding access control. No LLM, RAG, agent, MCP, telemetry integration, or external service call is implemented in the application. Those features can be added incrementally after this baseline.

## License

MIT. See [LICENSE](LICENSE).
