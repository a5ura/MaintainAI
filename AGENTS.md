# MaintainAI development instructions

These instructions apply throughout this repository. Follow the user's task scope and constraints; ask when authorization or requirements are unclear.

## Project orientation

MaintainAI is a local industrial equipment maintenance prototype built with Python, Streamlit, SQLite, and local semantic retrieval, with optional advisory cloud generation.

- `app.py`: Streamlit entry point and database connection lifecycle.
- `maintainai/database.py`: SQLite schema, connections, and database path.
- `maintainai/service.py`: Streamlit-independent validation and maintenance operations.
- `maintainai/ui.py` and feature-specific `*_ui.py` modules: navigation and presentation.
- `maintainai/csv_import.py`, `log_import.py`, and `seed.py`: validated imports and synthetic seeding.
- `maintainai/embeddings.py` and `semantic_search.py`: local embeddings, index storage, freshness, and retrieval.
- `maintainai/fault_detection.py`, `fault_matching.py`, and `guided_matching.py`: deterministic log patterns and retrieval queries.
- `maintainai/llm.py`, `rag.py`, `fault_rag.py`, and `guided_rag.py`: optional provider transport, evidence preparation, and response validation.
- `tests/`: pytest unit, integration, and Streamlit AppTest coverage.
- `sample_data/`: synthetic equipment, issue, and log CSV examples.
- `scripts/seed.py`: repeatable equipment/issue seed CLI.
- `data/`, `.cache/`, `.venv/`, `.tools/`, and `outputs/`: ignored local runtime/development artifacts, not shared project context.

Read [PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) for architecture, [DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) for checkpoints, and [README.md](README.md) for setup and user workflows. Check actual source and tests when documentation differs from implementation.

## Working approach

1. Verify the current branch, HEAD, and working tree before changing files. If the task specifies an expected branch or commit, stop on a mismatch.
2. Inspect relevant source, configuration, tests, and instructions before editing. Distinguish verified behavior, user-reported history, assumptions, and unverified capabilities.
3. Make focused, reviewable changes within the authorized scope. Preserve existing application behavior unless the task explicitly requests a behavior change. Avoid unrelated cleanup and speculative features.
4. Keep maintenance validation independent of Streamlit, use parameterized SQL, and preserve transactional imports and UTC timestamp handling.
5. Preserve user work and local data. Do not treat ignored files as disposable merely because Git does not track them.
6. Review the resulting files and diff, then run relevant checks permitted by the task. Never weaken, skip, or remove tests merely to obtain a passing result.

## Authorization boundaries

- Never stage, commit, push, create a PR, merge, delete branches, or modify Git configuration without explicit user approval for that operation. Approval for one operation does not imply approval for the others.
- Do not perform destructive operations without approval, including deleting data, resetting or discarding changes, and overwriting existing work.
- Respect task-specific restrictions on branch switching, installations, test execution, and file edits.
- Never expose `.env` contents, API keys, credentials, or private local data in terminal output, documentation, prompts, screenshots, diffs, or messages. Use `.env.example` and synthetic fixtures for examples.
- Do not make cloud AI requests without explicit authorization. The existence of credentials or a configured provider does not authorize a request. Never send private maintenance data as part of an incidental smoke test.

## AI and evidence safeguards

- Preserve explicit consent and explicit generation actions. Previewing evidence, navigating pages, and loading configuration must not grant consent or initiate generation.
- Preserve evidence freshness checks, index validation, snapshot/revision checks, invalidation of prior authorization, and rejection of responses when evidence changes during a request.
- Preserve provenance: technician reports, stored logs, deterministic temporal findings, historical ticket excerpts, and AI-generated hypotheses must remain distinguishable.
- Keep actual source IDs, equipment scope, citation validation, truncation/omission notices, uncertainty, advisory warnings, and retrieval-only fallbacks intact.
- Do not treat similarity scores as confidence or citations as proof of a diagnosis. Historical actions and temporal patterns do not establish a current cause or authorize maintenance work.
- Preserve the differences between standalone, fault, and guided consent flows. Do not claim all screens provide the guided flow's exact-message authorization.
- Do not add automatic retries, autonomous tool execution, machine control, or AI-driven record changes unless explicitly requested and reviewed with the relevant safeguards.

## Verification

Run relevant tests for code changes, scaled to the affected behavior and the user's constraints. Documentation-only work normally needs factual/link review and whitespace/status checks; do not claim a regression run unless it actually occurred.

Routine tests must use temporary databases, fake embeddings, and mocked providers/transports. Preserve the guards in `tests/conftest.py` against real model loading, developer configuration, and unmocked HTTP. Do not use the real maintenance database for tests.

From the repository root, the recommended focused test command to avoid previously observed Windows permissions problems is:

```powershell
.\.venv\Scripts\python.exe -m pytest -ra tests/test_operations.py --basetemp="$env:LOCALAPPDATA\MaintainAI-pytest-temp\focused" -p no:cacheprovider
```

This exact combination of test module, temporary directory, and options has not yet been tested. Select modules relevant to the change. See [README test instructions](README.md#run-tests-on-windows) for the full regression command. Pytest may clear its `--basetemp` directory: use it only for disposable test output and do not repoint it at existing work. If permissions block a check, report the limitation; do not broaden filesystem permissions or change Git configuration without approval.

Use `git diff --check` and `git status --short` for final review. Untracked files do not appear in ordinary `git diff` output; inspect new files directly and check their whitespace separately without staging them.

## Completion and handoff

Report the changed files and purpose, checks/tests actually run and their results, checks not run, material limitations or uncertainties, and proposed next actions. Keep historical test results separate from current validation. Stop at any review or approval boundary specified by the user.
