"""Explicit guided generation through the real shared pipeline and fake providers."""

from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest

from maintainai.database import connect
from maintainai.llm import LLMProvider, ProviderError, provider_from_environment as real_provider_factory
from maintainai.rag import CLAIM_FIELDS, generate_grounded_analysis as shared_generation
from maintainai.service import MaintenanceStore
from test_guided_ai_ui import (
    AI_KEY, ANALYSIS_REVISION, CONSENT, GENERATE, RETRIEVAL_REVISION, authorize, ready,
)
from test_guided_matching_ui import (
    ANALYSIS_KEY, DRAFT_KEY, MATCHES_KEY, add_logs, analyze, case, find, search, submit,
)


def completion(messages):
    envelope = json.loads(messages[1]["content"])
    claim = {"text": "A similar historical condition may be relevant; qualified personnel must verify applicability.",
             "issue_ids": envelope["allowed_issue_ids"][:1]}
    if "allowed_log_ids" in envelope:
        claim["log_ids"] = envelope["allowed_log_ids"][:1]
    data = {"issue_summary": "Advisory interpretation of the technician's reported symptoms.",
            **{field: [deepcopy(claim)] for field in CLAIM_FIELDS},
            "evidence_references": claim["issue_ids"],
            "uncertainty_or_missing_information": ["Current equipment conditions have not been independently verified."]}
    if "log_ids" in claim:
        data["log_evidence_references"] = claim["log_ids"]
    return json.dumps(data)


@pytest.fixture
def cloud(case, monkeypatch):
    provider = Mock(spec=LLMProvider)
    provider.generate.side_effect = completion
    factory = Mock(return_value=provider)
    monkeypatch.setattr("maintainai.rag.provider_from_environment", factory)
    backend = Mock(wraps=shared_generation)
    monkeypatch.setattr("maintainai.guided_ui.generate_grounded_analysis", backend)
    case.provider, case.provider_factory, case.backend = provider, factory, backend
    return case


def generate(cloud):
    find(cloud.app.button, GENERATE).click().run()
    assert not cloud.app.exception


def snapshot(cloud):
    with connect(cloud.path) as connection:
        return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
                for table in ("equipment", "issues", "equipment_logs", "issue_embeddings", "semantic_index_metadata")}


def assert_discarded(cloud, *, provider_calls):
    assert cloud.app.session_state[AI_KEY]["response"] is None
    assert cloud.app.session_state[AI_KEY]["consent"] is False
    assert not find(cloud.app.checkbox, CONSENT).value
    assert find(cloud.app.button, GENERATE).disabled
    assert any("response discarded" in item.value for item in cloud.app.warning)
    assert cloud.provider.generate.call_count == provider_calls
    cloud.app.run()
    assert not cloud.app.exception
    assert cloud.provider.generate.call_count == provider_calls


def test_explicit_generation_sends_exact_preview_and_uses_shared_log_validation_without_writes(cloud, monkeypatch):
    from maintainai import guided_ui

    ready(cloud)
    assert find(cloud.app.button, GENERATE).disabled
    cloud.provider_factory.assert_not_called()
    authorize(cloud)
    assert not find(cloud.app.button, GENERATE).disabled
    cloud.provider_factory.assert_not_called()
    preview = deepcopy(cloud.app.session_state[AI_KEY]["messages"])
    serialized = cloud.app.session_state[AI_KEY]["serialized_messages"]
    before = snapshot(cloud)
    model_calls = len(cloud.embedder.calls)
    builder = Mock(wraps=guided_ui.build_guided_messages)
    monkeypatch.setattr(guided_ui, "build_guided_messages", builder)
    generate(cloud)
    assert builder.call_count == 1  # Current preview only; no rebuilding after authorization.
    cloud.provider_factory.assert_called_once_with()
    cloud.provider.generate.assert_called_once()
    assert cloud.provider.generate.call_args.args[0] == preview
    assert cloud.backend.call_args.args[0] == preview
    assert json.dumps(preview, ensure_ascii=True) == serialized
    assert cloud.backend.call_args.kwargs["use_cloud"] is True
    assert cloud.backend.call_args.kwargs["logs"]
    result = cloud.app.session_state[AI_KEY]["response"]
    assert result["ai_generated"]
    assert all(result["analysis"][field][0]["issue_ids"] and result["analysis"][field][0]["log_ids"] for field in CLAIM_FIELDS)
    assert {"Possible causes", "Suggested diagnostic checks", "Uncertainty or missing information"} <= {
        item.value for item in cloud.app.subheader}
    assert any("Supporting evidence: Issue #" in item.value and "Log #" in item.value for item in cloud.app.caption)
    assert any("hypotheses — unverified" in item.value for item in cloud.app.caption)
    assert snapshot(cloud) == before
    assert cloud.log_reader.call_count == 3  # Explicit analysis, then pre/post request checks.
    assert cloud.detector.call_count == cloud.search.call_count == 1
    assert len(cloud.embedder.calls) == model_calls


@pytest.mark.parametrize("mode", ["no_equipment", "equipment_no_analysis", "empty_analysis", "no_patterns"])
def test_optional_equipment_and_logs_select_the_actual_citation_mode(cloud, mode):
    if mode == "no_patterns":
        add_logs(cloud, patterns=False)
        submit(cloud, cloud.pump)
        analyze(cloud)
        search(cloud)
    else:
        ready(cloud, logs=False, equipment=mode != "no_equipment")
        if mode == "empty_analysis":
            analyze(cloud)
            search(cloud)
    authorize(cloud)
    generate(cloud)
    result = cloud.app.session_state[AI_KEY]["response"]
    assert result["ai_generated"]
    log_aware = mode == "no_patterns"
    assert (cloud.backend.call_args.kwargs["logs"] is not None) is log_aware
    assert ("log_evidence_references" in result["analysis"]) is log_aware
    if mode in ("no_equipment", "equipment_no_analysis"):
        cloud.log_reader.assert_not_called()
    if mode == "no_patterns":
        assert cloud.app.session_state[AI_KEY]["context"]["observed_evidence"]["detected_patterns"] == []


def test_saved_response_survives_reruns_unsubmitted_edits_navigation_and_consent_changes_without_new_requests(cloud):
    ready(cloud)
    authorize(cloud)
    generate(cloud)
    saved = deepcopy(cloud.app.session_state[AI_KEY]["response"])
    calls = cloud.log_reader.call_count, cloud.detector.call_count, cloud.search.call_count, len(cloud.embedder.calls)
    find(cloud.app.text_area, "Problem description *").input("Unsubmitted new report")
    cloud.app.run()
    find(cloud.app.checkbox, CONSENT).uncheck().run()
    assert not cloud.app.exception
    find(cloud.app.sidebar.radio, "Navigate").set_value("Home").run()
    find(cloud.app.button, "Troubleshoot a Problem").click().run()
    assert not cloud.app.exception
    assert cloud.app.session_state[AI_KEY]["response"] == saved
    assert calls == (cloud.log_reader.call_count, cloud.detector.call_count, cloud.search.call_count, len(cloud.embedder.calls))
    cloud.provider_factory.assert_called_once()
    cloud.provider.generate.assert_called_once()


@pytest.mark.parametrize("failure", ["missing_credentials", "timeout", "rate_limit", "unavailable", "authentication", "quota"])
def test_provider_failures_keep_local_evidence_and_never_retry_automatically(cloud, failure):
    ready(cloud)
    if failure == "missing_credentials":
        cloud.provider_factory.side_effect = real_provider_factory
    else:
        cloud.provider.generate.side_effect = ProviderError(failure, "Synthetic sanitized provider failure")
    authorize(cloud)
    generate(cloud)
    result = deepcopy(cloud.app.session_state[AI_KEY]["response"])
    assert not result["ai_generated"]
    assert all(result["analysis"][field] == [] for field in CLAIM_FIELDS)
    assert result["analysis"]["evidence_references"]
    assert any("Local fallback — no AI-generated hypotheses" == item.value for item in cloud.app.caption)
    assert any("retrieval-only" in item.value for item in cloud.app.warning)
    cloud.app.run()
    assert not cloud.app.exception
    assert cloud.app.session_state[AI_KEY]["response"] == result
    cloud.provider_factory.assert_called_once()
    assert cloud.provider.generate.call_count == (0 if failure == "missing_credentials" else 1)


@pytest.mark.parametrize("failure", ["json", "issue_citation", "log_citation", "missing_log_citation", "unsafe", "confirmed_cause"])
def test_invalid_provider_output_is_discarded_by_existing_shared_validation(cloud, failure):
    ready(cloud)

    def invalid(messages):
        if failure == "json":
            return "PRIVATE MALFORMED OUTPUT"
        result = json.loads(completion(messages))
        if failure == "issue_citation":
            result["possible_causes"][0]["issue_ids"] = [999999]
        elif failure == "log_citation":
            result["possible_causes"][0]["log_ids"] = [999999]
        elif failure == "missing_log_citation":
            result["possible_causes"][0]["log_ids"] = []
        elif failure == "unsafe":
            result["suggested_next_steps"][0]["text"] = "Disable the safety interlock"
        else:
            result["issue_summary"] = "The current root cause is a failed seal"
        return json.dumps(result)

    cloud.provider.generate.side_effect = invalid
    authorize(cloud)
    generate(cloud)
    assert not cloud.app.session_state[AI_KEY]["response"]["ai_generated"]
    assert any("discarded" in item.value for item in cloud.app.warning)
    expected_diagnostic = {
        "json": "json.invalid [response]",
        "issue_citation": "citation.issue_ids [possible_causes.issue_ids]",
        "log_citation": "citation.log_ids [possible_causes.log_ids]",
        "missing_log_citation": "citation.log_ids [possible_causes.log_ids]",
        "unsafe": "safety.operational_instruction [suggested_next_steps.text]",
        "confirmed_cause": "text.cause_language [issue_summary]",
    }[failure]
    assert any(expected_diagnostic in item.value for item in cloud.app.warning)
    visible = "\n".join(item.value for item in cloud.app.text)
    assert "999999" not in visible and "PRIVATE MALFORMED OUTPUT" not in visible
    assert "Disable the safety interlock" not in visible
    cloud.app.run()
    assert not cloud.app.exception
    cloud.provider.generate.assert_called_once()


@pytest.mark.parametrize("failure", ["citation", "unsafe"])
def test_issue_only_mode_retains_its_existing_citation_and_safety_checks(cloud, failure):
    ready(cloud, logs=False, equipment=False)

    def invalid(messages):
        result = json.loads(completion(messages))
        if failure == "citation":
            result["possible_causes"][0]["issue_ids"] = [999999]
        else:
            result["possible_causes"][0]["text"] = "MoveJ to the target"
        return json.dumps(result)

    cloud.provider.generate.side_effect = invalid
    authorize(cloud)
    generate(cloud)
    assert cloud.backend.call_args.kwargs["logs"] is None
    assert not cloud.app.session_state[AI_KEY]["response"]["ai_generated"]


def test_no_history_keeps_generation_disabled_and_shows_only_local_fallback(cloud, monkeypatch):
    from maintainai import guided_ui

    submit(cloud, cloud.spare)
    search(cloud)
    assert find(cloud.app.checkbox, CONSENT).disabled
    assert find(cloud.app.button, GENERATE).disabled
    assert any("Local fallback — no AI-generated hypotheses" == item.value for item in cloud.app.caption)
    original = guided_ui.st.button

    def forged(label, *args, **kwargs):
        value = original(label, *args, **kwargs)
        return True if label == GENERATE else value

    monkeypatch.setattr(guided_ui.st, "button", forged)
    cloud.app.run()
    assert not cloud.app.exception
    cloud.provider_factory.assert_not_called()
    cloud.backend.assert_not_called()


@pytest.mark.parametrize("change", ["new_log", "edited_log"])
def test_live_log_changes_before_generation_abort_before_provider_initialization(cloud, change):
    ready(cloud)
    authorize(cloud)
    with connect(cloud.path) as connection:
        if change == "new_log":
            MaintenanceStore(connection).create_log(equipment_id=cloud.pump, timestamp="2026-10-25T12:01:00Z",
                                                    severity="INFO", message="New event after preview")
        else:
            with connection:
                connection.execute("UPDATE equipment_logs SET message=? WHERE equipment_id=?", ("Edited event", cloud.pump))
    generate(cloud)
    assert_discarded(cloud, provider_calls=0)
    cloud.provider_factory.assert_not_called()
    cloud.backend.assert_not_called()


def change_input(cloud, change, monkeypatch, tmp_path):
    if change == "draft":
        cloud.app.session_state[DRAFT_KEY] = {"description": "Changed submitted problem", "equipment_id": cloud.pump}
    elif change == "matches":
        cloud.app.session_state[MATCHES_KEY]["matches"][0]["similarity"] = .123
    elif change == "revision":
        cloud.app.session_state[RETRIEVAL_REVISION] += 1
    elif change == "analysis_revision":
        cloud.app.session_state[ANALYSIS_REVISION] += 1
    elif change == "messages":
        cloud.app.session_state[AI_KEY]["messages"][0]["content"] += "Changed instructions"
    elif change == "context":
        cloud.app.session_state[AI_KEY]["context"]["question"] = "Changed prepared evidence"
    elif change == "consent":
        cloud.app.session_state[AI_KEY]["consent"] = False
    elif change == "equipment":
        with connect(cloud.path) as connection:
            MaintenanceStore(connection).update_equipment_status(cloud.pump, "Maintenance")
    elif change == "deleted_equipment":
        with connect(cloud.path) as connection:
            with connection:
                connection.execute("DELETE FROM equipment_logs WHERE equipment_id=?", (cloud.pump,))
                connection.execute("DELETE FROM issues WHERE equipment_id=?", (cloud.pump,))
                connection.execute("DELETE FROM equipment WHERE id=?", (cloud.pump,))
    elif change == "index":
        with connect(cloud.path) as connection:
            with connection:
                connection.execute("UPDATE issue_embeddings SET vector=? WHERE issue_id=?",
                                   (np.ones(cloud.embedder.dimension, dtype="<f4").tobytes(), cloud.issues[0]))
    elif change == "database":
        other = tmp_path / "changed_workspace.db"
        with connect(cloud.path) as source, connect(other) as target:
            source.backup(target)
        monkeypatch.setenv("MAINTAINAI_DB_PATH", str(other))
    elif change == "history":
        with connect(cloud.path) as connection:
            MaintenanceStore(connection).update_issue(cloud.issues[0], status="Resolved", root_cause="Changed historical evidence")
        cloud.rebuild()
    elif change == "logs":
        with connect(cloud.path) as connection:
            MaintenanceStore(connection).create_log(equipment_id=cloud.pump, timestamp="2026-10-25T12:01:00Z",
                                                    severity="INFO", message="New live observation")


@pytest.mark.parametrize("change", ["draft", "equipment", "deleted_equipment", "database", "matches", "index", "revision", "analysis_revision", "messages", "context", "consent"])
def test_changes_between_preview_and_click_are_rejected_before_any_provider_call(cloud, monkeypatch, tmp_path, change):
    from maintainai import guided_ui

    ready(cloud)
    authorize(cloud)
    original = guided_ui.st.button

    def before_check(label, *args, **kwargs):
        value = original(label, *args, **kwargs)
        if label == GENERATE and value:
            change_input(cloud, change, monkeypatch, tmp_path)
        return value

    monkeypatch.setattr(guided_ui.st, "button", before_check)
    generate(cloud)
    assert_discarded(cloud, provider_calls=0)
    cloud.provider_factory.assert_not_called()
    cloud.backend.assert_not_called()


@pytest.mark.parametrize("change", ["logs", "history", "draft", "equipment", "deleted_equipment", "database", "matches", "index", "revision", "analysis_revision", "messages", "context", "consent", "sent_messages"])
def test_changes_during_provider_request_discard_response_revoke_consent_and_never_retry(cloud, monkeypatch, tmp_path, change):
    ready(cloud)
    authorize(cloud)

    def changed(messages):
        result = completion(messages)
        if change == "sent_messages":
            messages[0]["content"] += "Mutated provider argument"
        else:
            change_input(cloud, change, monkeypatch, tmp_path)
        return result

    cloud.provider.generate.side_effect = changed
    generate(cloud)
    assert_discarded(cloud, provider_calls=1)
    cloud.provider_factory.assert_called_once()
    assert not any(item.value == "Advisory interpretation of the technician's reported symptoms." for item in cloud.app.text)
