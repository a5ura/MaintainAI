"""Exact guided previews and session-only consent, without triggering generation."""

from copy import deepcopy
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect
from maintainai.embeddings import EmbeddingError
from maintainai.guided_rag import build_guided_context, build_guided_messages
from maintainai.rag import GUIDED_PROMPT
from maintainai.service import MaintenanceStore, ValidationError
from test_guided_matching_ui import (  # Reuse the real-index fixture and its cloud/network guards.
    ANALYSIS_KEY, DRAFT_KEY, MATCHES_KEY, add_logs, analyze, case, find, search, submit,
)

AI_KEY = "guided_troubleshooting_ai"
ANALYSIS_REVISION = "guided_troubleshooting_analysis_revision"
RETRIEVAL_REVISION = "guided_troubleshooting_retrieval_revision"
CONSENT = "Allow sending these exact previewed messages to the configured external AI provider"
GENERATE = "Generate AI Troubleshooting"


def ready(case, *, logs=True, equipment=True):
    if logs:
        add_logs(case)
    submit(case, case.pump if equipment else None)
    if logs:
        analyze(case)
    search(case)


def authorize(case):
    find(case.app.checkbox, CONSENT).check().run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["consent"] is True


def pending(case):
    state = case.app.session_state[AI_KEY]
    state["response"] = {"pending": "Synthetic response awaiting validation"}
    return deepcopy(state)


def assert_revoked(case, previous, *, disabled=None):
    state = case.app.session_state[AI_KEY]
    assert state["signature"] != previous["signature"]
    assert state["consent"] is False
    assert state["response"] is None
    assert not find(case.app.checkbox, CONSENT).value
    if disabled is not None:
        assert find(case.app.checkbox, CONSENT).disabled is disabled
    assert find(case.app.button, GENERATE).disabled


def test_exact_serialized_preview_matches_adapter_and_includes_both_provider_messages(case):
    ready(case)
    state = case.app.session_state[AI_KEY]
    with connect(case.path) as connection:
        equipment = MaintenanceStore(connection).get_equipment(case.pump)
    expected_context = build_guided_context(case.app.session_state[DRAFT_KEY], equipment=equipment,
                                            analysis=case.app.session_state[ANALYSIS_KEY],
                                            matches=case.app.session_state[MATCHES_KEY]["matches"])
    expected_messages = build_guided_messages(expected_context)
    assert state["context"] == expected_context
    assert state["messages"] == expected_messages
    serialized = json.dumps(expected_messages, ensure_ascii=True)
    assert state["serialized_messages"] == serialized
    assert any(item.value == serialized for item in case.app.code)
    assert json.loads(serialized) == expected_messages
    assert expected_messages[0]["role"] == "system" and expected_messages[0]["content"].startswith(GUIDED_PROMPT)
    assert expected_messages[1]["role"] == "user"
    envelope = json.loads(expected_messages[1]["content"])
    assert envelope["question"] == case.app.session_state[DRAFT_KEY]["description"]
    assert envelope["allowed_log_ids"] and envelope["allowed_issue_ids"]
    assert "Private reporter" not in serialized
    assert any("configured external AI provider" in item.value for item in case.app.info)
    assert not find(case.app.checkbox, CONSENT).value
    assert not find(case.app.checkbox, CONSENT).disabled
    assert find(case.app.button, GENERATE).disabled


def test_description_only_preview_is_available_but_consent_waits_for_explicit_history(case):
    submit(case)
    state = case.app.session_state[AI_KEY]
    assert state["messages"]
    assert state["context"]["selected_equipment"] is None
    assert state["context"]["citation_mode"] == "issue_only"
    assert find(case.app.checkbox, CONSENT).disabled
    assert not find(case.app.checkbox, CONSENT).value
    case.search.assert_not_called()
    case.factory.assert_not_called()
    case.log_reader.assert_not_called()
    case.detector.assert_not_called()
    assert not case.embedder.calls
    search(case)
    assert not find(case.app.checkbox, CONSENT).disabled
    assert case.app.session_state[AI_KEY]["context"]["citation_mode"] == "issue_only"
    authorize(case)


def test_signature_and_consent_survive_unchanged_reruns_checkbox_changes_and_navigation(case):
    ready(case)
    initial = deepcopy(case.app.session_state[AI_KEY])
    counters = (case.log_reader.call_count, case.detector.call_count, case.search.call_count, len(case.embedder.calls))
    authorize(case)
    assert case.app.session_state[AI_KEY]["signature"] == initial["signature"]
    case.app.run()
    assert not case.app.exception
    assert find(case.app.checkbox, CONSENT).value
    find(case.app.checkbox, CONSENT).uncheck().run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["consent"] is False
    assert case.app.session_state[AI_KEY]["signature"] == initial["signature"]
    authorize(case)
    find(case.app.sidebar.radio, "Navigate").set_value("Home").run()
    find(case.app.button, "Troubleshoot a Problem").click().run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["signature"] == initial["signature"]
    assert case.app.session_state[AI_KEY]["messages"] == initial["messages"]
    assert find(case.app.checkbox, CONSENT).value
    assert counters == (case.log_reader.call_count, case.detector.call_count, case.search.call_count, len(case.embedder.calls))
    assert case.app.session_state[ANALYSIS_REVISION] == 1
    assert case.app.session_state[RETRIEVAL_REVISION] == 1


def test_unsubmitted_edits_do_not_replace_or_authorize_a_different_draft(case):
    ready(case)
    authorize(case)
    saved = deepcopy(case.app.session_state[AI_KEY])
    find(case.app.selectbox, "Equipment (optional)").set_value(case.camera)
    find(case.app.text_area, "Problem description *").input("An unsubmitted camera problem")
    case.app.run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["signature"] == saved["signature"]
    assert case.app.session_state[AI_KEY]["messages"] == saved["messages"]
    assert find(case.app.checkbox, CONSENT).value
    assert case.search.call_count == case.log_reader.call_count == case.detector.call_count == 1


@pytest.mark.parametrize("change", ["description", "equipment", "no_equipment"])
def test_submitted_input_changes_revoke_consent_and_clear_pending_response(case, change):
    ready(case)
    authorize(case)
    old = pending(case)
    equipment = {"description": case.pump, "equipment": case.camera, "no_equipment": None}[change]
    description = "A different submitted vacuum problem" if change == "description" else "Vacuum gripper loses suction under load"
    submit(case, equipment, description)
    assert_revoked(case, old, disabled=True)
    assert case.search.call_count == 1
    search(case)
    assert not find(case.app.checkbox, CONSENT).value


@pytest.mark.parametrize("refresh", ["analysis", "retrieval"])
def test_identical_explicit_refresh_requires_new_consent_even_when_messages_are_identical(case, refresh):
    ready(case)
    authorize(case)
    old = pending(case)
    if refresh == "analysis":
        analyze(case)
        assert_revoked(case, old, disabled=True)
        assert case.app.session_state[ANALYSIS_REVISION] == 2
        search(case)
    else:
        search(case)
    assert_revoked(case, old, disabled=False)
    assert case.app.session_state[AI_KEY]["messages"] == old["messages"]
    assert case.app.session_state[RETRIEVAL_REVISION] == 2


@pytest.mark.parametrize("failure", ["analysis", "retrieval"])
def test_failed_explicit_attempts_increment_revision_and_revoke_authorization(case, failure):
    ready(case)
    authorize(case)
    old = pending(case)
    if failure == "analysis":
        case.detector.side_effect = ValueError("Synthetic log error")
        analyze(case)
        assert case.app.session_state[ANALYSIS_REVISION] == 2
    else:
        case.search.side_effect = EmbeddingError("Synthetic embedding error")
        search(case)
        assert case.app.session_state[RETRIEVAL_REVISION] == 2
    assert_revoked(case, old, disabled=True)


@pytest.mark.parametrize("change", ["log", "finding", "ranked_matches"])
def test_snapshot_changes_revoke_consent_without_automatic_analysis_or_search(case, change):
    ready(case)
    authorize(case)
    old = pending(case)
    if change == "log":
        case.app.session_state[ANALYSIS_KEY]["logs"][0]["message"] += " Updated snapshot observation"
    elif change == "finding":
        case.app.session_state[ANALYSIS_KEY]["findings"][0]["severity"] = "CRITICAL"
    else:
        case.app.session_state[MATCHES_KEY]["matches"][0]["similarity"] = .123
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old)
    assert case.log_reader.call_count == case.detector.call_count == case.search.call_count == 1


def test_changes_outside_truncated_preview_still_revoke_full_snapshot_authorization(case):
    ready(case, logs=False)
    case.app.session_state[MATCHES_KEY]["matches"][0]["root_cause"] = "x" * 2000
    case.app.run()
    assert not case.app.exception
    authorize(case)
    old = pending(case)
    case.app.session_state[MATCHES_KEY]["matches"][0]["root_cause"] += "Changed omitted tail"
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=False)
    assert case.app.session_state[AI_KEY]["messages"] == old["messages"]


@pytest.mark.parametrize("change", ["missing_index", "stale_history", "rebuilt_history", "rebuilt_index", "vectors", "invalid_vector"])
def test_index_changes_revoke_consent_and_hide_old_historical_evidence(case, change):
    ready(case)
    authorize(case)
    old = pending(case)
    with connect(case.path) as connection:
        if change == "missing_index":
            with connection:
                connection.execute("DELETE FROM semantic_index_metadata")
        elif change in ("stale_history", "rebuilt_history"):
            MaintenanceStore(connection).update_issue(case.issues[0], status="Resolved", root_cause="Changed finding")
        elif change in ("vectors", "invalid_vector"):
            vector = (np.ones(case.embedder.dimension, dtype="<f4").tobytes() if change == "vectors"
                      else "x" * (case.embedder.dimension * 4))
            with connection:
                connection.execute("UPDATE issue_embeddings SET vector=? WHERE issue_id=?", (vector, case.issues[0]))
    if change in ("rebuilt_history", "rebuilt_index"):
        case.rebuild()
    calls = len(case.embedder.calls)
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=True)
    assert not case.app.session_state[AI_KEY]["context"]["historical_findings"]
    assert any("Rebuild / refresh search index" in item.value or "Find Similar Maintenance Issues" in item.value
               for item in case.app.warning)
    assert len(case.embedder.calls) == calls
    assert case.log_reader.call_count == case.detector.call_count == case.search.call_count == 1


def test_empty_matches_do_not_authorize_generation_or_invent_records(case):
    submit(case, case.spare)
    search(case)
    state = case.app.session_state[AI_KEY]
    assert state["messages"]
    assert state["context"]["historical_findings"] == []
    assert not state["consent"]
    assert find(case.app.checkbox, CONSENT).disabled
    assert find(case.app.button, GENERATE).disabled


def test_stale_index_error_and_signature_remain_stable_without_automatic_retry(case):
    ready(case)
    with connect(case.path) as connection:
        MaintenanceStore(connection).update_issue(case.issues[0], status="Resolved", root_cause="Changed source")
    case.app.run()
    assert not case.app.exception
    previous = deepcopy(case.app.session_state[AI_KEY])
    factory_calls = case.factory.call_count
    case.app.run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["signature"] == previous["signature"]
    assert case.app.session_state[AI_KEY]["validation_errors"] == previous["validation_errors"]
    assert case.factory.call_count == factory_calls
    assert case.search.call_count == 1


def test_equipment_identity_change_revokes_consent_even_when_previewed_fields_are_unchanged(case):
    ready(case, logs=False)
    authorize(case)
    old = pending(case)
    with connect(case.path) as connection:
        MaintenanceStore(connection).update_equipment_status(case.pump, "Maintenance")
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=False)
    assert case.app.session_state[AI_KEY]["messages"] == old["messages"]


def test_database_identity_change_clears_consent_and_pending_response(case, tmp_path, monkeypatch):
    ready(case)
    authorize(case)
    old = pending(case)
    other = tmp_path / "copied_workspace.db"
    with connect(case.path) as source, connect(other) as target:
        source.backup(target)
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(other))
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=True)
    assert case.log_reader.call_count == case.detector.call_count == case.search.call_count == 1


def test_invalid_equipment_disables_consent_without_using_another_asset(case):
    ready(case)
    authorize(case)
    old = pending(case)
    case.app.session_state[DRAFT_KEY] = {"description": "Valid submitted symptom", "equipment_id": True}
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=True)
    assert case.app.session_state[AI_KEY]["messages"] is None
    assert case.app.session_state[AI_KEY]["serialized_messages"] is None


@pytest.mark.parametrize("builder", ["build_guided_context", "build_guided_messages"])
def test_evidence_building_errors_revoke_consent_and_remove_the_preview(case, monkeypatch, builder):
    from maintainai import guided_ui

    ready(case)
    authorize(case)
    old = pending(case)

    def failed(*args, **kwargs):
        raise ValidationError("Synthetic evidence preparation error")

    monkeypatch.setattr(guided_ui, builder, failed)
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, old, disabled=True)
    assert case.app.session_state[AI_KEY]["messages"] is None
    assert any("Synthetic evidence preparation error" in item.value for item in case.app.warning)


def test_returning_to_an_earlier_message_payload_does_not_resurrect_old_consent(case, monkeypatch):
    from maintainai import guided_ui

    ready(case)
    authorize(case)
    original_state = deepcopy(case.app.session_state[AI_KEY])
    original_builder = guided_ui.build_guided_messages

    def changed(context):
        messages = original_builder(context)
        messages[0]["content"] += "\nAdditional advisory instruction."
        return messages

    monkeypatch.setattr(guided_ui, "build_guided_messages", changed)
    case.app.run()
    assert not case.app.exception
    assert_revoked(case, original_state, disabled=False)
    monkeypatch.setattr(guided_ui, "build_guided_messages", original_builder)
    case.app.run()
    assert not case.app.exception
    assert case.app.session_state[AI_KEY]["signature"] == original_state["signature"]
    assert case.app.session_state[AI_KEY]["consent"] is False
    assert find(case.app.checkbox, CONSENT).value is False


def test_forged_disabled_click_without_consent_never_initializes_a_provider(case, monkeypatch):
    from maintainai import guided_ui

    ready(case)
    original = guided_ui.st.button

    def forced_click(label, *args, **kwargs):
        value = original(label, *args, **kwargs)
        return True if label == GENERATE else value

    monkeypatch.setattr(guided_ui.st, "button", forced_click)
    case.app.run()
    assert not case.app.exception
    assert find(case.app.button, GENERATE).disabled
    assert case.app.session_state[AI_KEY]["response"] is None


def test_consent_is_not_restored_in_a_new_streamlit_session(case):
    ready(case, logs=False)
    authorize(case)
    case.app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert not case.app.exception
    assert AI_KEY not in case.app.session_state
    find(case.app.sidebar.radio, "Navigate").set_value("Guided Troubleshooting").run()
    submit(case, case.pump)
    search(case)
    assert not find(case.app.checkbox, CONSENT).value
    assert case.app.session_state[AI_KEY]["consent"] is False
