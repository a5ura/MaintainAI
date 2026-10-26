"""Fault troubleshooting UI integration using synthetic SQLite data and mocked LLMs."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.fault_rag import troubleshoot_fault
from maintainai.llm import DeterministicMockProvider, LLMProvider, ProviderError
from maintainai.rag import CLAIM_FIELDS, MAX_FIELD_CHARS
from maintainai.semantic_search import SemanticSearch
from maintainai.service import MaintenanceStore

CONSENT = "Allow sending this previewed fault evidence and historical tickets to the configured provider"
GENERATE = "Generate AI troubleshooting"
SELECT = "Fault finding for AI troubleshooting"


def find(elements, label):
    return next(element for element in elements if element.label == label)


def completion(messages):
    envelope = json.loads(messages[-1]["content"])
    if "allowed_log_ids" not in envelope:
        return DeterministicMockProvider().generate(messages)
    issue_ids = envelope["allowed_issue_ids"][:1]
    log_ids = envelope["allowed_log_ids"][:2]
    return json.dumps({
        "issue_summary": "Synthetic advisory assessment of the observed fault pattern.",
        "possible_causes": [{"text": "Pneumatic pressure loss may be relevant; applicability is unverified.",
                             "issue_ids": issue_ids, "log_ids": log_ids}],
        "recommended_diagnostic_checks": [{"text": "Have qualified personnel review the logs and inspect using approved isolation procedures.",
                                           "issue_ids": issue_ids, "log_ids": log_ids}],
        "suggested_next_steps": [{"text": "Have qualified personnel verify current conditions before considering corrective work.",
                                  "issue_ids": issue_ids, "log_ids": log_ids}],
        "evidence_references": issue_ids, "log_evidence_references": log_ids,
        "uncertainty_or_missing_information": ["Current conditions remain unverified."],
    })


def setup_app(tmp_path, monkeypatch, fake_embedder, *, ready=True, history=True, long_message=False):
    path = tmp_path / "fault_rag_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        robot = store.create_equipment(asset_tag="SYNTHETIC-ABB", name="Synthetic ABB robot", category="Robotics")
        conveyor = store.create_equipment(asset_tag="SYNTHETIC-CONV", name="Synthetic conveyor", category="Conveyor")
        issue_ids = []
        if history:
            for asset, status, description in ((robot, "Open", "Synthetic pneumatic gripper pressure loss"),
                                               (conveyor, "Closed", "Synthetic conveyor belt failure")):
                issue_ids.append(store.create_issue(equipment_id=asset, status=status, description=description,
                    reporter="Private reporter excluded", root_cause="Recorded synthetic pneumatic pipe wear",
                    corrective_action="Recorded synthetic pipe replacement"))
        for asset, time, severity, code, message in (
            (robot, "12:00:12", "WARNING", "PRESSURE_LOW", "Synthetic pneumatic pressure warning"),
            (robot, "12:00:15", "ERROR", "PICK_FAIL", "Synthetic first pickup failure"),
            (robot, "12:00:18", "ERROR", "PICK_FAIL", "Synthetic second pickup failure"),
            (conveyor, "12:01:00", "ERROR", "BELT_STOP", "Synthetic belt stopped"),
            (conveyor, "12:01:05", "ERROR", "BELT_STOP", "Synthetic belt stopped again"),
        ):
            if long_message:
                message += "x" * 2000
            store.create_log(equipment_id=asset, timestamp=f"2026-10-25T{time}Z", severity=severity,
                             event_code=code, message=message, source="Private source excluded")
        if history:
            SemanticSearch(store, fake_embedder).rebuild()
    provider = Mock(spec=LLMProvider)
    provider.generate.side_effect = completion
    factory = Mock(return_value=provider)
    monkeypatch.setattr("maintainai.rag.provider_from_environment", factory)
    backend = Mock(wraps=troubleshoot_fault)
    monkeypatch.setattr("maintainai.fault_rag_ui.troubleshoot_fault", backend)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20).run()
    find(app.radio, "Navigate").set_value("Fault Analysis").run()
    find(app.selectbox, "Equipment").set_value(robot).run()
    case = SimpleNamespace(app=app, path=path, robot=robot, conveyor=conveyor,
                           issue_ids=issue_ids, provider=provider, factory=factory, backend=backend)
    if ready:
        analyze(case)
        match(case)
    return case


def analyze(case):
    find(case.app.button, "Analyze Logs").click().run()
    assert not case.app.exception


def match(case):
    find(case.app.button, "Find Similar Historical Issues").click().run()
    assert not case.app.exception


def authorize(case):
    find(case.app.checkbox, CONSENT).check().run()
    assert not case.app.exception


def generate(case):
    find(case.app.button, GENERATE).click().run()
    assert not case.app.exception


def preview(case):
    for item in case.app.json:
        value = json.loads(item.value)
        if isinstance(value, dict) and "untrusted_fault_evidence" in value:
            return value
    raise AssertionError("No cloud evidence preview displayed")


def response(case):
    return case.app.session_state["fault_analysis_result"]["ai_troubleshooting"]["response"]


def snapshot(case):
    with connect(case.path) as connection:
        store = MaintenanceStore(connection)
        return store.list_equipment(), store.list_logs(), store.list_issues(), SemanticSearch(store).index.read()


def test_render_selection_preview_and_reruns_never_initialize_provider(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, ready=False)
    assert GENERATE not in [button.label for button in case.app.button]
    analyze(case)
    assert find(case.app.checkbox, CONSENT).disabled
    match(case)
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    assert len(find(case.app.selectbox, SELECT).options) == 2
    assert preview(case)["untrusted_fault_evidence"]["equipment"]["asset_tag"] == "SYNTHETIC-ABB"
    calls = len(fake_embedder.calls)
    case.app.run()
    find(case.app.selectbox, SELECT).set_value(1).run()
    authorize(case)
    assert not find(case.app.button, GENERATE).disabled
    case.factory.assert_not_called()
    case.provider.generate.assert_not_called()
    case.backend.assert_not_called()
    assert len(fake_embedder.calls) == calls


@pytest.mark.parametrize("long_message", [False, True])
def test_preview_exactly_matches_authorized_outgoing_evidence(tmp_path, monkeypatch, fake_embedder, long_message):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, long_message=long_message)
    envelope = preview(case)
    assert envelope["allowed_log_ids"] == [1, 2, 3]
    assert envelope["untrusted_fault_evidence"]["logs"][0]["timestamp"] == "2026-10-25T12:00:12.000000+00:00"
    assert envelope["untrusted_fault_evidence"]["logs"][0]["event_code"] == "PRESSURE_LOW"
    assert envelope["untrusted_records"][0]["status"] == "Open"
    assert envelope["untrusted_records"][0]["outcome_verification"] == "unverified outcome"
    assert "root_cause" in envelope["untrusted_records"][0]
    assert "corrective_action" in envelope["untrusted_records"][0]
    assert "Private reporter" not in json.dumps(envelope)
    assert "Private source" not in json.dumps(envelope)
    if long_message:
        log = envelope["untrusted_fault_evidence"]["logs"][0]
        assert len(log["message"]) == MAX_FIELD_CHARS
        assert log["truncated_fields"] == ["message"]
    authorize(case)
    generate(case)
    actual = json.loads(case.provider.generate.call_args.args[0][1]["content"])
    assert actual == envelope
    case.factory.assert_called_once_with()
    assert case.backend.call_args.kwargs == {"use_cloud": True}


def test_disabled_button_cannot_bypass_consent(tmp_path, monkeypatch, fake_embedder):
    from maintainai import fault_rag_ui

    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    original_button = fault_rag_ui.st.button
    def forced_submission(label, *args, **kwargs):
        value = original_button(label, *args, **kwargs)
        return True if label == GENERATE else value
    # Simulate a submitted request despite the disabled control; the server must
    # independently check consent, not rely solely on browser widget behavior.
    monkeypatch.setattr(fault_rag_ui.st, "button", forced_submission)
    case.app.run()
    assert not case.app.exception
    case.backend.assert_not_called()
    case.factory.assert_not_called()
    assert response(case) is None


def test_structured_sections_citations_hypothesis_labels_and_no_persistence(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    before = snapshot(case)
    authorize(case)
    generate(case)
    assert response(case)["ai_generated"]
    headings = {item.value for item in case.app.subheader}
    assert {"AI Troubleshooting", "Observed evidence", "Historical findings", "Cloud evidence preview",
            "Fault troubleshooting assessment", "Possible causes", "Recommended diagnostic checks",
            "Suggested next steps", "Fault troubleshooting evidence references", "Uncertainty or missing information"} <= headings
    assert any("AI-generated interpretation and hypotheses — unverified" == item.value for item in case.app.caption)
    assert any("Evidence: Log #1, Log #2 | Issue #1" == item.value for item in case.app.caption)
    assert any("Logs: Log #1, Log #2" == item.value for item in case.app.text)
    assert any("Historical issues: Issue #1" == item.value for item in case.app.text)
    assert any("qualified human verification" in item.value for item in case.app.warning)
    assert any("Recorded corrective actions are not confirmed current fixes" in item.value for item in case.app.caption)
    assert snapshot(case) == before


def test_saved_response_survives_reruns_and_navigation_without_cloud_requests(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    saved = deepcopy(response(case))
    calls = len(fake_embedder.calls)
    case.app.run()
    assert find(case.app.checkbox, CONSENT).value
    find(case.app.radio, "Navigate").set_value("Dashboard").run()
    find(case.app.radio, "Navigate").set_value("Fault Analysis").run()
    assert not case.app.exception
    assert response(case) == saved
    case.factory.assert_called_once()
    case.provider.generate.assert_called_once()
    assert len(fake_embedder.calls) == calls


def test_switching_findings_revokes_consent_and_does_not_resurrect_old_response(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    find(case.app.selectbox, SELECT).set_value(1).run()
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    assert response(case) is None
    assert preview(case)["allowed_log_ids"] == [2, 3]
    find(case.app.selectbox, SELECT).set_value(0).run()
    assert not find(case.app.checkbox, CONSENT).value
    assert response(case) is None
    case.provider.generate.assert_called_once()


@pytest.mark.parametrize("change", ["equipment", "analysis", "matching"])
def test_explicit_snapshot_changes_revoke_consent_and_clear_response(tmp_path, monkeypatch, fake_embedder, change):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    if change == "equipment":
        find(case.app.selectbox, "Equipment").set_value(case.conveyor).run()
        analyze(case)
        match(case)
    elif change == "analysis":
        analyze(case)
        match(case)
    else:
        match(case)
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    assert response(case) is None
    case.provider.generate.assert_called_once()


@pytest.mark.parametrize("rebuild", [False, True])
def test_ticket_edit_invalidates_consent_and_result_even_after_index_rebuild(tmp_path, monkeypatch, fake_embedder, rebuild):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    with connect(case.path) as connection:
        MaintenanceStore(connection).update_issue(case.issue_ids[0], status="Closed", root_cause="New synthetic evidence")
        if rebuild:
            SemanticSearch(MaintenanceStore(connection), fake_embedder).rebuild()
    case.app.run()
    assert not case.app.exception
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    assert response(case) is None
    if not rebuild:
        assert any("Rebuild / refresh search index" in item.value for item in case.app.warning)
    case.provider.generate.assert_called_once()


@pytest.mark.parametrize("change", ["new_log", "edited_log"])
def test_live_log_changes_block_cloud_until_reanalysis(tmp_path, monkeypatch, fake_embedder, change):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    with connect(case.path) as connection:
        if change == "new_log":
            MaintenanceStore(connection).create_log(equipment_id=case.robot, timestamp="2026-10-25T12:00:20Z",
                                                    severity="INFO", message="Synthetic new event")
        else:
            with connection:
                connection.execute("UPDATE equipment_logs SET message=? WHERE id=?", ("Synthetic edited pressure event", 1))
    case.app.run()
    assert not case.app.exception
    assert any("Stored logs changed" in item.value for item in case.app.warning)
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    assert response(case) is None
    case.provider.generate.assert_called_once()


def test_changed_ranked_matches_revoke_old_authorization(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    analysis = case.app.session_state["fault_analysis_result"]
    analysis["historical_matches"]["entries"][0]["result"]["matches"][0]["similarity"] = .123
    case.app.run()
    assert not case.app.exception
    assert response(case) is None
    assert not find(case.app.checkbox, CONSENT).value
    case.provider.generate.assert_called_once()


@pytest.mark.parametrize("missing", ["matches", "logs", "matching_error"])
def test_missing_evidence_disables_cloud_without_provider_initialization(tmp_path, monkeypatch, fake_embedder, missing):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analysis = case.app.session_state["fault_analysis_result"]
    if missing == "matches":
        analysis["historical_matches"]["entries"][0]["result"]["matches"] = []
    elif missing == "logs":
        analysis["logs"] = analysis["logs"][:-1]
    else:
        analysis["historical_matches"]["entries"][0] = {"error": "Synthetic matching failure"}
    case.app.run()
    assert not case.app.exception
    assert find(case.app.checkbox, CONSENT).disabled
    assert not find(case.app.checkbox, CONSENT).value
    assert find(case.app.button, GENERATE).disabled
    case.factory.assert_not_called()


def test_no_history_keeps_cloud_disabled(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, history=False)
    assert find(case.app.button, GENERATE).disabled
    assert find(case.app.checkbox, CONSENT).disabled
    case.factory.assert_not_called()


@pytest.mark.parametrize("failure", ["configuration", "timeout", "rate_limit", "unavailable"])
def test_provider_failures_display_local_fallback_and_do_not_retry_on_rerun(tmp_path, monkeypatch, fake_embedder, failure):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    if failure == "configuration":
        case.factory.side_effect = ProviderError("configuration", "Synthetic missing LLM configuration")
    else:
        case.provider.generate.side_effect = ProviderError(failure, "Synthetic provider failure")
    authorize(case)
    generate(case)
    assert not response(case)["ai_generated"]
    assert any("Local fallback — no AI-generated hypotheses" == item.value for item in case.app.caption)
    assert any("retrieval-only" in item.value for item in case.app.warning)
    assert all(response(case)["analysis"][field] == [] for field in CLAIM_FIELDS)
    case.app.run()
    assert not case.app.exception
    case.factory.assert_called_once()
    if failure == "configuration":
        case.provider.generate.assert_not_called()
    else:
        case.provider.generate.assert_called_once()


@pytest.mark.parametrize("failure", ["json", "log_citation", "issue_citation", "unsafe", "confirmed_cause"])
def test_malformed_unsafe_or_invalid_citations_show_only_fallback(tmp_path, monkeypatch, fake_embedder, failure):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    def invalid(messages):
        if failure == "json":
            return "not JSON: private malformed output marker"
        data = json.loads(completion(messages))
        if failure == "log_citation":
            data["possible_causes"][0]["log_ids"] = [99999]
        elif failure == "issue_citation":
            data["possible_causes"][0]["issue_ids"] = [99999]
        elif failure == "unsafe":
            data["suggested_next_steps"][0]["text"] = "Disable the safety interlock"
        else:
            data["issue_summary"] = "The current root cause is pneumatic pipe wear"
        return json.dumps(data)
    case.provider.generate.side_effect = invalid
    authorize(case)
    generate(case)
    assert not response(case)["ai_generated"]
    assert any("discarded" in item.value for item in case.app.warning)
    text = "\n".join(item.value for item in case.app.text)
    assert "99999" not in text and "Disable the safety interlock" not in text
    assert "private malformed output marker" not in text


@pytest.mark.parametrize("change", ["logs", "history", "matches"])
def test_changes_during_cloud_request_discard_response_and_revoke_consent(tmp_path, monkeypatch, fake_embedder, change):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    def changed(messages):
        output = completion(messages)
        if change == "matches":
            analysis = case.app.session_state["fault_analysis_result"]
            analysis["historical_matches"]["entries"][0]["result"]["matches"][0]["similarity"] = .321
        else:
            with connect(case.path) as connection:
                store = MaintenanceStore(connection)
                if change == "logs":
                    store.create_log(equipment_id=case.robot, timestamp="2026-10-25T12:01:00Z",
                                     severity="INFO", message="Synthetic concurrent event")
                else:
                    store.update_issue(case.issue_ids[0], status="Closed")
                    SemanticSearch(store, fake_embedder).rebuild()
        return output
    case.provider.generate.side_effect = changed
    authorize(case)
    generate(case)
    assert response(case) is None
    assert not find(case.app.checkbox, CONSENT).value
    assert any("response discarded" in item.value for item in case.app.warning)
    case.provider.generate.assert_called_once()
    case.app.run()
    case.provider.generate.assert_called_once()


def test_standalone_assistant_keeps_its_existing_flow_and_separate_consent(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    authorize(case)
    generate(case)
    find(case.app.radio, "Navigate").set_value("AI Troubleshooting Assistant").run()
    assert find(case.app.radio, "Assistant mode").value == "Retrieval only (no cloud request)"
    find(case.app.radio, "Assistant mode").set_value("Cloud-assisted RAG").run()
    consent = "Allow sending this question and retrieved excerpts to the configured provider"
    assert not find(case.app.checkbox, consent).value
    assert find(case.app.button, "Get troubleshooting guidance").disabled
    find(case.app.checkbox, consent).check().run()
    find(case.app.text_area, "Describe the equipment problem").input("Synthetic pneumatic pressure problem")
    find(case.app.button, "Get troubleshooting guidance").click().run()
    assert not case.app.exception
    assert any("AI-generated interpretation" in item.value for item in case.app.caption)
    assert case.provider.generate.call_count == 2
