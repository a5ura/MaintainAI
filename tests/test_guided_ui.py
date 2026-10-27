"""Guided troubleshooting validation, explicit local analysis, and session state."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.fault_detection import detect_fault_patterns
from maintainai.service import MaintenanceStore, ValidationError

DRAFT_KEY = "guided_troubleshooting_draft"
ANALYSIS_KEY = "guided_troubleshooting_log_analysis"


def find(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def guided(tmp_path, monkeypatch):
    path = tmp_path / "guided.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        equipment_id = store.create_equipment(asset_tag="GUIDED-001", name="Test pump", category="Pump")
        other_id = store.create_equipment(asset_tag="GUIDED-002", name="Test conveyor", category="Transport")
    guards = []
    for target in (
        "maintainai.semantic_search.SemanticSearch.__init__",
        "maintainai.rag.TroubleshootingAssistant.answer",
        "maintainai.fault_ui.detect_fault_patterns",
        "maintainai.fault_matching.match_fault_to_issues",
        "maintainai.fault_ui.match_fault_to_issues",
        "maintainai.fault_rag.troubleshoot_fault",
        "maintainai.fault_rag_ui.troubleshoot_fault",
        "maintainai.rag.generate_grounded_analysis",
        "maintainai.llm.provider_from_environment",
        "maintainai.rag.provider_from_environment",
        "maintainai.llm.ChatCompletionsProvider",
        "requests.sessions.Session.request",
    ):
        guard = Mock(side_effect=AssertionError("Guided troubleshooting must not run historical matching or AI/network calls"))
        monkeypatch.setattr(target, guard)
        guards.append(guard)
    log_reader = Mock(wraps=MaintenanceStore.list_logs)
    monkeypatch.setattr(MaintenanceStore, "list_logs", lambda self, **kwargs: log_reader(self, **kwargs))
    detector = Mock(wraps=detect_fault_patterns)
    monkeypatch.setattr("maintainai.guided_ui.detect_fault_patterns", detector)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.sidebar.radio, "Navigate").set_value("Guided Troubleshooting").run()
    assert not app.exception
    yield SimpleNamespace(app=app, path=path, equipment_id=equipment_id, other_id=other_id,
                          log_reader=log_reader, detector=detector)
    for guard in guards:
        guard.assert_not_called()


def test_initial_form_does_not_create_a_draft_or_run_analysis(guided):
    app = guided.app
    assert any(item.value == "Troubleshoot a Problem" for item in app.subheader)
    equipment = find(app.selectbox, "Equipment (optional)")
    assert equipment.value is None
    assert equipment.options == ["No equipment selected", "GUIDED-001 — Test pump", "GUIDED-002 — Test conveyor"]
    assert find(app.text_area, "Problem description *").proto.max_chars == 4000
    assert any("runs only when you click" in item.value for item in app.info)
    assert "Analyze Stored Logs" not in [item.label for item in app.button]
    assert DRAFT_KEY not in app.session_state
    guided.log_reader.assert_not_called()
    guided.detector.assert_not_called()
    find(app.text_area, "Problem description *").input("An unsubmitted problem")
    app.run()
    assert not app.exception
    assert DRAFT_KEY not in app.session_state
    guided.log_reader.assert_not_called()
    guided.detector.assert_not_called()


@pytest.mark.parametrize("description, error", [
    ("", "Problem description is required"),
    (" \n\t ", "Problem description is required"),
])
def test_invalid_description_does_not_save_a_draft(guided, description, error):
    app = guided.app
    find(app.text_area, "Problem description *").input(description)
    find(app.button, "Analyze Problem").click().run()
    assert not app.exception
    assert any(error in item.value for item in app.error)
    assert DRAFT_KEY not in app.session_state
    assert not app.success


def test_overlong_description_is_rejected_if_widget_limit_is_bypassed(guided, monkeypatch):
    from maintainai import guided_ui

    original = guided_ui.st.text_area

    def oversized_input(*args, **kwargs):
        original(*args, **kwargs)
        return "x" * 4001

    # Streamlit caps widget input itself; bypass that cap to exercise the
    # independent server-side validation before any draft is stored.
    monkeypatch.setattr(guided_ui.st, "text_area", oversized_input)
    find(guided.app.button, "Analyze Problem").click().run()
    assert not guided.app.exception
    assert any("at most 4000 characters" in item.value for item in guided.app.error)
    assert DRAFT_KEY not in guided.app.session_state


@pytest.mark.parametrize("equipment_id", [True, "1", 999999])
def test_invalid_equipment_id_cannot_be_stored(guided, monkeypatch, equipment_id):
    from maintainai import guided_ui

    original = guided_ui.st.selectbox

    def invalid_selection(*args, **kwargs):
        original(*args, **kwargs)
        return equipment_id

    monkeypatch.setattr(guided_ui.st, "selectbox", invalid_selection)
    find(guided.app.text_area, "Problem description *").input("Intermittent pressure loss")
    find(guided.app.button, "Analyze Problem").click().run()
    assert not guided.app.exception
    assert any("valid registered equipment ID" in item.value for item in guided.app.error)
    assert DRAFT_KEY not in guided.app.session_state


@pytest.mark.parametrize("selected", [False, True])
def test_valid_submission_stores_clean_description_and_optional_integer_id(guided, selected):
    app = guided.app
    equipment_id = guided.other_id if selected else None
    find(app.selectbox, "Equipment (optional)").set_value(equipment_id)
    find(app.text_area, "Problem description *").input(" \n Pump loses pressure\nunder load. \t")
    find(app.button, "Analyze Problem").click().run()
    assert not app.exception
    assert app.session_state[DRAFT_KEY] == {
        "description": "Pump loses pressure\nunder load.", "equipment_id": equipment_id,
    }
    if selected:
        assert type(app.session_state[DRAFT_KEY]["equipment_id"]) is int
    assert app.success
    assert find(app.button, "Analyze Stored Logs").disabled is (not selected)
    if not selected:
        assert any("description-only troubleshooting" in item.value for item in app.info)
    assert any(item.value == "Pump loses pressure\nunder load." for item in app.text)
    assert any(item.value == "Equipment: " + (
        "GUIDED-002 — Test conveyor" if selected else "No equipment selected") for item in app.text)
    guided.log_reader.assert_not_called()
    guided.detector.assert_not_called()


def test_4000_character_description_is_accepted(guided):
    app = guided.app
    find(app.text_area, "Problem description *").input("x" * 4000)
    find(app.button, "Analyze Problem").click().run()
    assert not app.exception and not app.error
    assert app.session_state[DRAFT_KEY]["description"] == "x" * 4000


def test_draft_survives_reruns_invalid_submission_and_sidebar_navigation(guided):
    app = guided.app
    find(app.selectbox, "Equipment (optional)").set_value(guided.equipment_id)
    find(app.text_area, "Problem description *").input("Submitted pressure problem")
    find(app.button, "Analyze Problem").click().run()
    saved = deepcopy(app.session_state[DRAFT_KEY])
    app.run()
    assert not app.exception
    assert app.session_state[DRAFT_KEY] == saved
    find(app.text_area, "Problem description *").input(" ")
    find(app.button, "Analyze Problem").click().run()
    assert not app.exception and app.error
    assert app.session_state[DRAFT_KEY] == saved
    find(app.sidebar.radio, "Navigate").set_value("Home").run()
    find(app.button, "Troubleshoot a Problem").click().run()
    assert not app.exception
    assert find(app.sidebar.radio, "Navigate").value == "Guided Troubleshooting"
    assert app.session_state[DRAFT_KEY] == saved
    assert find(app.selectbox, "Equipment (optional)").value == guided.equipment_id
    assert find(app.text_area, "Problem description *").value == saved["description"]


def test_empty_registry_still_accepts_description_without_associating_equipment(guided):
    with connect(guided.path) as connection:
        with connection:
            connection.execute("DELETE FROM equipment")
    app = guided.app.run()
    assert not app.exception
    assert find(app.selectbox, "Equipment (optional)").options == ["No equipment selected"]
    find(app.text_area, "Problem description *").input("Unregistered machine stops intermittently")
    find(app.button, "Analyze Problem").click().run()
    assert not app.exception
    assert app.session_state[DRAFT_KEY]["equipment_id"] is None
    with connect(guided.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM equipment").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM issues").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM equipment_logs").fetchone()[0] == 0


def submit_problem(guided, equipment_id, description="Pump pressure drops during operation"):
    find(guided.app.selectbox, "Equipment (optional)").set_value(equipment_id)
    find(guided.app.text_area, "Problem description *").input(description)
    find(guided.app.button, "Analyze Problem").click().run()
    assert not guided.app.exception


def add_pattern_logs(guided):
    with connect(guided.path) as connection:
        store = MaintenanceStore(connection)
        ids = store.import_log_records([
            {"equipment_id": equipment_id, "timestamp": f"2026-10-25T12:00:{second:02d}Z",
             "severity": severity, "event_code": code, "message": "Synthetic observed event"}
            for equipment_id, second, severity, code in [
                (guided.equipment_id, 0, "WARNING", "PRESSURE_LOW"),
                (guided.equipment_id, 10, "ERROR", "PUMP_STOP"),
                (guided.equipment_id, 20, "ERROR", "PUMP_STOP"),
                (guided.other_id, 30, "ERROR", "CONVEYOR_STOP"),
                (guided.other_id, 40, "ERROR", "CONVEYOR_STOP"),
            ]
        ])
    return ids


def analyze_stored_logs(guided):
    find(guided.app.button, "Analyze Stored Logs").click().run()
    assert not guided.app.exception


def test_explicit_analysis_uses_only_submitted_asset_and_real_detection_output(guided):
    ids = add_pattern_logs(guided)
    submit_problem(guided, guided.equipment_id)
    guided.log_reader.assert_not_called()
    guided.detector.assert_not_called()
    analyze_stored_logs(guided)
    assert guided.log_reader.call_args.kwargs == {"equipment_id": guided.equipment_id}
    assert guided.log_reader.call_count == 1
    guided.detector.assert_called_once()
    logs = guided.detector.call_args.args[0]
    assert {log["id"] for log in logs} == set(ids[:3])
    assert {log["equipment_id"] for log in logs} == {guided.equipment_id}
    result = guided.app.session_state[ANALYSIS_KEY]
    assert result["draft"] == guided.app.session_state[DRAFT_KEY]
    assert result["findings"] == detect_fault_patterns(logs)
    assert {metric.label: metric.value for metric in guided.app.metric} == {
        "Logs examined": "3", "Detected fault patterns": "2",
    }
    for finding in result["findings"]:
        assert any(finding["pattern_type"] in item.value for item in guided.app.subheader)
        assert any(item.value == finding["explanation"] for item in guided.app.text)
        assert any(item.value == "Supporting log IDs: " + ", ".join(map(str, finding["log_ids"]))
                   for item in guided.app.caption)
    assert "fault_analysis_result" not in guided.app.session_state


def test_empty_logs_show_zero_counts_and_informative_state(guided):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    guided.detector.assert_called_once_with([])
    assert [metric.value for metric in guided.app.metric] == ["0", "0"]
    assert any("No stored logs were found" in item.value for item in guided.app.info)
    guided.app.run()
    assert not guided.app.exception
    assert guided.log_reader.call_count == guided.detector.call_count == 1


def test_logs_without_patterns_do_not_claim_equipment_is_healthy(guided):
    with connect(guided.path) as connection:
        MaintenanceStore(connection).create_log(equipment_id=guided.equipment_id,
            timestamp="2026-10-25T12:00:00Z", severity="ERROR", event_code="ISOLATED", message="Single stop")
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    assert [metric.value for metric in guided.app.metric] == ["1", "0"]
    assert any("No supported fault patterns were detected" in item.value and "does not establish" in item.value
               for item in guided.app.info)


def test_analysis_persists_without_automatic_work_on_edit_rerun_or_navigation(guided):
    add_pattern_logs(guided)
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    saved = deepcopy(guided.app.session_state[ANALYSIS_KEY])
    find(guided.app.selectbox, "Equipment (optional)").set_value(guided.other_id)
    find(guided.app.text_area, "Problem description *").input("Unsubmitted conveyor problem")
    guided.app.run()
    assert not guided.app.exception
    assert guided.app.session_state[ANALYSIS_KEY] == saved
    find(guided.app.sidebar.radio, "Navigate").set_value("Home").run()
    find(guided.app.button, "Troubleshoot a Problem").click().run()
    assert not guided.app.exception
    assert guided.app.session_state[ANALYSIS_KEY] == saved
    assert [metric.value for metric in guided.app.metric] == ["3", "2"]
    assert guided.log_reader.call_count == guided.detector.call_count == 1


def test_analysis_click_uses_submitted_equipment_despite_unsubmitted_edits(guided):
    add_pattern_logs(guided)
    submit_problem(guided, guided.equipment_id)
    find(guided.app.selectbox, "Equipment (optional)").set_value(guided.other_id)
    find(guided.app.text_area, "Problem description *").input("Not yet submitted")
    analyze_stored_logs(guided)
    assert guided.log_reader.call_args.kwargs == {"equipment_id": guided.equipment_id}
    assert guided.app.session_state[ANALYSIS_KEY]["draft"]["equipment_id"] == guided.equipment_id


@pytest.mark.parametrize("change", ["description", "equipment", "no_equipment"])
def test_changed_submitted_draft_invalidates_analysis_without_reanalysis(guided, change):
    add_pattern_logs(guided)
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    equipment_id = {"description": guided.equipment_id, "equipment": guided.other_id, "no_equipment": None}[change]
    description = "Different submitted pressure problem" if change == "description" else "Pump pressure drops during operation"
    submit_problem(guided, equipment_id, description)
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    assert guided.log_reader.call_count == guided.detector.call_count == 1
    guided.app.run()
    assert not guided.app.exception
    assert ANALYSIS_KEY not in guided.app.session_state
    if equipment_id is not None:
        analyze_stored_logs(guided)
        assert guided.log_reader.call_args.kwargs == {"equipment_id": equipment_id}
        assert guided.log_reader.call_count == guided.detector.call_count == 2


@pytest.mark.parametrize("equipment_id", [None, True, "1", -1, 999999])
def test_invalid_submitted_asset_cannot_analyze_even_if_disabled_button_is_bypassed(guided, monkeypatch, equipment_id):
    from maintainai import guided_ui

    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    guided.app.session_state[DRAFT_KEY] = {"description": "Existing draft", "equipment_id": equipment_id}
    original = guided_ui.st.button

    def forced_click(label, *args, **kwargs):
        value = original(label, *args, **kwargs)
        return True if label == "Analyze Stored Logs" else value

    monkeypatch.setattr(guided_ui.st, "button", forced_click)
    guided.app.run()
    assert not guided.app.exception
    assert find(guided.app.button, "Analyze Stored Logs").disabled
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    assert guided.log_reader.call_count == guided.detector.call_count == 1


def test_deleted_equipment_invalidates_results_and_disables_analysis(guided):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    with connect(guided.path) as connection:
        with connection:
            connection.execute("DELETE FROM equipment WHERE id = ?", (guided.equipment_id,))
    guided.app.run()
    assert not guided.app.exception
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    assert find(guided.app.button, "Analyze Stored Logs").disabled
    assert any("no longer valid or registered" in item.value for item in guided.app.warning)
    assert guided.log_reader.call_count == guided.detector.call_count == 1


@pytest.mark.parametrize("failure", ["retrieval", "detection"])
def test_failed_refresh_discards_old_results_without_automatic_retry(guided, failure):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    target = guided.log_reader if failure == "retrieval" else guided.detector
    target.side_effect = ValidationError("Equipment disappeared") if failure == "retrieval" else ValueError("Invalid timestamp")
    analyze_stored_logs(guided)
    assert any("could not be completed" in item.value for item in guided.app.error)
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    calls = guided.log_reader.call_count, guided.detector.call_count
    guided.app.run()
    assert not guided.app.exception
    assert (guided.log_reader.call_count, guided.detector.call_count) == calls


def test_new_logs_require_an_explicit_refresh(guided):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    add_pattern_logs(guided)
    guided.app.run()
    assert not guided.app.exception
    assert [metric.value for metric in guided.app.metric] == ["0", "0"]
    assert guided.log_reader.call_count == guided.detector.call_count == 1
    analyze_stored_logs(guided)
    assert [metric.value for metric in guided.app.metric] == ["3", "2"]


def test_database_switch_hides_previous_analysis_even_with_same_equipment_id(guided, tmp_path, monkeypatch):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    other_path = tmp_path / "other_workspace.db"
    with connect(other_path) as connection:
        initialize(connection)
        equipment_id = MaintenanceStore(connection).create_equipment(asset_tag="OTHER", name="Other asset", category="Test")
        assert equipment_id == guided.equipment_id
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(other_path))
    guided.app.run()
    assert not guided.app.exception
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    assert guided.log_reader.call_count == guided.detector.call_count == 1


def test_replacement_asset_reusing_an_id_does_not_inherit_analysis(guided):
    submit_problem(guided, guided.equipment_id)
    analyze_stored_logs(guided)
    with connect(guided.path) as connection:
        with connection:
            connection.execute("DELETE FROM equipment")
        replacement_id = MaintenanceStore(connection).create_equipment(
            asset_tag="REPLACEMENT", name="Replacement asset", category="Test")
        assert replacement_id == guided.equipment_id
    guided.app.run()
    assert not guided.app.exception
    assert ANALYSIS_KEY not in guided.app.session_state
    assert not guided.app.metric
    assert guided.log_reader.call_count == guided.detector.call_count == 1
