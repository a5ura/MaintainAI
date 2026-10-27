"""Description-aware retrieval with the real search/index and fake local embeddings."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.embeddings import EmbeddingError
from maintainai.fault_detection import detect_fault_patterns
from maintainai.semantic_search import SemanticSearch
from maintainai.service import MaintenanceStore

DRAFT_KEY = "guided_troubleshooting_draft"
ANALYSIS_KEY = "guided_troubleshooting_log_analysis"
MATCHES_KEY = "guided_troubleshooting_matches"
FIND = "Find Similar Maintenance Issues"


def find(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def case(tmp_path, monkeypatch, fake_embedder):
    path = tmp_path / "guided_matching.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    original_search, original_rebuild = SemanticSearch.search, SemanticSearch.rebuild
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        pump = store.create_equipment(asset_tag="MATCH-PUMP", name="Pump", category="Test")
        camera = store.create_equipment(asset_tag="MATCH-CAM", name="Camera", category="Test")
        spare = store.create_equipment(asset_tag="MATCH-SPARE", name="Spare", category="Test")
        issues = [store.create_issue(equipment_id=pump, description=f"Vacuum suction seal leak {number}",
                  reporter="Private reporter", status="Open" if number == 0 else "Closed",
                  root_cause="Recorded seal observation", corrective_action="Recorded cup replacement")
                  for number in range(6)]
        camera_issue = store.create_issue(equipment_id=camera, description="Camera image focus lens illumination",
                                         reporter="Private reporter", status="In Progress")
        SemanticSearch(store, fake_embedder).rebuild()
    fake_embedder.calls.clear()
    guards = []
    for target in ("maintainai.rag.TroubleshootingAssistant.answer", "maintainai.fault_matching.match_fault_to_issues",
                   "maintainai.fault_rag.troubleshoot_fault", "maintainai.fault_rag_ui.troubleshoot_fault",
                   "maintainai.rag.generate_grounded_analysis", "maintainai.rag.provider_from_environment",
                   "maintainai.llm.provider_from_environment", "maintainai.llm.ChatCompletionsProvider",
                   "requests.sessions.Session.request", "maintainai.semantic_search.SemanticSearch.rebuild"):
        guard = Mock(side_effect=AssertionError("No cloud calls, fault-only matching, or automatic index rebuilds"))
        monkeypatch.setattr(target, guard)
        guards.append(guard)
    embedder_factory = Mock(return_value=fake_embedder)
    monkeypatch.setattr("maintainai.guided_ui.MiniLMEmbedder", embedder_factory)
    search_spy = Mock(wraps=original_search)
    monkeypatch.setattr(SemanticSearch, "search", lambda self, *args, **kwargs: search_spy(self, *args, **kwargs))
    log_reader = Mock(wraps=MaintenanceStore.list_logs)
    monkeypatch.setattr(MaintenanceStore, "list_logs", lambda self, **kwargs: log_reader(self, **kwargs))
    detector = Mock(wraps=detect_fault_patterns)
    monkeypatch.setattr("maintainai.guided_ui.detect_fault_patterns", detector)

    def rebuild():
        with connect(path) as connection:
            original_rebuild(SemanticSearch(MaintenanceStore(connection), fake_embedder))

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.sidebar.radio, "Navigate").set_value("Guided Troubleshooting").run()
    assert not app.exception
    yield SimpleNamespace(app=app, path=path, pump=pump, camera=camera, spare=spare, issues=issues,
                          camera_issue=camera_issue, embedder=fake_embedder, factory=embedder_factory,
                          search=search_spy, log_reader=log_reader, detector=detector, rebuild=rebuild)
    for guard in guards:
        guard.assert_not_called()


def submit(case, equipment_id=None, description="Vacuum gripper loses suction under load"):
    find(case.app.selectbox, "Equipment (optional)").set_value(equipment_id)
    find(case.app.text_area, "Problem description *").input(description)
    find(case.app.button, "Analyze Problem").click().run()
    assert not case.app.exception


def search(case):
    find(case.app.button, FIND).click().run()
    assert not case.app.exception


def add_logs(case, *, patterns=True):
    with connect(case.path) as connection:
        store = MaintenanceStore(connection)
        rows = [(case.pump, 0, "WARNING", "PRESSURE_LOW", "Pressure warning"),
                (case.pump, 10, "ERROR", "PICK_FAIL", "Suction pick failure"),
                (case.pump, 20, "ERROR", "PICK_FAIL", "Suction pick failure"),
                (case.camera, 30, "ERROR", "OTHER_ASSET", "Unrelated camera secret")]
        if not patterns:
            rows = [(case.pump, 0, "INFO", "PRESSURE_READING", "Observed vacuum pressure reading")]
        for equipment_id, second, severity, code, message in rows:
            store.create_log(equipment_id=equipment_id, timestamp=f"2026-10-25T12:00:{second:02d}Z",
                             severity=severity, event_code=code, message=message)


def analyze(case):
    find(case.app.button, "Analyze Stored Logs").click().run()
    assert not case.app.exception


def test_description_only_retrieval_is_explicit_ranked_bounded_and_read_only(case):
    assert FIND not in [button.label for button in case.app.button]
    submit(case)
    case.app.run()
    case.search.assert_not_called()
    case.factory.assert_not_called()
    assert not case.embedder.calls
    with connect(case.path) as connection:
        before = MaintenanceStore(connection).list_issues(), SemanticSearch(MaintenanceStore(connection), case.embedder).index.read()
    search(case)
    result = case.app.session_state[MATCHES_KEY]
    assert len(result["matches"]) == 5
    assert case.search.call_count == 1
    assert case.search.call_args.kwargs == {"limit": 5, "equipment_id": None}
    assert case.search.call_args.args[1] == "Technician-reported problem:\nVacuum gripper loses suction under load"
    assert case.embedder.calls == [[result["query"]]]
    assert [issue["similarity"] for issue in result["matches"]] == sorted(
        (issue["similarity"] for issue in result["matches"]), reverse=True)
    case.factory.assert_called_with(offline=True)
    case.log_reader.assert_not_called()
    case.detector.assert_not_called()
    assert any("all registered equipment" in item.value for item in case.app.caption)
    assert any("not diagnostic confidence or probability" in item.value for item in case.app.caption)
    assert len(case.app.expander) == 5
    assert any("Status: Open" in item.value for item in case.app.text)
    assert any("do not establish a successful repair" in item.value for item in case.app.caption)
    assert any("applicability to this problem is unverified" in item.value for item in case.app.caption)
    assert any(item.value == "Recorded cup replacement" for item in case.app.text)
    with connect(case.path) as connection:
        after = MaintenanceStore(connection).list_issues(), SemanticSearch(MaintenanceStore(connection), case.embedder).index.read()
    assert before == after


def test_selected_equipment_filter_is_applied_before_top_five(case):
    submit(case, case.camera)
    search(case)
    assert [issue["id"] for issue in case.app.session_state[MATCHES_KEY]["matches"]] == [case.camera_issue]
    assert case.search.call_args.kwargs == {"limit": 5, "equipment_id": case.camera}
    assert any("submitted equipment only" in item.value for item in case.app.caption)


def test_empty_equipment_matches_never_broaden_scope_or_retry(case):
    submit(case, case.spare)
    search(case)
    assert case.app.session_state[MATCHES_KEY]["matches"] == []
    assert any("within the submitted equipment scope" in item.value for item in case.app.info)
    assert case.search.call_args.kwargs["equipment_id"] == case.spare
    case.app.run()
    assert not case.app.exception
    assert case.search.call_count == 1
    assert not case.embedder.calls


@pytest.mark.parametrize("mode", ["patterns", "no_patterns", "no_logs"])
def test_query_combines_description_with_actual_analyzed_log_evidence_when_available(case, mode):
    if mode != "no_logs":
        add_logs(case, patterns=mode == "patterns")
    submit(case, case.pump, "Technician reports a vacuum loss during transfer")
    analyze(case)
    search(case)
    query = case.app.session_state[MATCHES_KEY]["query"]
    assert "Technician reports a vacuum loss during transfer" in query
    assert "Unrelated camera secret" not in query
    if mode == "patterns":
        assert "PRESSURE_LOW" in query and "PICK_FAIL" in query
        assert "Pattern: repeated failures" in query
    elif mode == "no_patterns":
        assert "Observed vacuum pressure reading" in query
        assert "Pattern:" not in query
    else:
        assert "Recorded log evidence" not in query
    assert case.search.call_count == 1
    assert case.log_reader.call_count == case.detector.call_count == 1
    assert case.app.session_state[MATCHES_KEY]["matches"]


@pytest.mark.parametrize("mode", ["missing", "stale"])
def test_missing_or_stale_index_guides_to_existing_workflow_without_embedding_or_rebuild(case, mode):
    with connect(case.path) as connection:
        if mode == "missing":
            with connection:
                connection.execute("DELETE FROM semantic_index_metadata")
        else:
            MaintenanceStore(connection).update_issue(case.issues[0], status="Open", root_cause="Changed source")
    submit(case)
    search(case)
    assert MATCHES_KEY not in case.app.session_state
    assert any("Similar Issues" in item.value and "Rebuild / refresh search index" in item.value for item in case.app.warning)
    case.search.assert_not_called()
    assert not case.embedder.calls
    case.app.run()
    assert not case.app.exception
    case.search.assert_not_called()


def test_matches_persist_without_searching_on_form_edits_reruns_or_navigation(case):
    submit(case)
    search(case)
    saved = deepcopy(case.app.session_state[MATCHES_KEY])
    calls = len(case.embedder.calls)
    find(case.app.selectbox, "Equipment (optional)").set_value(case.camera)
    find(case.app.text_area, "Problem description *").input("Unsubmitted camera description")
    case.app.run()
    assert not case.app.exception
    assert case.app.session_state[MATCHES_KEY] == saved
    find(case.app.sidebar.radio, "Navigate").set_value("Home").run()
    find(case.app.button, "Troubleshoot a Problem").click().run()
    assert not case.app.exception
    assert case.app.session_state[MATCHES_KEY] == saved
    assert case.search.call_count == 1
    assert len(case.embedder.calls) == calls


@pytest.mark.parametrize("change", ["description", "equipment", "no_equipment"])
def test_new_submitted_draft_discards_matches_without_automatic_search(case, change):
    submit(case, case.pump)
    search(case)
    equipment_id = {"description": case.pump, "equipment": case.camera, "no_equipment": None}[change]
    description = "Changed vacuum symptom" if change == "description" else "Vacuum gripper loses suction under load"
    submit(case, equipment_id, description)
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert case.search.call_count == 1


@pytest.mark.parametrize("change", ["analyze", "reanalyze", "remove", "mutate", "failed_analysis"])
def test_log_snapshot_changes_invalidate_matches_without_automatic_search(case, change):
    add_logs(case)
    submit(case, case.pump)
    if change != "analyze":
        analyze(case)
    search(case)
    if change in ("analyze", "reanalyze"):
        analyze(case)
    elif change == "failed_analysis":
        case.detector.side_effect = ValueError("Invalid log timestamp")
        analyze(case)
    else:
        if change == "remove":
            del case.app.session_state[ANALYSIS_KEY]
        else:
            case.app.session_state[ANALYSIS_KEY]["logs"][0]["message"] = "Changed snapshot evidence"
        case.app.run()
        assert not case.app.exception
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert case.search.call_count == 1


@pytest.mark.parametrize("change", ["stale_history", "rebuilt_history", "rebuilt_index", "vectors"])
def test_history_or_index_changes_hide_matches_without_rerunning_search(case, change):
    submit(case)
    search(case)
    if change in ("stale_history", "rebuilt_history"):
        with connect(case.path) as connection:
            MaintenanceStore(connection).update_issue(case.issues[0], status="Resolved", root_cause="Changed finding")
    if change in ("rebuilt_history", "rebuilt_index"):
        case.rebuild()
    elif change == "vectors":
        with connect(case.path) as connection:
            with connection:
                connection.execute("UPDATE issue_embeddings SET vector=? WHERE issue_id=?",
                                   (np.ones(case.embedder.dimension, dtype="<f4").tobytes(), case.issues[0]))
    calls = len(case.embedder.calls)
    case.app.run()
    assert not case.app.exception
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert case.search.call_count == 1
    assert len(case.embedder.calls) == calls
    assert any("Rebuild / refresh search index" in item.value or "Earlier matches are hidden" in item.value
               for item in case.app.warning)


def test_failed_explicit_search_discards_old_matches_without_automatic_retry(case):
    submit(case)
    search(case)
    case.search.side_effect = EmbeddingError("Cached embedding model unavailable")
    search(case)
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert any("could not be completed" in item.value for item in case.app.error)
    case.app.run()
    assert not case.app.exception
    assert case.search.call_count == 2


@pytest.mark.parametrize("change", ["history", "snapshot"])
def test_context_changes_during_embedding_discard_the_response(case, change):
    submit(case, case.pump)

    def changed():
        if change == "history":
            with connect(case.path) as connection:
                MaintenanceStore(connection).update_issue(case.issues[0], status="Closed")
            case.rebuild()
        else:
            case.app.session_state[DRAFT_KEY] = {"description": "Changed during search", "equipment_id": case.pump}

    case.embedder.hook = changed
    search(case)
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert any("changed during retrieval" in item.value for item in case.app.warning) or case.app.error


def test_deleted_selected_equipment_never_falls_back_to_all_equipment(case):
    submit(case, case.pump)
    search(case)
    with connect(case.path) as connection:
        with connection:
            connection.execute("DELETE FROM issues WHERE equipment_id=?", (case.pump,))
            connection.execute("DELETE FROM equipment WHERE id=?", (case.pump,))
    case.app.run()
    assert not case.app.exception
    assert find(case.app.button, FIND).disabled
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert case.search.call_count == 1


def test_description_changes_ranking_with_identical_analyzed_logs(case):
    with connect(case.path) as connection:
        camera_on_pump = MaintenanceStore(connection).create_issue(equipment_id=case.pump,
            description="Camera image focus lens illumination", reporter="Demo", status="Open")
    case.rebuild()
    add_logs(case)
    submit(case, case.pump)
    analyze(case)
    search(case)
    assert case.app.session_state[MATCHES_KEY]["matches"][0]["id"] != camera_on_pump
    original_logs = deepcopy(case.app.session_state[ANALYSIS_KEY]["logs"])
    submit(case, case.pump, "Camera image focus lens illumination")
    assert MATCHES_KEY not in case.app.session_state
    analyze(case)
    assert case.app.session_state[ANALYSIS_KEY]["logs"] == original_logs
    search(case)
    assert case.app.session_state[MATCHES_KEY]["matches"][0]["id"] == camera_on_pump


def test_current_empty_index_returns_empty_matches_without_model_inference(case):
    with connect(case.path) as connection:
        with connection:
            connection.execute("DELETE FROM issues")
    case.rebuild()
    case.embedder.calls.clear()
    submit(case)
    search(case)
    assert case.app.session_state[MATCHES_KEY]["matches"] == []
    assert any("No historical issues were found across registered equipment" in item.value for item in case.app.info)
    assert not case.embedder.calls


def test_invalid_selected_equipment_cannot_search_even_if_disabled_button_is_bypassed(case, monkeypatch):
    from maintainai import guided_ui

    submit(case, case.pump)
    search(case)
    case.app.session_state[DRAFT_KEY] = {"description": "Valid description", "equipment_id": True}
    original = guided_ui.st.button

    def forced_click(label, *args, **kwargs):
        value = original(label, *args, **kwargs)
        return True if label == FIND else value

    monkeypatch.setattr(guided_ui.st, "button", forced_click)
    case.app.run()
    assert not case.app.exception
    assert find(case.app.button, FIND).disabled
    assert MATCHES_KEY not in case.app.session_state
    assert not case.app.expander
    assert case.search.call_count == 1
