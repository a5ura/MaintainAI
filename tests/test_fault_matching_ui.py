"""Streamlit integration with synthetic SQLite history and deterministic embeddings."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.embeddings import EmbeddingError
from maintainai.fault_matching import HISTORICAL_NOTICE, FaultMatchResult, match_fault_to_issues
from maintainai.semantic_search import SemanticSearch
from maintainai.service import MaintenanceStore, ValidationError


BUTTON = "Find Similar Historical Issues"


def find(elements, label):
    return next(element for element in elements if element.label == label)


def setup_app(tmp_path, monkeypatch, fake_embedder, *, history=True, indexed=True, log_mode="patterns"):
    path = tmp_path / "fault_matching_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    monkeypatch.setattr("maintainai.semantic_search.MiniLMEmbedder", lambda: fake_embedder)
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        robot = store.create_equipment(asset_tag="SYNTHETIC-VAC", name="Synthetic vacuum gripper", category="Robotics")
        conveyor = store.create_equipment(asset_tag="SYNTHETIC-CONV", name="Synthetic conveyor", category="Conveyor")
        issue_ids = []
        if history:
            for number in range(7):
                issue_ids.append(store.create_issue(
                    equipment_id=robot if number < 6 else conveyor,
                    description="Synthetic vacuum suction leak" if number < 6 else "Synthetic conveyor belt failure",
                    reporter="Synthetic tester", status="Closed" if number < 6 else "Open",
                    root_cause=f"Recorded synthetic seal wear {number}",
                    corrective_action=f"Recorded synthetic replacement {number}"))
        events = []
        if log_mode != "empty":
            events.append((robot, "12:00:00", "INFO", "START", "Synthetic startup"))
        if log_mode == "patterns":
            events.extend([
                (robot, "12:00:12", "WARNING", "PRESSURE_LOW", "Synthetic gripper vacuum pressure low"),
                (robot, "12:00:15", "ERROR", "PICK_FAIL", "Synthetic suction loss drops part"),
                (robot, "12:00:18", "ERROR", "PICK_FAIL", "Synthetic gripper pickup failed"),
                (conveyor, "12:01:00", "ERROR", "BELT_STOP", "Synthetic conveyor belt stopped"),
                (conveyor, "12:01:05", "ERROR", "BELT_STOP", "Synthetic conveyor belt stopped again"),
            ])
        for asset, time, severity, code, message in events:
            store.create_log(equipment_id=asset, timestamp=f"2026-10-25T{time}Z", severity=severity,
                             event_code=code, message=message)
        if history and indexed:
            SemanticSearch(store, fake_embedder).rebuild()
    matcher = Mock(wraps=match_fault_to_issues)
    monkeypatch.setattr("maintainai.fault_ui.match_fault_to_issues", matcher)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    find(app.radio, "Navigate").set_value("Fault Analysis").run()
    find(app.selectbox, "Equipment").set_value(robot).run()
    assert not app.exception
    return SimpleNamespace(app=app, path=path, robot=robot, conveyor=conveyor,
                           issue_ids=issue_ids, matcher=matcher, embedder=fake_embedder)


def analyze(case):
    find(case.app.button, "Analyze Logs").click().run()
    assert not case.app.exception


def match(case):
    find(case.app.button, BUTTON).click().run()
    assert not case.app.exception


def cache(case):
    return case.app.session_state["fault_analysis_result"].get("historical_matches")


def database_snapshot(path):
    with connect(path) as connection:
        store = MaintenanceStore(connection)
        return store.list_equipment(), store.list_issues(), store.list_logs(), SemanticSearch(store).index.read()


def test_matching_requires_findings_and_explicit_button(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    assert BUTTON not in [button.label for button in case.app.button]
    analyze(case)
    assert BUTTON in [button.label for button in case.app.button]
    case.app.run()
    case.matcher.assert_not_called()
    assert len(fake_embedder.calls) == 1  # Initial fixture index build only.
    assert not case.app.expander


@pytest.mark.parametrize("mode", ["empty", "no-patterns"])
def test_no_matching_button_without_detected_patterns(tmp_path, monkeypatch, fake_embedder, mode):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, log_mode=mode)
    analyze(case)
    assert BUTTON not in [button.label for button in case.app.button]
    case.matcher.assert_not_called()


def test_actual_matching_resolves_evidence_and_displays_ranked_original_records(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    before = database_snapshot(case.path)
    analyze(case)
    match(case)
    assert case.matcher.call_count == 2
    for call in case.matcher.call_args_list:
        finding, supporting, search = call.args
        assert [log["id"] for log in supporting] == finding["log_ids"]
        assert all(log["equipment_id"] == finding["equipment_id"] for log in supporting)
        assert isinstance(search, SemanticSearch)
    entries = cache(case)["entries"]
    assert len(case.app.expander) == 10
    expected_labels = []
    for entry in entries:
        records = entry["result"]["matches"]
        assert len(records) == 5
        assert [row["similarity"] for row in records] == sorted(
            (row["similarity"] for row in records), reverse=True)
        expected_labels.extend(f"{rank}. Issue #{row['issue_id']} | cosine similarity {row['similarity']:.3f}"
                               for rank, row in enumerate(records, 1))
    assert [item.label for item in case.app.expander] == expected_labels
    assert any("SYNTHETIC-VAC" in item.value and "Status: Closed" in item.value for item in case.app.text)
    for text in ("Synthetic vacuum suction leak", "Recorded synthetic seal wear 0", "Recorded synthetic replacement 0"):
        assert any(item.value == text for item in case.app.text)
    assert any("Recorded root cause" in item.value for item in case.app.markdown)
    assert any("Recorded corrective action" in item.value for item in case.app.markdown)
    assert any("Original record: Issues" in item.value for item in case.app.caption)
    assert any("not confidence percentages" in item.value for item in case.app.caption)
    assert any("Similarity does not establish the current root cause" in item.value for item in case.app.caption)
    assert database_snapshot(case.path) == before


def test_all_equipment_matches_each_finding_separately(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    find(case.app.selectbox, "Equipment").set_value(None).run()
    analyze(case)
    match(case)
    assert case.matcher.call_count == 3
    assert {call.args[0]["equipment_id"] for call in case.matcher.call_args_list} == {case.robot, case.conveyor}
    for call in case.matcher.call_args_list:
        finding, supporting, _ = call.args
        assert all(log["equipment_id"] == finding["equipment_id"] for log in supporting)


def test_matches_survive_reruns_and_navigation_without_searching_again(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    saved = deepcopy(cache(case))
    labels = [item.label for item in case.app.expander]
    calls = len(fake_embedder.calls)
    case.app.run()
    find(case.app.radio, "Navigate").set_value("Dashboard").run()
    find(case.app.radio, "Navigate").set_value("Fault Analysis").run()
    assert not case.app.exception
    assert cache(case) == saved
    assert [item.label for item in case.app.expander] == labels
    assert case.matcher.call_count == 2
    assert len(fake_embedder.calls) == calls


def test_equipment_change_clears_matches_and_needs_new_analysis(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    find(case.app.selectbox, "Equipment").set_value(case.conveyor).run()
    assert not case.app.expander
    assert BUTTON not in [button.label for button in case.app.button]
    analyze(case)
    assert cache(case) is None
    assert not case.app.expander
    assert case.matcher.call_count == 2
    match(case)
    assert case.matcher.call_count == 3


@pytest.mark.parametrize("failed", [False, True])
def test_new_analysis_discards_matches_even_if_analysis_fails(tmp_path, monkeypatch, fake_embedder, failed):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    if failed:
        monkeypatch.setattr("maintainai.fault_ui.detect_fault_patterns", Mock(side_effect=ValueError("Synthetic invalid event")))
    analyze(case)
    assert not case.app.expander
    assert case.matcher.call_count == 2
    if not failed:
        assert cache(case) is None


def test_no_historical_issues_is_informative_and_does_not_embed(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, history=False)
    analyze(case)
    match(case)
    assert any("No historical maintenance issues" in item.value for item in case.app.info)
    assert not case.app.expander
    assert fake_embedder.calls == []
    case.matcher.assert_not_called()


def test_empty_matches_are_retained_without_automatic_retry(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    case.matcher.side_effect = lambda finding, logs, search: FaultMatchResult(
        "Synthetic query", [], finding["log_ids"], [], HISTORICAL_NOTICE)
    analyze(case)
    match(case)
    assert sum("No historical issues were retrieved" in item.value for item in case.app.info) == 2
    assert not case.app.expander
    case.app.run()
    assert case.matcher.call_count == 2


@pytest.mark.parametrize("missing_index", [False, True])
def test_missing_or_stale_index_directs_to_existing_rebuild_workflow(tmp_path, monkeypatch, fake_embedder, missing_index):
    case = setup_app(tmp_path, monkeypatch, fake_embedder, indexed=not missing_index)
    if not missing_index:
        with connect(case.path) as connection:
            MaintenanceStore(connection).update_issue(case.issue_ids[0], status="Open")
    calls = len(fake_embedder.calls)
    before = database_snapshot(case.path)
    analyze(case)
    match(case)
    assert any("Open Similar Issues" in item.value and "Rebuild / refresh search index" in item.value
               for item in case.app.warning)
    assert cache(case) is None
    assert not case.app.expander
    case.matcher.assert_not_called()
    assert len(fake_embedder.calls) == calls
    assert database_snapshot(case.path) == before


@pytest.mark.parametrize("rebuild_after_edit", [False, True])
def test_cached_matches_are_hidden_after_history_changes_even_if_rebuilt(tmp_path, monkeypatch, fake_embedder, rebuild_after_edit):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    with connect(case.path) as connection:
        MaintenanceStore(connection).update_issue(case.issue_ids[0], status="Open", root_cause="New synthetic finding")
        if rebuild_after_edit:
            SemanticSearch(MaintenanceStore(connection), fake_embedder).rebuild()
    calls = len(fake_embedder.calls)
    case.app.run()
    assert not case.app.exception
    assert not case.app.expander
    assert cache(case) is None
    assert any("Earlier historical matches are hidden" in item.value for item in case.app.warning)
    assert case.matcher.call_count == 2
    assert len(fake_embedder.calls) == calls


def test_history_edit_between_findings_discards_partial_results(tmp_path, monkeypatch, fake_embedder):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    count = 0
    def edit_after_first(finding, logs, search):
        nonlocal count
        result = match_fault_to_issues(finding, logs, search)
        count += 1
        if count == 1:
            with connect(case.path) as connection:
                MaintenanceStore(connection).update_issue(case.issue_ids[0], status="Open")
        return result
    case.matcher.side_effect = edit_after_first
    analyze(case)
    match(case)
    assert cache(case) is None
    assert not case.app.expander
    assert any("Rebuild / refresh search index" in item.value for item in case.app.warning)


@pytest.mark.parametrize("error", [EmbeddingError("Synthetic local model unavailable"),
                                  ValidationError("Synthetic invalid supporting log")])
def test_failed_explicit_search_replaces_old_results_and_survives_reruns(tmp_path, monkeypatch, fake_embedder, error):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    case.matcher.side_effect = error
    match(case)
    assert not case.app.expander
    assert sum("Historical matching unavailable" in item.value for item in case.app.error) == 2
    assert any(str(error) == item.value for item in case.app.text)
    case.app.run()
    assert not case.app.exception
    assert case.matcher.call_count == 4
    assert not case.app.expander


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_supporting_logs_are_reported_without_crashing(tmp_path, monkeypatch, fake_embedder, all_missing):
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    result = deepcopy(case.app.session_state["fault_analysis_result"])
    # Simulate an incomplete snapshot, retaining the unrelated startup log.
    result["logs"] = [log for log in result["logs"] if log["id"] == 1 or (not all_missing and log["id"] != 2)]
    case.app.session_state["fault_analysis_result"] = result
    calls = len(fake_embedder.calls)
    match(case)
    assert any("Missing supporting log IDs" in item.value for item in case.app.warning)
    assert not case.app.exception
    if all_missing:
        assert not case.app.expander
        assert len(fake_embedder.calls) == calls
        assert any("Supporting evidence logs are unavailable" in item.value for item in case.app.info)


def test_matching_never_initializes_a_cloud_provider(tmp_path, monkeypatch, fake_embedder):
    provider = Mock(side_effect=AssertionError("Cloud provider must not be initialized"))
    monkeypatch.setattr("maintainai.rag.provider_from_environment", provider)
    case = setup_app(tmp_path, monkeypatch, fake_embedder)
    analyze(case)
    match(case)
    assert case.app.expander
    provider.assert_not_called()
