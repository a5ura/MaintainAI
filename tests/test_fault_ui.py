from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from maintainai.database import PROJECT_ROOT, connect, initialize
from maintainai.fault_detection import detect_fault_patterns
from maintainai.service import MaintenanceStore


def find(elements, label):
    return next(element for element in elements if element.label == label)


def setup_app(tmp_path, monkeypatch, *, mode="patterns", with_equipment=True):
    path = tmp_path / "fault_ui.db"
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(path))
    ids = []
    with connect(path) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        if with_equipment:
            for tag, name in (("FAULT-A", "Synthetic robot"), ("FAULT-B", "Synthetic conveyor"), ("FAULT-C", "Synthetic spare")):
                ids.append(store.create_equipment(asset_tag=tag, name=name, category="Synthetic equipment"))
        if mode == "patterns":
            rows = [(ids[0], "12:00:00", "INFO", "START"),
                    (ids[0], "12:00:12", "WARNING", "PRESSURE_LOW"),
                    (ids[0], "12:00:15", "ERROR", "PICK_FAIL"),
                    (ids[0], "12:00:18", "ERROR", "PICK_FAIL"),
                    (ids[1], "12:01:00", "ERROR", "CONVEYOR_STOP"),
                    (ids[1], "12:01:05", "CRITICAL", "CONVEYOR_STOP")]
        elif mode == "no-patterns":
            rows = [(ids[0], "12:00:00", "INFO", "START"),
                    (ids[0], "12:00:10", "WARNING", "WARN"),
                    (ids[0], "12:03:20", "ERROR", "OTHER")]
        else:
            rows = []
        for equipment_id, time, severity, code in rows:
            store.create_log(equipment_id=equipment_id, timestamp=f"2026-10-25T{time}Z", severity=severity,
                             event_code=code, message=f"Synthetic event {code}", source="Demo controller")
    detector = Mock(wraps=detect_fault_patterns)
    monkeypatch.setattr("maintainai.fault_ui.detect_fault_patterns", detector)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()
    assert "Fault Analysis" in find(app.radio, "Navigate").options
    find(app.radio, "Navigate").set_value("Fault Analysis").run()
    return app, path, ids, detector


def metrics(app):
    return {item.label: item.value for item in app.metric}


def analyze(app):
    find(app.button, "Analyze Logs").click().run()
    assert not app.exception


def test_fault_page_navigation_initial_state_and_no_automatic_analysis(tmp_path, monkeypatch):
    app, _, _, detector = setup_app(tmp_path, monkeypatch)
    assert not app.exception
    assert any(item.value == "Fault Analysis" for item in app.header)
    assert find(app.selectbox, "Equipment").options[0] == "All Equipment"
    assert len(find(app.selectbox, "Equipment").options) == 4
    assert any("potential correlations" in item.value and "not confirmed root causes" in item.value for item in app.warning)
    assert any("WARNING" in item.value and "CRITICAL" in item.value for item in app.caption)
    assert any("click Analyze Logs" in item.value for item in app.info)
    assert not app.metric
    assert not app.dataframe
    app.run()
    detector.assert_not_called()


@pytest.mark.parametrize("with_equipment", [False, True])
def test_fault_page_empty_logs_is_safe_and_informative(tmp_path, monkeypatch, with_equipment):
    app, _, _, detector = setup_app(tmp_path, monkeypatch, mode="empty", with_equipment=with_equipment)
    if not with_equipment:
        assert any("No equipment registered" in item.value for item in app.info)
    analyze(app)
    assert metrics(app) == {"Logs examined": "0", "Findings": "0"}
    assert any("No logs found for this selection" in item.value for item in app.info)
    assert not app.dataframe
    detector.assert_called_once_with([])


def test_fault_page_successful_detection_metadata_and_unique_evidence(tmp_path, monkeypatch):
    app, path, _, detector = setup_app(tmp_path, monkeypatch)
    with connect(path) as connection:
        store = MaintenanceStore(connection)
        before_logs, before_equipment = store.list_logs(), store.list_equipment()
    analyze(app)
    assert metrics(app) == {"Logs examined": "6", "Findings": "3"}
    assert any("warning_to_error" in item.value for item in app.subheader)
    assert sum("repeated_failures" in item.value for item in app.subheader) == 2
    assert any("Supporting log IDs: 2, 3, 4" == item.value for item in app.caption)
    assert any("Start (UTC): 2026-10-25T12:00:12.000000+00:00" in item.value for item in app.text)
    assert any("End (UTC): 2026-10-25T12:00:18.000000+00:00" in item.value for item in app.text)
    assert any("not a confirmed root cause" in item.value for item in app.text)
    assert any("CRITICAL" in item.value and item.proto.icon == "🚨" for item in app.error)
    assert any("ERROR" in item.value and item.proto.icon == "🔴" for item in app.error)
    frame = app.dataframe[0].value
    assert list(frame.columns) == ["Log ID", "Timestamp (UTC)", "Asset Tag", "Severity", "Event Code", "Message"]
    assert set(frame["Log ID"]) == {2, 3, 4, 5, 6}
    assert len(frame) == len(set(frame["Log ID"]))
    assert {"🟡 WARNING", "🔴 ERROR", "🟣 CRITICAL"} <= set(frame["Severity"])
    assert all(value.endswith("+00:00") for value in frame["Timestamp (UTC)"])
    assert set(frame["Asset Tag"]) == {"FAULT-A", "FAULT-B"}
    assert all(value.startswith("Synthetic event") for value in frame["Message"])
    detector.assert_called_once()
    with connect(path) as connection:
        store = MaintenanceStore(connection)
        assert store.list_logs() == before_logs
        assert store.list_equipment() == before_equipment


def test_equipment_filter_clears_results_and_needs_explicit_analysis(tmp_path, monkeypatch):
    app, _, ids, detector = setup_app(tmp_path, monkeypatch)
    analyze(app)
    find(app.selectbox, "Equipment").set_value(ids[0]).run()
    assert not app.metric and not app.dataframe
    assert any("click Analyze Logs" in item.value for item in app.info)
    assert detector.call_count == 1
    analyze(app)
    assert metrics(app) == {"Logs examined": "4", "Findings": "2"}
    assert set(app.dataframe[0].value["Asset Tag"]) == {"FAULT-A"}
    assert {row["equipment_id"] for row in detector.call_args.args[0]} == {ids[0]}
    find(app.selectbox, "Equipment").set_value(ids[1]).run()
    assert not app.metric and not app.dataframe
    analyze(app)
    assert metrics(app) == {"Logs examined": "2", "Findings": "1"}
    assert set(app.dataframe[0].value["Asset Tag"]) == {"FAULT-B"}
    find(app.selectbox, "Equipment").set_value(ids[2]).run()
    assert not app.metric
    analyze(app)
    assert metrics(app) == {"Logs examined": "0", "Findings": "0"}
    find(app.selectbox, "Equipment").set_value(None).run()
    assert not app.metric
    analyze(app)
    assert metrics(app) == {"Logs examined": "6", "Findings": "3"}
    assert detector.call_count == 5


def test_no_findings_does_not_claim_equipment_is_healthy(tmp_path, monkeypatch):
    app, _, _, detector = setup_app(tmp_path, monkeypatch, mode="no-patterns")
    analyze(app)
    assert metrics(app) == {"Logs examined": "3", "Findings": "0"}
    assert any("No supported fault patterns" in item.value and "does not establish" in item.value for item in app.info)
    assert len(app.dataframe[0].value) == 3
    assert set(app.dataframe[0].value["Severity"]) == {"⚪ INFO", "🟡 WARNING", "🔴 ERROR"}
    detector.assert_called_once()


def test_results_and_selection_survive_reruns_and_navigation(tmp_path, monkeypatch):
    app, _, ids, detector = setup_app(tmp_path, monkeypatch)
    find(app.selectbox, "Equipment").set_value(ids[0]).run()
    analyze(app)
    frame = app.dataframe[0].value.copy()
    app.run()
    assert not app.exception
    assert metrics(app) == {"Logs examined": "4", "Findings": "2"}
    pd.testing.assert_frame_equal(app.dataframe[0].value, frame)
    detector.assert_called_once()
    find(app.radio, "Navigate").set_value("Dashboard").run()
    find(app.radio, "Navigate").set_value("Fault Analysis").run()
    assert not app.exception
    assert find(app.selectbox, "Equipment").value == ids[0]
    assert metrics(app) == {"Logs examined": "4", "Findings": "2"}
    pd.testing.assert_frame_equal(app.dataframe[0].value, frame)
    detector.assert_called_once()


def test_snapshot_does_not_reanalyze_new_logs_until_button_is_pressed(tmp_path, monkeypatch):
    app, path, ids, detector = setup_app(tmp_path, monkeypatch)
    analyze(app)
    with connect(path) as connection:
        MaintenanceStore(connection).create_log(equipment_id=ids[0], timestamp="2026-10-25T12:00:20Z",
                                               severity="ERROR", event_code="PICK_FAIL", message="Synthetic new log")
    app.run()
    assert metrics(app) == {"Logs examined": "6", "Findings": "3"}
    detector.assert_called_once()
    assert any("last explicit run" in item.value and "Press Analyze Logs again" in item.value for item in app.caption)
    analyze(app)
    assert metrics(app) == {"Logs examined": "7", "Findings": "3"}
    assert detector.call_count == 2


def test_failed_explicit_analysis_discards_previous_results(tmp_path, monkeypatch):
    app, _, _, detector = setup_app(tmp_path, monkeypatch)
    analyze(app)
    detector.side_effect = ValueError("Synthetic invalid timestamp")
    find(app.button, "Analyze Logs").click().run()
    assert not app.exception
    assert any("Fault analysis could not be completed" in item.value for item in app.error)
    assert not app.metric and not app.dataframe
    app.run()
    assert not app.metric and not app.dataframe
    assert detector.call_count == 2


def test_database_change_clears_a_previous_snapshot(tmp_path, monkeypatch):
    app, _, ids, detector = setup_app(tmp_path, monkeypatch)
    find(app.selectbox, "Equipment").set_value(ids[0]).run()
    analyze(app)
    other = tmp_path / "other.db"
    with connect(other) as connection:
        initialize(connection)
        store = MaintenanceStore(connection)
        equipment_id = store.create_equipment(asset_tag="OTHER", name="Synthetic other asset", category="Test")
        store.create_log(equipment_id=equipment_id, timestamp="2026-10-25T12:00:00Z", severity="INFO", message="Synthetic other log")
    monkeypatch.setenv("MAINTAINAI_DB_PATH", str(other))
    app.run()
    assert not app.exception
    assert not app.metric and not app.dataframe
    assert find(app.selectbox, "Equipment").value is None
    detector.assert_called_once()
    analyze(app)
    assert metrics(app) == {"Logs examined": "1", "Findings": "0"}
    assert detector.call_count == 2
